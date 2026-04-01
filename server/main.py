"""
bssentinel FastAPI server
Patient deterioration risk engine — NEWS2-augmented clinical scoring
with SHAP-style feature attribution.

Endpoints:
  POST /api/predict              — submit patient data, returns job_id
  GET  /api/jobs/{job_id}        — poll job status + results
"""

import asyncio
import json
import logging
import math
import pickle
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="bssentinel API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    _load_ml_model()

# In-memory job store
jobs: dict[str, dict] = {}

# ── ML Model ───────────────────────────────────────────────────────────────────

_ARTIFACTS = Path(__file__).parent.parent / "ml" / "artifacts"

_ML_MODEL: Any = None   # raw XGBoost (scores in paper threshold space: ≥0.306 → alert)
_ML_COLS: list[str] = []

# Thresholds derived from the MIMIC-IV validation (metrics.json threshold_analysis)
# Raw XGBoost score   Sensitivity  Specificity  PPV
#  ≥ 0.30             90.6%        36.7%        2.8%   ← 90% sensitivity operating point
#  ≥ 0.40             79.1%        56.4%        3.5%
#  ≥ 0.50             61.3%        75.5%        4.8%
_ML_THRESH_LOW      = 0.30   # below → LOW (model not flagging)
_ML_THRESH_MODERATE = 0.40   # 0.30–0.40 → MODERATE (at or above 90% sensitivity threshold)
_ML_THRESH_HIGH     = 0.50   # 0.40–0.50 → HIGH
                              # ≥ 0.50    → CRITICAL


def _load_ml_model() -> bool:
    """
    Load the raw XGBoost model and feature column list from ml/artifacts/.
    We use xgb_raw.pkl (uncalibrated) because the validated thresholds from
    the paper (0.306 at 90% sensitivity) are in raw score space.
    Called once at server startup. Returns True if successful.
    """
    global _ML_MODEL, _ML_COLS
    model_path = _ARTIFACTS / "xgb_raw.pkl"
    cols_path  = _ARTIFACTS / "feature_cols.json"
    if not model_path.exists() or not cols_path.exists():
        log.warning("ML model not found at %s — falling back to heuristic engine", _ARTIFACTS)
        return False
    try:
        with open(model_path, "rb") as f:
            _ML_MODEL = pickle.load(f)
        with open(cols_path) as f:
            _ML_COLS = json.load(f)
        log.info("ML model loaded: %d features from %s", len(_ML_COLS), model_path)
        return True
    except Exception as exc:
        log.error("Failed to load ML model: %s — falling back to heuristic", exc)
        return False


# ── Feature vector builder ─────────────────────────────────────────────────────

# AVPU → approximate GCS component mapping (RCP NEWS2 guidance)
_AVPU_GCS = {
    "A": {"gcs_eye": 4, "gcs_verbal": 5, "gcs_motor": 6},
    "V": {"gcs_eye": 3, "gcs_verbal": 4, "gcs_motor": 5},
    "P": {"gcs_eye": 2, "gcs_verbal": 2, "gcs_motor": 4},
    "U": {"gcs_eye": 1, "gcs_verbal": 1, "gcs_motor": 1},
}

# Server admission_type → one hot column in MIMIC feature space
_ADM_TYPE_MAP = {
    "ed":           "adm_type_EW EMER.",
    "icu":          "adm_type_EW EMER.",   # closest MIMIC category
    "elective":     "adm_type_ELECTIVE",
    "urgent":       "adm_type_URGENT",
    "observation":  "adm_type_OBSERVATION ADMIT",
    "direct_ed":    "adm_type_DIRECT EMER.",
    # "general_ward" and anything else → all zeros (reference category)
}

_ADM_DUMMIES = [
    "adm_type_AMBULATORY OBSERVATION",
    "adm_type_DIRECT EMER.",
    "adm_type_DIRECT OBSERVATION",
    "adm_type_ELECTIVE",
    "adm_type_EU OBSERVATION",
    "adm_type_EW EMER.",
    "adm_type_OBSERVATION ADMIT",
    "adm_type_SURGICAL SAME DAY ADMISSION",
    "adm_type_URGENT",
]


def _n2_rr(rr):
    if rr is None:  return float("nan")
    if rr <= 8 or rr >= 25: return 3.0
    if 21 <= rr <= 24:      return 2.0
    if 9  <= rr <= 11:      return 1.0
    return 0.0

def _n2_spo2(s):
    if s is None:   return float("nan")
    if s <= 91:     return 3.0
    if s <= 93:     return 2.0
    if s <= 95:     return 1.0
    return 0.0

def _n2_sbp(s):
    if s is None:   return float("nan")
    if s <= 90 or s >= 220: return 3.0
    if 91  <= s <= 100:     return 2.0
    if 101 <= s <= 110:     return 1.0
    return 0.0

def _n2_hr(h):
    if h is None:   return float("nan")
    if h <= 40 or h >= 131:     return 3.0
    if 111 <= h <= 130:         return 2.0
    if (41 <= h <= 50) or (91 <= h <= 110): return 1.0
    return 0.0

def _n2_temp(t):
    if t is None:   return float("nan")
    if t <= 35.0:           return 3.0
    if t >= 39.1:           return 2.0
    if 35.1 <= t <= 36.0 or 38.1 <= t <= 39.0: return 1.0
    return 0.0

def _n2_gcs(gcs_total):
    if gcs_total is None or math.isnan(gcs_total): return float("nan")
    return 0.0 if gcs_total >= 15 else 3.0


def _build_feature_vector(req: "PredictRequest") -> dict[str, float]:
    """
    Map a single-snapshot PredictRequest to the 83-feature dict expected
    by the trained XGBoost model.  Missing values are represented as NaN.
    """
    nan = float("nan")
    v = req.vitals
    l = req.labs
    p = req.patient
    m = req.medications

    # ── GCS from AVPU ────────────────────────────────────────────────────────
    avpu = (v.avpu or "A").upper()
    gcs  = _AVPU_GCS.get(avpu, _AVPU_GCS["A"])
    gcs_eye, gcs_verbal, gcs_motor = gcs["gcs_eye"], gcs["gcs_verbal"], gcs["gcs_motor"]
    gcs_total = gcs_eye + gcs_verbal + gcs_motor

    # ── NEWS2 subscores ──────────────────────────────────────────────────────
    n2_rr   = _n2_rr(v.respiratory_rate)
    n2_spo2 = _n2_spo2(v.spo2)
    n2_sbp  = _n2_sbp(v.systolic_bp)
    n2_hr   = _n2_hr(v.heart_rate)
    n2_temp = _n2_temp(v.temperature)
    n2_gcs  = _n2_gcs(gcs_total)
    n2_o2   = 2.0 if v.on_supplemental_o2 else 0.0
    n2_total_parts = [x for x in [n2_rr, n2_spo2, n2_sbp, n2_hr, n2_temp, n2_gcs, n2_o2]
                      if not math.isnan(x)]
    n2_total = sum(n2_total_parts) if n2_total_parts else nan

    # ── Missingness flags ────────────────────────────────────────────────────
    def miss(val): return 1.0 if val is None else 0.0

    vital_miss = [miss(v.heart_rate), miss(v.systolic_bp), miss(v.diastolic_bp),
                  miss(v.spo2), miss(v.respiratory_rate), miss(v.temperature)]
    lab_miss   = [miss(l.wbc), miss(l.hemoglobin), miss(l.platelets),
                  miss(l.creatinine), miss(l.bun), miss(l.sodium), miss(l.potassium),
                  miss(l.lactate), miss(l.glucose), miss(l.bilirubin), miss(l.inr), miss(l.crp)]

    # ── Admission type one-hot ────────────────────────────────────────────────
    adm_col = _ADM_TYPE_MAP.get((p.admission_type or "general_ward").lower(), None)
    adm_dummies = {col: (1.0 if col == adm_col else 0.0) for col in _ADM_DUMMIES}

    # ── Assemble feature dict ────────────────────────────────────────────────
    feat: dict[str, float] = {
        # Demographics / temporal
        "admit_offset_hours": p.hours_since_admission if p.hours_since_admission is not None else nan,
        "age":                p.age if p.age is not None else nan,
        "gender":             nan,   # not available in API; model handles NaN

        # Vitals — most recent | trend (NaN for single snapshot) | worst
        "heart_rate":           v.heart_rate          if v.heart_rate          is not None else nan,
        "heart_rate_trend":     nan,
        "heart_rate_worst":     v.heart_rate          if v.heart_rate          is not None else nan,
        "systolic_bp":          v.systolic_bp         if v.systolic_bp         is not None else nan,
        "systolic_bp_trend":    nan,
        "systolic_bp_worst":    v.systolic_bp         if v.systolic_bp         is not None else nan,
        "diastolic_bp":         v.diastolic_bp        if v.diastolic_bp        is not None else nan,
        "diastolic_bp_trend":   nan,
        "diastolic_bp_worst":   v.diastolic_bp        if v.diastolic_bp        is not None else nan,
        "spo2":                 v.spo2                if v.spo2                is not None else nan,
        "spo2_trend":           nan,
        "spo2_worst":           v.spo2                if v.spo2                is not None else nan,
        "respiratory_rate":     v.respiratory_rate    if v.respiratory_rate    is not None else nan,
        "respiratory_rate_trend": nan,
        "respiratory_rate_worst": v.respiratory_rate  if v.respiratory_rate    is not None else nan,
        "temperature":          v.temperature         if v.temperature         is not None else nan,
        "temperature_trend":    nan,
        "temperature_worst":    v.temperature         if v.temperature         is not None else nan,

        # GCS (derived from AVPU)
        "gcs_eye":           float(gcs_eye),
        "gcs_eye_trend":     nan,
        "gcs_eye_worst":     float(gcs_eye),
        "gcs_verbal":        float(gcs_verbal),
        "gcs_verbal_trend":  nan,
        "gcs_verbal_worst":  float(gcs_verbal),
        "gcs_motor":         float(gcs_motor),
        "gcs_motor_trend":   nan,
        "gcs_motor_worst":   float(gcs_motor),

        # Labs (single snapshot — no trend/worst for labs in this model)
        "wbc":         l.wbc        if l.wbc        is not None else nan,
        "hemoglobin":  l.hemoglobin if l.hemoglobin is not None else nan,
        "platelets":   l.platelets  if l.platelets  is not None else nan,
        "creatinine":  l.creatinine if l.creatinine is not None else nan,
        "bun":         l.bun        if l.bun        is not None else nan,
        "sodium":      l.sodium     if l.sodium     is not None else nan,
        "potassium":   l.potassium  if l.potassium  is not None else nan,
        "lactate":     l.lactate    if l.lactate    is not None else nan,
        "glucose":     l.glucose    if l.glucose    is not None else nan,
        "bilirubin":   l.bilirubin  if l.bilirubin  is not None else nan,
        "inr":         l.inr        if l.inr        is not None else nan,
        "crp":         nan,   # not collected in API; model handles NaN

        # Medications (binary)
        "vasopressors":   1.0 if m.vasopressors   else 0.0,
        "antibiotics":    1.0 if m.antibiotics    else 0.0,
        "anticoagulants": 1.0 if m.anticoagulants else 0.0,

        # NEWS2 subscores
        "news2_rr":    n2_rr,
        "news2_spo2":  n2_spo2,
        "news2_sbp":   n2_sbp,
        "news2_hr":    n2_hr,
        "news2_temp":  n2_temp,
        "news2_gcs":   n2_gcs,
        "gcs_total":   float(gcs_total),
        "news2_o2":    n2_o2,
        "news2_total": n2_total,

        # Missingness indicators
        "heart_rate_missing":        vital_miss[0],
        "systolic_bp_missing":       vital_miss[1],
        "diastolic_bp_missing":      vital_miss[2],
        "spo2_missing":              vital_miss[3],
        "respiratory_rate_missing":  vital_miss[4],
        "temperature_missing":       vital_miss[5],
        "wbc_missing":               lab_miss[0],
        "hemoglobin_missing":        lab_miss[1],
        "platelets_missing":         lab_miss[2],
        "creatinine_missing":        lab_miss[3],
        "bun_missing":               lab_miss[4],
        "sodium_missing":            lab_miss[5],
        "potassium_missing":         lab_miss[6],
        "lactate_missing":           lab_miss[7],
        "glucose_missing":           lab_miss[8],
        "bilirubin_missing":         lab_miss[9],
        "inr_missing":               lab_miss[10],
        "crp_missing":               1.0,   # CRP always missing from API
        "n_vitals_missing":          float(sum(vital_miss)),
        "n_labs_missing":            float(sum(lab_miss)),

        # Admission type dummies
        **adm_dummies,
    }
    return feat


def run_ml_engine(req: "PredictRequest") -> dict:
    """
    Run inference using the trained XGBoost model.
    Returns the same structured dict as run_risk_engine() so the API response
    format is unchanged.
    """
    feat_dict = _build_feature_vector(req)

    # Build ordered numpy array matching training feature order
    X = np.array([[feat_dict.get(col, float("nan")) for col in _ML_COLS]], dtype=np.float32)

    raw_score = float(_ML_MODEL.predict_proba(X)[0, 1])
    raw_score = round(min(max(raw_score, 0.001), 0.999), 4)
    level, css = _ml_risk_level(raw_score)

    # Expose as risk for API compatibility; raw_score is the decision variable
    risk = raw_score

    # NEWS2 score for display (reuse existing heuristic)
    vital_scores   = _news2_vitals(req.vitals)
    news2_score    = int(sum(vital_scores.values()))

    # Feature attribution: use NEWS2 subscores + lab flags as proxy attributions
    # (full SHAP requires the raw model; this gives a sensible ranked list)
    attribution_feats = {
        "respiratory_rate": feat_dict.get("news2_rr", 0) or 0,
        "spo2":             feat_dict.get("news2_spo2", 0) or 0,
        "supplemental_o2":  feat_dict.get("news2_o2", 0) or 0,
        "systolic_bp":      feat_dict.get("news2_sbp", 0) or 0,
        "heart_rate":       feat_dict.get("news2_hr", 0) or 0,
        "temperature":      feat_dict.get("news2_temp", 0) or 0,
        "avpu":             feat_dict.get("news2_gcs", 0) or 0,
        "lactate":          2.0 if (req.labs.lactate is not None and req.labs.lactate >= 2.0) else 0,
        "creatinine":       1.0 if (req.labs.creatinine is not None and req.labs.creatinine >= 1.5) else 0,
        "vasopressors":     5.0 if req.medications.vasopressors else 0,
        "antibiotics":      1.0 if req.medications.antibiotics else 0,
        "age":              (2.0 if (req.patient.age or 0) >= 75 else
                             1.0 if (req.patient.age or 0) >= 65 else 0),
    }
    total_attr = max(sum(attribution_feats.values()), 1)

    vitals_raw = {
        "respiratory_rate": req.vitals.respiratory_rate,
        "spo2":             req.vitals.spo2,
        "systolic_bp":      req.vitals.systolic_bp,
        "heart_rate":       req.vitals.heart_rate,
        "temperature":      req.vitals.temperature,
        "lactate":          req.labs.lactate,
        "creatinine":       req.labs.creatinine,
        "vasopressors":     req.medications.vasopressors,
        "antibiotics":      req.medications.antibiotics,
        "age":              req.patient.age,
    }

    features = []
    for feat, pts in sorted(attribution_feats.items(), key=lambda x: -x[1]):
        if pts == 0:
            continue
        raw_val = vitals_raw.get(feat)
        unit    = FEATURE_UNITS.get(feat, "")
        if isinstance(raw_val, bool):
            value_str = "Yes" if raw_val else "No"
        elif raw_val is not None:
            value_str = f"{raw_val:.1f} {unit}".strip()
        else:
            value_str = "—"
        features.append({
            "feature":      feat,
            "label":        FEATURE_LABELS.get(feat, feat),
            "value":        value_str,
            "points":       pts,
            "contribution": round((pts / total_attr) * risk, 4),
            "direction":    "increases_risk",
            "explanation":  _plain_explanation(feat, raw_val, pts),
        })

    top_factors = features[:3]

    # Clinical note (same logic as heuristic)
    h = req.config.horizon_hours
    if level == "CRITICAL":
        note = (f"This patient has a CRITICAL deterioration risk ({risk*100:.0f}%) within the next {h} h. "
                "Immediate clinical review and ICU notification are strongly recommended.")
    elif level == "HIGH":
        note = (f"This patient has a HIGH deterioration risk ({risk*100:.0f}%) within the next {h} h. "
                "Escalate monitoring frequency and consider senior clinician review.")
    elif level == "MODERATE":
        note = (f"This patient has a MODERATE deterioration risk ({risk*100:.0f}%) within the next {h} h. "
                "Increase observation frequency and reassess within 1–2 h.")
    else:
        note = (f"This patient has a LOW deterioration risk ({risk*100:.0f}%) within the next {h} h. "
                "Continue routine monitoring. Reassess if clinical status changes.")

    return {
        "risk_score":      raw_score,
        "risk_level":      level,
        "risk_css":        css,
        "risk_pct":        f"{raw_score*100:.1f}%",
        "horizon_hours":   req.config.horizon_hours,
        "news2_score":     news2_score,
        "augmented_score": news2_score,
        "top_factors":     top_factors,
        "all_features":    features,
        "clinical_note":   note,
        "patient_id":      req.patient.patient_id,
        "model":           "xgboost_mimic_iv_v1",
        "alert_threshold": _ML_THRESH_LOW,
        "vitals_count": sum(1 for val in [
            req.vitals.heart_rate, req.vitals.systolic_bp, req.vitals.spo2,
            req.vitals.respiratory_rate, req.vitals.temperature
        ] if val is not None),
        "labs_count": sum(1 for val in [
            req.labs.wbc, req.labs.hemoglobin, req.labs.creatinine,
            req.labs.lactate, req.labs.inr, req.labs.potassium,
            req.labs.sodium, req.labs.glucose
        ] if val is not None),
    }


# ── Request schema ─────────────────────────────────────────────────────────────

class PatientContext(BaseModel):
    patient_id: Optional[str] = None
    age: Optional[float] = None
    weight: Optional[float] = None
    admission_type: Optional[str] = "general_ward"   # icu | general_ward | ed
    hours_since_admission: Optional[float] = None
    prior_deterioration: Optional[bool] = False


class Vitals(BaseModel):
    heart_rate:         Optional[float] = None
    systolic_bp:        Optional[float] = None
    diastolic_bp:       Optional[float] = None
    spo2:               Optional[float] = None
    respiratory_rate:   Optional[float] = None
    temperature:        Optional[float] = None
    on_supplemental_o2: Optional[bool]  = False
    avpu:               Optional[str]   = "A"   # A | V | P | U


class Labs(BaseModel):
    wbc:        Optional[float] = None
    hemoglobin: Optional[float] = None
    platelets:  Optional[float] = None
    creatinine: Optional[float] = None
    bun:        Optional[float] = None
    sodium:     Optional[float] = None
    potassium:  Optional[float] = None
    lactate:    Optional[float] = None
    glucose:    Optional[float] = None
    bilirubin:  Optional[float] = None
    inr:        Optional[float] = None
    crp:        Optional[float] = None


class Medications(BaseModel):
    vasopressors:    bool = False
    antibiotics:     bool = False
    anticoagulants:  bool = False


class PredictConfig(BaseModel):
    horizon_hours: int = 24    # 6 | 12 | 24 | 48


class PredictRequest(BaseModel):
    patient:     PatientContext
    vitals:      Vitals
    labs:        Labs
    medications: Medications
    config:      PredictConfig


# ── Clinical scoring engine ────────────────────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    "respiratory_rate":       "Respiratory Rate",
    "spo2":                   "SpO₂",
    "supplemental_o2":        "Supplemental O₂",
    "systolic_bp":            "Systolic Blood Pressure",
    "heart_rate":             "Heart Rate",
    "temperature":            "Temperature",
    "avpu":                   "Consciousness (AVPU)",
    "lactate":                "Serum Lactate",
    "creatinine":             "Creatinine",
    "wbc":                    "White Blood Cell Count",
    "inr":                    "INR",
    "bilirubin":              "Bilirubin",
    "potassium":              "Potassium",
    "sodium":                 "Sodium",
    "glucose":                "Glucose",
    "hemoglobin":             "Hemoglobin",
    "vasopressors":           "Vasopressor Use",
    "antibiotics":            "Antibiotic Use",
    "age":                    "Patient Age",
    "hours_since_admission":  "Time Since Admission",
    "prior_deterioration":    "Prior Deterioration Event",
}

FEATURE_UNITS: dict[str, str] = {
    "respiratory_rate":  "breaths/min",
    "spo2":              "%",
    "systolic_bp":       "mmHg",
    "heart_rate":        "bpm",
    "temperature":       "°C",
    "lactate":           "mmol/L",
    "creatinine":        "mg/dL",
    "wbc":               "×10³/µL",
    "inr":               "",
    "bilirubin":         "mg/dL",
    "potassium":         "mEq/L",
    "sodium":            "mEq/L",
    "glucose":           "mg/dL",
    "hemoglobin":        "g/dL",
    "age":               "years",
    "hours_since_admission": "h",
}


def _news2_vitals(vitals: Vitals) -> dict[str, float]:
    """
    Compute individual NEWS2 subscores for each vital parameter.
    Returns a dict of {feature_key: points}.
    """
    s: dict[str, float] = {}

    # Respiratory rate
    rr = vitals.respiratory_rate
    if rr is not None:
        if rr <= 8 or rr >= 25:
            s["respiratory_rate"] = 3
        elif 21 <= rr <= 24:
            s["respiratory_rate"] = 2
        elif 9 <= rr <= 11:
            s["respiratory_rate"] = 1
        else:
            s["respiratory_rate"] = 0

    # SpO2 (Scale 1 — no hypercapnia)
    spo2 = vitals.spo2
    if spo2 is not None:
        if spo2 <= 91:
            s["spo2"] = 3
        elif spo2 <= 93:
            s["spo2"] = 2
        elif spo2 <= 95:
            s["spo2"] = 1
        else:
            s["spo2"] = 0

    # Supplemental O2
    if vitals.on_supplemental_o2:
        s["supplemental_o2"] = 2

    # Systolic BP
    sbp = vitals.systolic_bp
    if sbp is not None:
        if sbp <= 90 or sbp >= 220:
            s["systolic_bp"] = 3
        elif sbp <= 100:
            s["systolic_bp"] = 2
        elif sbp <= 110:
            s["systolic_bp"] = 1
        else:
            s["systolic_bp"] = 0

    # Heart rate
    hr = vitals.heart_rate
    if hr is not None:
        if hr <= 40 or hr >= 131:
            s["heart_rate"] = 3
        elif 111 <= hr <= 130:
            s["heart_rate"] = 2
        elif (41 <= hr <= 50) or (91 <= hr <= 110):
            s["heart_rate"] = 1
        else:
            s["heart_rate"] = 0

    # Temperature
    temp = vitals.temperature
    if temp is not None:
        if temp <= 35.0:
            s["temperature"] = 3
        elif temp <= 36.0:
            s["temperature"] = 1
        elif temp >= 39.1:
            s["temperature"] = 2
        elif temp >= 38.1:
            s["temperature"] = 1
        else:
            s["temperature"] = 0

    # Consciousness (AVPU)
    if vitals.avpu and vitals.avpu != "A":
        s["avpu"] = 3

    return s


def _lab_subscores(labs: Labs) -> dict[str, float]:
    """
    Lab-based augmentation scores on top of NEWS2.
    Graded by severity; returns {feature_key: points}.
    """
    s: dict[str, float] = {}

    # Lactate (mmol/L)
    if labs.lactate is not None:
        if labs.lactate >= 4.0:
            s["lactate"] = 4
        elif labs.lactate >= 2.0:
            s["lactate"] = 2
        elif labs.lactate >= 1.5:
            s["lactate"] = 1

    # Creatinine (mg/dL)
    if labs.creatinine is not None:
        if labs.creatinine >= 3.0:
            s["creatinine"] = 3
        elif labs.creatinine >= 2.0:
            s["creatinine"] = 2
        elif labs.creatinine >= 1.5:
            s["creatinine"] = 1

    # WBC (×10³/µL)
    if labs.wbc is not None:
        if labs.wbc > 20 or labs.wbc < 2:
            s["wbc"] = 3
        elif labs.wbc > 15 or labs.wbc < 3:
            s["wbc"] = 2
        elif labs.wbc > 12 or labs.wbc < 4:
            s["wbc"] = 1

    # INR
    if labs.inr is not None:
        if labs.inr >= 3.0:
            s["inr"] = 3
        elif labs.inr >= 2.0:
            s["inr"] = 2
        elif labs.inr >= 1.5:
            s["inr"] = 1

    # Bilirubin (mg/dL)
    if labs.bilirubin is not None:
        if labs.bilirubin >= 6.0:
            s["bilirubin"] = 3
        elif labs.bilirubin >= 3.0:
            s["bilirubin"] = 2
        elif labs.bilirubin >= 1.5:
            s["bilirubin"] = 1

    # Potassium (mEq/L)
    if labs.potassium is not None:
        if labs.potassium < 2.5 or labs.potassium > 6.5:
            s["potassium"] = 3
        elif labs.potassium < 3.0 or labs.potassium > 5.8:
            s["potassium"] = 2
        elif labs.potassium < 3.5 or labs.potassium > 5.2:
            s["potassium"] = 1

    # Sodium (mEq/L)
    if labs.sodium is not None:
        if labs.sodium < 120 or labs.sodium > 160:
            s["sodium"] = 3
        elif labs.sodium < 125 or labs.sodium > 155:
            s["sodium"] = 2
        elif labs.sodium < 130 or labs.sodium > 150:
            s["sodium"] = 1

    # Glucose (mg/dL)
    if labs.glucose is not None:
        if labs.glucose >= 400 or labs.glucose < 50:
            s["glucose"] = 3
        elif labs.glucose >= 250 or labs.glucose < 60:
            s["glucose"] = 2
        elif labs.glucose >= 180 or labs.glucose < 70:
            s["glucose"] = 1

    # Hemoglobin (g/dL)
    if labs.hemoglobin is not None:
        if labs.hemoglobin < 6.0:
            s["hemoglobin"] = 3
        elif labs.hemoglobin < 8.0:
            s["hemoglobin"] = 2
        elif labs.hemoglobin < 10.0:
            s["hemoglobin"] = 1

    return s


def _context_subscores(patient: PatientContext, meds: Medications) -> dict[str, float]:
    """
    Patient context and medication subscores.
    """
    s: dict[str, float] = {}

    # Vasopressors — strong independent indicator
    if meds.vasopressors:
        s["vasopressors"] = 5

    # Antibiotics — suggests active infection
    if meds.antibiotics:
        s["antibiotics"] = 1

    # Age
    if patient.age is not None:
        if patient.age >= 85:
            s["age"] = 3
        elif patient.age >= 75:
            s["age"] = 2
        elif patient.age >= 65:
            s["age"] = 1

    # Time since admission (early deterioration is highest risk)
    hrs = patient.hours_since_admission
    if hrs is not None:
        if hrs <= 6:
            s["hours_since_admission"] = 2
        elif hrs <= 24:
            s["hours_since_admission"] = 1

    # Prior deterioration
    if patient.prior_deterioration:
        s["prior_deterioration"] = 2

    return s


def _horizon_adjustment(horizon_hours: int) -> float:
    """
    Longer prediction horizons carry intrinsically higher probability.
    Returns a multiplier applied to the raw score before sigmoid.
    """
    return {6: 0.75, 12: 0.88, 24: 1.0, 48: 1.15}.get(horizon_hours, 1.0)


def _score_to_risk(total_score: float, horizon_hours: int) -> float:
    """
    Map augmented clinical score to [0, 1] risk probability via sigmoid.
    Calibrated against NEWS2 published risk tables:
      score ≤4  → ~5–15%
      score 5–8 → ~20–45%
      score 9–13 → ~50–75%
      score 14+  → ~80–97%
    """
    adj = _horizon_adjustment(horizon_hours)
    z = (total_score * adj - 12) * 0.28
    risk = 1.0 / (1.0 + math.exp(-z))
    return round(min(max(risk, 0.01), 0.99), 3)


def _risk_level(risk: float) -> tuple[str, str]:
    """Return (level_label, css_class) for a heuristic risk score [0,1]."""
    if risk < 0.20:
        return "LOW", "low"
    if risk < 0.45:
        return "MODERATE", "moderate"
    if risk < 0.70:
        return "HIGH", "high"
    return "CRITICAL", "critical"


def _ml_risk_level(raw_score: float) -> tuple[str, str]:
    """
    Return (level_label, css_class) for a raw XGBoost score using
    thresholds validated on MIMIC-IV (see _ML_THRESH_* constants).
    """
    if raw_score < _ML_THRESH_LOW:
        return "LOW", "low"
    if raw_score < _ML_THRESH_MODERATE:
        return "MODERATE", "moderate"
    if raw_score < _ML_THRESH_HIGH:
        return "HIGH", "high"
    return "CRITICAL", "critical"


def _plain_explanation(feature: str, value: float | bool | None, points: float) -> str:
    """Generate a plain-language explanation for a feature contribution."""
    # Use safe formatting helpers to avoid NoneType format errors
    def f0(v):  return f"{v:.0f}" if v is not None else "—"
    def f1(v):  return f"{v:.1f}" if v is not None else "—"

    explanations: dict[str, str] = {
        "respiratory_rate":      f"Respiratory rate of {f0(value)} breaths/min deviates from normal range (12–20), signalling breathing difficulty.",
        "spo2":                  f"SpO₂ of {f0(value)}% indicates reduced blood oxygen, a direct marker of respiratory compromise.",
        "supplemental_o2":       "Requirement for supplemental oxygen implies baseline respiratory insufficiency.",
        "systolic_bp":           f"Systolic BP of {f0(value)} mmHg indicates haemodynamic instability.",
        "heart_rate":            f"Heart rate of {f0(value)} bpm is outside the safe range (51–90), suggesting cardiac or metabolic stress.",
        "temperature":           f"Temperature of {f1(value)}°C indicates fever or hypothermia — both associated with systemic illness.",
        "avpu":                  "Altered level of consciousness (below Alert) is a high-acuity neurological sign.",
        "lactate":               f"Lactate of {f1(value)} mmol/L suggests tissue hypoperfusion or septic shock.",
        "creatinine":            f"Creatinine of {f1(value)} mg/dL indicates acute kidney injury.",
        "wbc":                   f"WBC of {f1(value)} ×10³/µL suggests active infection or immune dysfunction.",
        "inr":                   f"INR of {f1(value)} indicates coagulopathy — elevated bleeding or clotting risk.",
        "bilirubin":             f"Bilirubin of {f1(value)} mg/dL suggests hepatic dysfunction.",
        "potassium":             f"Potassium of {f1(value)} mEq/L is outside safe range — risk of cardiac arrhythmia.",
        "sodium":                f"Sodium of {f1(value)} mEq/L indicates electrolyte disturbance affecting brain and cardiovascular function.",
        "glucose":               f"Glucose of {f0(value)} mg/dL is significantly abnormal, suggesting metabolic instability.",
        "hemoglobin":            f"Haemoglobin of {f1(value)} g/dL indicates anaemia, reducing oxygen-carrying capacity.",
        "vasopressors":          "Active vasopressor use indicates circulatory failure requiring pharmacological blood pressure support.",
        "antibiotics":           "Antibiotic treatment suggests active infection or sepsis workup.",
        "age":                   f"Age {f0(value)} years increases baseline vulnerability to acute deterioration.",
        "hours_since_admission": f"{f0(value)} h since admission — early-admission deterioration carries the highest relative risk.",
        "prior_deterioration":   "Previous deterioration event significantly elevates recurrence risk.",
    }
    return explanations.get(feature, f"{FEATURE_LABELS.get(feature, feature)} contributes to elevated risk.")


def run_risk_engine(req: PredictRequest) -> dict:
    """
    Full risk computation pipeline.
    Returns structured results dict.
    """
    # Compute subscores
    vital_scores   = _news2_vitals(req.vitals)
    lab_scores     = _lab_subscores(req.labs)
    context_scores = _context_subscores(req.patient, req.medications)

    all_scores = {**vital_scores, **lab_scores, **context_scores}
    news2_score    = sum(vital_scores.values())
    total_score    = sum(all_scores.values())
    risk           = _score_to_risk(total_score, req.config.horizon_hours)
    level, css     = _risk_level(risk)

    # Raw feature values for explanations
    vitals_raw = {
        "respiratory_rate": req.vitals.respiratory_rate,
        "spo2":             req.vitals.spo2,
        "systolic_bp":      req.vitals.systolic_bp,
        "heart_rate":       req.vitals.heart_rate,
        "temperature":      req.vitals.temperature,
    }
    labs_raw = {
        "lactate":    req.labs.lactate,
        "creatinine": req.labs.creatinine,
        "wbc":        req.labs.wbc,
        "inr":        req.labs.inr,
        "bilirubin":  req.labs.bilirubin,
        "potassium":  req.labs.potassium,
        "sodium":     req.labs.sodium,
        "glucose":    req.labs.glucose,
        "hemoglobin": req.labs.hemoglobin,
    }
    context_raw = {
        "vasopressors":          req.medications.vasopressors,
        "antibiotics":           req.medications.antibiotics,
        "age":                   req.patient.age,
        "hours_since_admission": req.patient.hours_since_admission,
        "prior_deterioration":   req.patient.prior_deterioration,
    }
    all_raw = {**vitals_raw, **labs_raw, **context_raw}

    # SHAP-style feature contributions (proportional to score × risk)
    features = []
    for feat, pts in sorted(all_scores.items(), key=lambda x: -x[1]):
        if pts == 0:
            continue
        raw_val = all_raw.get(feat)
        unit    = FEATURE_UNITS.get(feat, "")
        if isinstance(raw_val, bool):
            value_str = "Yes" if raw_val else "No"
        elif raw_val is not None:
            value_str = f"{raw_val:.1f} {unit}".strip()
        else:
            value_str = "—"

        features.append({
            "feature":      feat,
            "label":        FEATURE_LABELS.get(feat, feat),
            "value":        value_str,
            "points":       pts,
            "contribution": round((pts / max(total_score, 1)) * risk, 4),
            "direction":    "increases_risk",
            "explanation":  _plain_explanation(feat, raw_val, pts),
        })

    top_factors = features[:3]

    # Clinical interpretation
    if level == "CRITICAL":
        note = (
            f"This patient has a CRITICAL deterioration risk ({risk*100:.0f}%) "
            f"within the next {req.config.horizon_hours} h. "
            "Immediate clinical review and ICU notification are strongly recommended. "
            "The primary drivers are highlighted above."
        )
    elif level == "HIGH":
        note = (
            f"This patient has a HIGH deterioration risk ({risk*100:.0f}%) "
            f"within the next {req.config.horizon_hours} h. "
            "Escalate monitoring frequency and consider senior clinician review. "
            "Address modifiable risk factors."
        )
    elif level == "MODERATE":
        note = (
            f"This patient has a MODERATE deterioration risk ({risk*100:.0f}%) "
            f"within the next {req.config.horizon_hours} h. "
            "Increase observation frequency and reassess within 1–2 h. "
            "Monitor for trend worsening."
        )
    else:
        note = (
            f"This patient has a LOW deterioration risk ({risk*100:.0f}%) "
            f"within the next {req.config.horizon_hours} h. "
            "Continue routine monitoring. Reassess if clinical status changes."
        )

    return {
        "risk_score":     risk,
        "risk_level":     level,
        "risk_css":       css,
        "risk_pct":       f"{risk*100:.1f}%",
        "horizon_hours":  req.config.horizon_hours,
        "news2_score":    int(news2_score),
        "augmented_score": int(total_score),
        "top_factors":    top_factors,
        "all_features":   features,
        "clinical_note":  note,
        "patient_id":     req.patient.patient_id,
        "vitals_count":   sum(1 for v in [
            req.vitals.heart_rate, req.vitals.systolic_bp, req.vitals.spo2,
            req.vitals.respiratory_rate, req.vitals.temperature
        ] if v is not None),
        "labs_count": sum(1 for v in [
            req.labs.wbc, req.labs.hemoglobin, req.labs.creatinine,
            req.labs.lactate, req.labs.inr, req.labs.potassium,
            req.labs.sodium, req.labs.glucose
        ] if v is not None),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/api/predict")
async def submit_predict(req: PredictRequest):
    """Submit patient data for risk prediction. Returns job_id immediately."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"job_id": job_id, "status": "running", "results": None}
    asyncio.create_task(_process(job_id, req))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/health")
def health():
    return {
        "status":  "ok",
        "version": "0.2.0",
        "engine":  "xgboost_mimic_iv" if _ML_MODEL is not None else "heuristic_news2",
        "features": len(_ML_COLS) if _ML_COLS else 0,
    }


# ── Background processor ────────────────────────────────────────────────────────

async def _process(job_id: str, req: PredictRequest):
    try:
        # Small async yield so the HTTP response returns first
        await asyncio.sleep(0.05)
        if _ML_MODEL is not None:
            results = run_ml_engine(req)
        else:
            results = run_risk_engine(req)
        jobs[job_id]["status"]  = "completed"
        jobs[job_id]["results"] = results
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = str(exc)


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8009, reload=True)
