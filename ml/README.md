# bssentinel — MIMIC-IV Validation Pipeline

Trains and validates the patient deterioration risk model on MIMIC-IV.
This is the process that turns bssentinel from a heuristic scorer into a
clinically validated, publication-ready risk engine.

## Prerequisites

### 1. MIMIC-IV Access
MIMIC-IV requires credentialing via PhysioNet:
1. Complete the CITI "Data or Specimens Only Research" course
2. Register at https://physionet.org and link your credential
3. Request access to `physionet.org/content/mimiciv/`
4. Download and extract — you need the `hosp/` and `icu/` modules

Expected directory structure:
```
mimic_iv/
├── hosp/
│   ├── admissions.csv
│   ├── patients.csv
│   ├── labevents.csv
│   ├── d_labitems.csv
│   └── prescriptions.csv
└── icu/
    ├── icustays.csv
    ├── chartevents.csv      ← ~3.5 GB
    ├── inputevents.csv
    └── procedureevents.csv
```

### 2. Python environment
```bash
cd softwares/bssentinel/ml
pip install -r requirements.txt
```

---

## Running the pipeline

### Development run (fast, 2000 admissions)
```bash
python train.py \
  --mimic /path/to/mimic_iv \
  --output ./artifacts \
  --horizon 24 \
  --dev
```
Completes in ~10–20 minutes. Use this to verify the pipeline works end-to-end.

### Full training run
```bash
python train.py \
  --mimic /path/to/mimic_iv \
  --output ./artifacts_24h \
  --horizon 24 \
  --cache ./cache
```
Completes in ~2–4 hours on a modern laptop (chartevents I/O dominates).
Use `--cache` to avoid re-reading the raw CSVs on reruns.

### Multi-horizon (recommended for production)
```bash
for HORIZON in 6 12 24 48; do
  python train.py \
    --mimic /path/to/mimic_iv \
    --output ./artifacts_${HORIZON}h \
    --horizon $HORIZON \
    --cache ./cache
done
```

---

## Output artefacts

```
artifacts/
├── model_xgb.pkl            ← calibrated XGBoost — load this in server/main.py
├── model_lr.pkl             ← calibrated logistic regression (baseline)
├── xgb_raw.pkl              ← uncalibrated XGBoost — used by SHAP explainer
├── feature_cols.json        ← feature names in correct order
├── feature_importance.csv   ← XGBoost gain importance
├── shap_global.csv          ← mean |SHAP| per feature
├── oof_predictions.parquet  ← out-of-fold predictions for diagnostics
├── metrics.json             ← full validation metrics
└── figures/
    ├── roc_curve.png        ← AUROC comparison (XGB vs LR vs NEWS2)
    ├── pr_curve.png         ← AUPRC comparison
    ├── calibration.png      ← reliability diagram
    ├── threshold_analysis.png
    ├── shap_importance.png  ← global SHAP bar chart
    └── shap_beeswarm.png    ← SHAP beeswarm summary
```

---

## Expected performance (published benchmarks)

Based on similar MIMIC-IV deterioration studies:

| Model | AUROC | AUPRC |
|---|---|---|
| NEWS2 alone | ~0.74 | ~0.18 |
| Logistic Regression | ~0.79 | ~0.23 |
| **XGBoost (this pipeline)** | **~0.83–0.87** | **~0.30–0.38** |

Target: AUROC ≥ 0.82 to be clinically meaningful vs NEWS2.
DeLong p < 0.001 vs NEWS2 expected with full MIMIC-IV cohort (~50k admissions).

---

## Deploying the trained model

Once training is complete, update the server to use the trained model:

```python
# In softwares/bssentinel/server/main.py, add at the top:
from pathlib import Path
from ..ml.explain import build_server_explain_fn

MODEL_DIR    = Path(__file__).parent.parent / "ml" / "artifacts_24h"
FEATURE_COLS = json.load(open(MODEL_DIR / "feature_cols.json"))

# Load at startup (returns None if no model found → falls back to heuristic)
_ml_predict = build_server_explain_fn(MODEL_DIR, FEATURE_COLS)
```

Then in `run_risk_engine()`, call `_ml_predict(feature_dict, horizon_hours)` instead
of the heuristic scoring. See `explain.py → build_server_explain_fn()` for details.

---

## Files

| File | Purpose |
|---|---|
| `data_loader.py` | MIMIC-IV cohort extraction, vital/lab/med loading, outcome labelling |
| `feature_engineering.py` | Feature matrix construction, NEWS2 scoring, missingness flags |
| `model.py` | XGBoost + LR training, cross-validation, calibration, persistence |
| `evaluate.py` | AUROC/AUPRC with CIs, DeLong test, calibration plot, threshold analysis |
| `explain.py` | Real SHAP values, global importance, server integration function |
| `train.py` | End-to-end orchestrator |
| `requirements.txt` | Python dependencies |
