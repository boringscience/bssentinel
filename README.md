# bssentinel — Patient Deterioration Risk Engine

A clinical decision support system that predicts acute patient deterioration in hospitalized patients. Combines the **NEWS2** scoring system with an **XGBoost model** trained on MIMIC-IV, achieving AUROC 0.758 vs 0.568 for NEWS2 alone.

bssentinel is a clincial tool from Boring Science Suite managed by Boring Science LLC. 

**Predicts** risk of ICU transfer, intubation, cardiac arrest, or death within 6, 12, 24, or 48 hours.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+ with pnpm

### Install & Run

```bash
# Install JS dependencies
pnpm install

# Install Python dependencies
pip install -r server/requirements.txt

# Run frontend + backend together
pnpm run dev:full
```

- Frontend: http://localhost:5183
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## API

### `POST /api/predict`

Submit patient data; returns a `job_id` immediately.

```json
{
  "patient":     { "patient_id": "P001", "age": 65, "admission_type": "general_ward", "hours_since_admission": 24 },
  "vitals":      { "heart_rate": 95, "systolic_bp": 120, "diastolic_bp": 75, "spo2": 96, "respiratory_rate": 18, "temperature": 37.5, "on_supplemental_o2": false, "avpu": "A" },
  "labs":        { "wbc": 7.2, "hemoglobin": 13.5, "creatinine": 1.1, "lactate": 1.2, "potassium": 4.0, "sodium": 140 },
  "medications": { "vasopressors": false, "antibiotics": false, "anticoagulants": false },
  "config":      { "horizon_hours": 24 }
}
```

### `GET /api/jobs/{job_id}`

Poll for results.

```json
{
  "job_id": "a3f8b2c1",
  "status": "completed",
  "results": {
    "risk_level": "CRITICAL",
    "risk_score": 0.65,
    "news2_score": 4,
    "alert_threshold": 0.30,
    "top_factors": ["respiratory_rate", "lactate", "vasopressors"],
    "clinical_note": "..."
  }
}
```

### `GET /api/health`

Health check and model load status.

---

## Model Performance

Trained and validated on MIMIC-IV (1.3M observations, ~50k admissions):

| Model | AUROC | AUPRC |
|---|---|---|
| NEWS2 alone | 0.568 | — |
| Logistic Regression | 0.708 | 0.068 |
| **XGBoost (bssentinel)** | **0.758** | **0.102** |

At the 90% sensitivity operating point: threshold ≥ 0.30 → 90.6% sensitivity, 2.8% PPV.

---

## Retraining the Model

The trained models in `ml/artifacts/` can be used directly. To retrain from MIMIC-IV:

```bash
cd ml
# See ml/README.md for MIMIC-IV access instructions
python train.py --mimic /path/to/mimic_iv --output ./artifacts --horizon 24
```

See [ml/README.md](ml/README.md) for full pipeline documentation.

---

## Architecture

```
bssentinel/
├── src/                        # React frontend
├── server/
│   ├── main.py                 # FastAPI backend + inference engine
│   └── requirements.txt
├── ml/
│   ├── train.py                # End-to-end training orchestrator
│   ├── data_loader.py          # MIMIC-IV cohort extraction
│   ├── feature_engineering.py  # Feature matrix + NEWS2 scoring
│   ├── model.py                # XGBoost/LR training + calibration
│   ├── evaluate.py             # Validation metrics
│   ├── explain.py              # SHAP explainability
│   ├── artifacts/              # Trained models + metrics
│   └── README.md               # ML pipeline docs
├── paper/                      # Manuscript (LaTeX + figures)
├── package.json
└── vite.config.js
```

**Design notes:**
- Async API: `/api/predict` is non-blocking; poll `/api/jobs/{id}` for results
- Stateless: in-memory job store (not suitable for multi-process production deployments)
- Graceful fallback: if trained model not found, falls back to pure NEWS2 heuristic
- CORS is wide-open for development — restrict `allow_origins` in production

---

## Data & Privacy

- **MIMIC-IV** is a public but credentialed dataset (PhysioNet registration required)
- Raw MIMIC-IV data is **not** included in this repo
- Trained model artifacts are derived from MIMIC-IV and are safe to distribute
- The live API accepts patient vitals but **does not persist data to disk**

---

## Authors

Sasi Jagadeesan, Suman Mishra, Pritam Kumar Panda — BORING SCIENCE LLC

See `paper/` for the associated manuscript.
