# Telco Customer Churn Prediction

End-to-end customer churn prediction system.

## Overview

The pipeline predicts whether a telecom customer will churn, using the
[Telco Customer Churn](data/raw/Telco-Customer-Churn.csv) dataset.

1. **Preprocessing** (`src/data/preprocessing.py`): cleaning, train/test split, feature engineering
   (`tenure_group`, `high_charges`, `num_services`, `charge_per_service`) and a `ColumnTransformer`
   fitted on the training split only. Feature thresholds are saved so the API can apply exactly the same
   transformations at inference time.
2. **Training** (`src/models/train.py`): XGBoost, Logistic Regression and Random Forest, each tuned with
   `RandomizedSearchCV` (5-fold stratified CV, ROC-AUC). All runs are tracked in MLflow, and the model with
   the best test ROC-AUC is saved to `models/best_model.pkl`.
3. **Serving** (`src/api/main.py`): FastAPI service for single and batch predictions.

Configuration lives in `configs/config.yaml`.

### Tech stack

Python 3.14, uv, scikit-learn, XGBoost, MLflow, FastAPI, Docker.

### Project structure

```
configs/    config.yaml (paths, features, hyperparameters, tuning ranges, API settings)
data/       raw and processed datasets
models/     trained artifacts (best_model.pkl, preprocessor.pkl, feature_thresholds.yaml)
notebooks/  experimentation
reports/    evaluation figures per model
src/        data, models, api, utils
tests/      pytest suite
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## How to run

Run every command from the project root, because the config and artifact paths are relative to it.

```bash
uv run python src/data/preprocessing.py           # preprocessing only
uv run python src/models/train.py                 # full training pipeline (writes models/)
uv run mlflow ui --port 5000                      # MLflow UI at http://localhost:5000
uv run uvicorn src.api.main:app --reload --port 8000   # API, docs at http://localhost:8000/docs
uv run pytest -v                                  # tests
docker compose up --build                         # API (8000) + MLflow server (5000)
```

Train a model before starting the API. Without trained artifacts, the prediction endpoints return 503.

## Model results

These numbers are from the last training run on 2026-11-21. CV AUC is the best 5-fold CV ROC-AUC from
tuning. The other metrics are on the test set.

| Model | CV AUC | Test ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **random_forest** (selected) | 0.8471 | 0.8423 | 0.6311 | 0.5321 | 0.7754 |
| xgboost | 0.8468 | 0.8415 | 0.6258 | 0.5363 | 0.7513 |
| logistic_regression | 0.8462 | 0.8416 | 0.6130 | 0.4992 | 0.7941 |

Evaluation plots (confusion matrix, ROC curve, feature importance) are in `reports/figures/<model>/`.

## API usage

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Model and preprocessor load status |
| POST | `/predict` | Predict churn for one customer |
| POST | `/predict/batch` | Predict churn for up to 500 customers (`{"customers": [...]}`) |
| GET | `/metrics` | Loaded model type and training date |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20
  }'
```

Example response (from the current `random_forest` model):

```json
{"churn_prediction": true, "churn_label": "Churn", "churn_probability": 0.7649, "confidence": "Medium", "risk_level": "Critical"}
```

`risk_level` comes from the churn probability: Low < 0.30 ≤ Medium < 0.50 ≤ High < 0.75 ≤ Critical.
`confidence` measures how far the probability is from 0.5.
