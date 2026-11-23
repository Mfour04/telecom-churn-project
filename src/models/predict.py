#Prediction logic for a single customer (used by the API)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocessing import preprocess_single

DECISION_THRESHOLD = 0.5


def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.75:
        return "High"
    return "Critical"


def get_confidence(probability: float) -> str:
    distance = abs(probability - 0.5) * 2
    if distance >= 0.6:
        return "High"
    if distance >= 0.3:
        return "Medium"
    return "Low"


def predict_one(data: dict, model, preprocessor, config: dict) -> dict:
    """Run preprocessing + model inference for one customer record."""
    X = preprocess_single(data, preprocessor, config)
    probability = float(model.predict_proba(X)[0, 1])
    is_churn = probability >= DECISION_THRESHOLD

    return {
        "churn_prediction": is_churn,
        "churn_label": "Churn" if is_churn else "No churn",
        "churn_probability": round(probability, 4),
        "confidence": get_confidence(probability),
        "risk_level": get_risk_level(probability),
    }
