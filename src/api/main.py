#Fast API application
#Rest endpoints for real-time inference
import yaml
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.schemas import (
    CustomerFeatures, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, MetricsResponse
)
from src.data.preprocessing import preprocess_single
from src.models.predict import predict_one
from src.utils.logger import get_logger

logger = get_logger(__name__)

model = None
preprocessor = None
config = None
model_metric = {}

def load_config() -> dict: 
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

 #Load model artifact from startup   
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, config, model_metric

    config = load_config()
    logger.info("Starting up FastAPI")

    model_path = Path(config["api"]["model_path"])
    pre_path = Path(config["api"]["preprocessor_path"])

    if model_path.exists():
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}. Run training first")

    if pre_path.exists():
        preprocessor = joblib.load(pre_path)
        logger.info(f"Preprocessor loaded from {pre_path}")
    else:
        logger.warning(f"Preprocessor not found at {pre_path}")

    yield
    logger.info("API shut down.")

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "End-to-end ML system to predict customer churn in telecom.\n\n"
        "Provides real-time churn prediction with probability scores."
    ),
    version="1.0.0",
    lifespan= lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

#Helper

def require_model():
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not load."
        )


#Routes

@app.get("/")
def root():
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        version=app.version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    require_model()
    try:
        result = predict_one(customer.model_dump(mode="json"), model, preprocessor, config)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        )
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    require_model()
    try:
        predictions = [
            PredictionResponse(
                **predict_one(c.model_dump(mode="json"), model, preprocessor, config)
            )
            for c in request.customers
        ]
    except Exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed.",
        )

    total = len(predictions)
    churn_count = sum(p.churn_prediction for p in predictions)
    return BatchPredictionResponse(
        total=total,
        churn_count=churn_count,
        no_churn_count=total - churn_count,
        churn_rate=round(churn_count / total, 4),
        predictions=predictions,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not load."
        )
    model_path = Path(config["api"]["model_path"])
    training_date = (
        datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        if model_path.exists() else None
    )
    return MetricsResponse(
        model_type=type(model).__name__,
        metrics={},
        training_date=training_date,
    )