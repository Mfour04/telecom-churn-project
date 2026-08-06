#Fast API application
#Rest endpoints for real-time inference
import yaml
import joblib
import numpy as np
from pathlib import Path 
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
from src.utils.logger import get_logger

logger = get_logger(__name__)

model = None
preprocessor = None
config = None
model_metric = {}

def load_config() -> dict: 
    with open("configs/config.yaml") as f:
        return yaml.safe_load()

 #Load model artifact from startup   
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, config, model_metric

    config = load_config()
    logger.info("Starting up FastAPI")

    model_path = Path(["api"]["model_path"])
    pre_path = Path(["api"]["preprocessor_path"])

    if model_path.exists():
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    else: 
        logger.info(f"Model not found at {model_path}. Run training first")

    if pre_path.exists():
        model = joblib.load(pre_path)
        logger.info(f"Preprocessor loaded from {pre_path}")
    else: 
        logger.info(f"Preprocessor not found at {pre_path}")

    yield
    logger.info("API shut down.")

app = FastAPI(
    title="Telso Customer Churn Prediction API",
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