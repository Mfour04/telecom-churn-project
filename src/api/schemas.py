#Pydantic schemas for FastAPI request/response validation
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from enum import Enum

class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"

class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"

class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    no = "No"

class MultipleLinesEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone = "No phone service"

class AddOnServiceEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"

class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"

class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"

class CustomerFeatures(BaseModel):
    #Input feature for churn prediction

    gender: GenderEnum = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Is senior city (0 or 1)")
    Partner: YesNoEnum = Field(..., description="Has partner")
    Dependents: YesNoEnum = Field(..., description="Has dependent")
    tenure: int = Field(..., ge=0, le=100 ,description="Month as customer")
    PhoneService: YesNoEnum = Field(..., description="Has phone service")
    MultipleLines: MultipleLinesEnum = Field(..., description="Has multiple lines")
    InternetService: InternetServiceEnum = Field(..., description="Internet service type")
    OnlineSecurity: AddOnServiceEnum = Field(..., description="Has online security")
    OnlineBackup: AddOnServiceEnum = Field(..., description="Has online backup")
    DeviceProtection: AddOnServiceEnum = Field(..., description="Has device protection")
    TechSupport: AddOnServiceEnum = Field(..., description="Has tech support")
    StreamingTV: AddOnServiceEnum = Field(..., description="Has streaming TV")
    StreamingMovies: AddOnServiceEnum = Field(..., description="Has streaming movies")
    Contract: ContractEnum = Field(..., description="Contract type")
    PaperlessBilling: YesNoEnum = Field(..., description="Uses paperless billing")
    PaymentMethod: PaymentMethodEnum = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charges in USD")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
            }
        }
    )

class PredictionResponse(BaseModel): 
    #Prediction result for a single customer
    churn_prediction: bool = Field(..., description="Will the customer churn?")
    churn_label: str = Field(..., description="'Churn' or 'No churn'")
    churn_probability: float = Field(..., description="Probability of churning (0-1)")
    confidence: str = Field(..., description="Confidence level: Low/Medium/High")
    risk_level: str = Field(..., description="Risk level: Low/Medium/High/Critical")

class BatchPredictionRequest(BaseModel):
    #Batch prediction request.
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=500)

class BatchPredictionResponse(BaseModel):
    #Batch prediction response.
    total: int
    churn_count: int 
    no_churn_count: int
    churn_rate: float
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    #API health status.
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    #Stored model evaluation metrics.
    model_type: str
    metrics: dict
    training_date: Optional[str] = None
