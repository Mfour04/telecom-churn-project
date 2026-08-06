import pytest
import pandas as pd 
import numpy as np 
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.append(str(Path(__file__).parent.parent))
from src.data.preprocessing import clean_data, feature_engineering

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No"],
        "tenure": [12, 0, 60],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "No", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract": ["Month-to-month", "Two year", "One year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 89.10, 20.25],
        "TotalCharges": ["358.2", " ", "1221.0"],   # space = missing
        "Churn": ["No", "Yes", "No"]
    })

# Test data cleaning
class TestCleanData:
    def test_removes_customer_id(self, sample_df):
        df = clean_data(sample_df)
        assert "customerID" not in df.columns, "customerID should be removed"
    
    def test_encodes_churn_as_binary(self, sample_df):
        df = clean_data(sample_df)
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_total_charges_numeric(self, sample_df):
        df = clean_data(sample_df)
        assert pd.api.types.is_float_dtype(df["TotalCharges"])

    def test_imputes_missing_total_charges(self, sample_df):
        df = clean_data(sample_df)
        assert df["TotalCharges"].isna().sum() == 0

    def test_shape_consistent(self, sample_df):
        df = clean_data(sample_df)
        assert df.shape[0] == sample_df.shape[0]

class TestFeatureEngineering:
    def test_tenure_group_created(self, sample_df):
        df = clean_data(sample_df)
        df_fe = feature_engineering(df)
        assert "tenure_group" in df_fe.columns, "tenure_group should be created"

    def test_num_services_created(self, sample_df):
        df = clean_data(sample_df)
        df = feature_engineering(df)
        assert "num_services" in df.columns, "num_services should be created"
        assert (df["num_services"] >= 0).all()

    def test_charge_per_service_positive(self, sample_df):
        df = clean_data(sample_df)
        df = feature_engineering(df)
        assert (df["charge_per_service"] > 0).all()
    
    def test_no_nulls_after_engineering(self, sample_df):
        df = clean_data(sample_df)
        df = feature_engineering(df)
        for col in ["nume_service", "high_charges", "charge_per_service"]:
            assert df[col].isna().sum() == 0, f"Nulls in {col}"