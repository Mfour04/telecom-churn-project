import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Integration tests for the FastAPI endpoints.
Uses TestClient to test without running a live server.
"""

SAMPLE_CUSTOMER = {
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


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_version(self, client):
        data = client.get("/").json()
        assert "version" in data


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data


class TestPredictEndpoint:
    def test_predict_valid_customer(self, client):
        """Test prediction with valid data — may return 503 if no model trained yet."""
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        # Either succeeds or returns 503 (model not found)
        assert response.status_code in [200, 503]

    def test_predict_invalid_gender(self, client):
        bad = {**SAMPLE_CUSTOMER, "gender": "Unknown"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422   # Validation error

    def test_predict_negative_tenure(self, client):
        bad = {**SAMPLE_CUSTOMER, "tenure": -1}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_missing_field(self, client):
        bad = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "gender"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_single(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code in [200, 503]

    def test_batch_predict_multiple(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code in [200, 503]

    def test_batch_predict_empty_raises_422(self, client):
        payload = {"customers": []}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422
