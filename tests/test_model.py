"""
Unit tests for model evaluation utilities.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.evaluate import compute_metrics


@pytest.fixture
def binary_predictions():
    """Balanced binary prediction fixture."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.3, 0.7, 0.15, 0.85])
    return y_true, y_pred, y_prob


class TestComputeMetrics:
    def test_returns_dict(self, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert isinstance(metrics, dict)

    def test_has_required_keys(self, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_values_in_range(self, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of range [0,1]"

    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.05, 0.05, 0.95, 0.95])
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_rounding(self, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)
        for k, v in metrics.items():
            assert v == round(v, 4), f"Metric {k} not rounded to 4 dp"