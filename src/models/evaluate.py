import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute full set of classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class

    Returns:
        Dictionary of metric name → value
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "avg_precision": round(average_precision_score(y_true, y_proba), 4),
    }

    logger.info("=" * 50)
    logger.info("MODEL EVALUATION METRICS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k:20s}: {v:.4f}")
    logger.info("=" * 50)

    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None) -> plt.Figure:  
    """
    Plot and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted binary labels
        output_path: Path to save the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn "],
                ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to: {save_path}")

    return fig

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    Plot and save ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        output_path: Path to save the ROC curve plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True,alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to: {save_path}")

    return fig

def plot_feature_importance(model, feature_names: list,
                             top_n: int = 20, save_path: str = None) -> plt.Figure:
    """Plot top-N feature importances."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        logger.warning("Model does not have feature_importances_ attribute.")
        return None

    df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df["feature"][::-1], df["importance"][::-1], color="#2563EB")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def generate_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: np.ndarray, model=None,
                                 feature_names: list = None,
                                 output_dir: str = "reports/figures") -> dict:
    """
    Generate complete evaluation report with metrics and plots.

    Returns:
        Dictionary of metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    plot_confusion_matrix(y_true, y_pred, save_path=str(output_path / "confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, save_path=str(output_path / "roc_curve.png"))

    if model is not None and feature_names is not None:
        plot_feature_importance(
            model, feature_names,
            save_path=str(output_path / "feature_importance.png")
        )

    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['No Churn', 'Churn'])}")

    return metrics