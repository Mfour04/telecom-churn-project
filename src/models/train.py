import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from src.data.preprocessing import prepare_data, load_config
from src.models.evaluate import generate_evaluation_report, compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_models(config: dict) -> dict:
    """Instantiate all candidate models from config."""
    xgb_cfg = config["model"]["xgboost"]
    lr_cfg  = config["model"]["logistic_regression"]
    rf_cfg  = config["model"]["random_forest"]

    return {
        "xgboost": XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            eval_metric=xgb_cfg["eval_metric"],
            random_state=xgb_cfg["random_state"],
            scale_pos_weight=xgb_cfg["scale_pos_weight"],
            verbosity=0
        ),
        "logistic_regression": LogisticRegression(
            C=lr_cfg["C"],
            max_iter=lr_cfg["max_iter"],
            random_state=lr_cfg["random_state"],
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=rf_cfg["random_state"],
            class_weight="balanced",
            n_jobs=-1
        )
    }


def train_and_track(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    config: dict
) -> tuple:
    """
    Train one model and log everything to MLflow.

    Returns:
        (metrics dict, mlflow run_id)
    """
    experiment_name = config["model"]["experiment_name"]
    mlflow.set_tracking_uri(config["paths"]["mlflow_tracking_uri"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}") as run:
        run_id = run.info.run_id
        logger.info(f"\n{'='*55}")
        logger.info(f"Training: {model_name} | MLflow run: {run_id[:8]}")
        logger.info("=" * 55)

        # Log model hyperparameters
        mlflow.log_params(model.get_params())

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute + log metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Log dataset info
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        # Generate & log evaluation plots
        generate_evaluation_report(
            y_test, y_pred, y_prob,
            model=model if hasattr(model, "feature_importances_") else None,
            feature_names=feature_names,
            output_dir=f"reports/figures/{model_name}"
        )
        mlflow.log_artifacts(f"reports/figures/{model_name}", artifact_path="evaluation")

        # Log model
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"  ROC-AUC : {metrics['roc_auc']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")

    return metrics, run_id


def select_best_model(results: dict) -> str:
    """Select model with highest ROC-AUC score."""
    best = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    logger.info(f"Best model: {best} (ROC-AUC = {results[best]['metrics']['roc_auc']:.4f})")
    return best


def save_best_model(model, preprocessor, config: dict) -> None:
    """Persist best model and preprocessor to disk."""
    model_dir = Path(config["api"]["model_path"]).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config["api"]["model_path"])
    joblib.dump(preprocessor, config["api"]["preprocessor_path"])

    logger.info(f"Model saved      -> {config['api']['model_path']}")
    logger.info(f"Preprocessor saved -> {config['api']['preprocessor_path']}")


def run_training_pipeline() -> None:
    """Main entry point: full training pipeline."""
    logger.info("Starting training pipeline...")

    config = load_config()
    logger.info("Loading configuration...")

    # Prepare data
    result = prepare_data(config)

    if len(result) != 6:
        raise ValueError(
            f"prepare_data() returned {len(result)} values, expected 6: "
            "(X_train, X_test, y_train, y_test, preprocessor, feature_names). "
            "Please update src/data/preprocessing.py to the latest version."
        )

    X_train, X_test, y_train, y_test, preprocessor, feature_names = result

    # Sanity check: dung shapes
    assert X_train.ndim == 2, f"X_train should be 2D, got {X_train.shape}"
    assert y_train.ndim == 1, f"y_train should be 1D, got {y_train.shape}"
    assert X_test.ndim  == 2, f"X_test should be 2D, got {X_test.shape}"
    assert y_test.ndim  == 1, f"y_test should be 1D, got {y_test.shape}"

    logger.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    logger.info(f"Features: {len(feature_names)}")

    models = get_models(config)
    results = {}

    for name, model in models.items():
        metrics, run_id = train_and_track(
            name, model,
            X_train, X_test,
            y_train, y_test,
            feature_names, config
        )
        results[name] = {"model": model, "metrics": metrics, "run_id": run_id}

    # Select and save best
    best_name = select_best_model(results)
    best_model = results[best_name]["model"]
    save_best_model(best_model, preprocessor, config)

    # Summary table
    logger.info("Model Comparison:")
    logger.info(f"  {'Model':<25} {'ROC-AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    logger.info("  " + "-" * 65)
    for name, r in results.items():
        m = r["metrics"]
        tag = " <- BEST" if name == best_name else ""
        logger.info(
            f"  {name:<25} {m['roc_auc']:>8.4f} {m['f1_score']:>8.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>8.4f}{tag}"
        )

    logger.info("Training pipeline complete!")
    logger.info("Run: mlflow ui --port 5000  ->  http://localhost:5000")


if __name__ == "__main__":
    run_training_pipeline()