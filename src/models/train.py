import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if __name__ == "__main__":
    os.chdir(_PROJECT_ROOT)  # config/artifact/log paths are relative to the project root

from scipy.stats import randint, uniform
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.data.preprocessing import prepare_data, load_config, resolve_path
from src.models.evaluate import generate_evaluation_report, compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_base_models(config: dict) -> dict:
    """Instantiate all candidate models from config."""
    xgb_cfg = config["model"]["xgboost"]
    lr_cfg  = config["model"]["logistic_regression"]
    rf_cfg  = config["model"]["random_forest"]

    return {
        "xgboost": XGBClassifier(
            eval_metric=xgb_cfg["eval_metric"],
            random_state=xgb_cfg["random_state"],
            scale_pos_weight=xgb_cfg["scale_pos_weight"],
            verbosity=0,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=lr_cfg["max_iter"],
            random_state=lr_cfg["random_state"],
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            random_state=rf_cfg["random_state"],
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
def get_param_distribution(config: dict) -> dict:
    """Define hyperparameter search space for tuning."""
    tuning_cfg = config["tuning"]

    return {
        "xgboost": {
            "n_estimators": randint(
                tuning_cfg["xgboost"]["n_estimators"][0],
                tuning_cfg["xgboost"]["n_estimators"][1]
            ),
            "max_depth": randint(
                tuning_cfg["xgboost"]["max_depth"][0],
                tuning_cfg["xgboost"]["max_depth"][1]
            ),
            "learning_rate": uniform(
                tuning_cfg["xgboost"]["learning_rate"][0],
                tuning_cfg["xgboost"]["learning_rate"][1]
            ),
            "subsample": uniform(
                tuning_cfg["xgboost"]["subsample"][0],
                tuning_cfg["xgboost"]["subsample"][1]
            ),
            "colsample_bytree": uniform(
                tuning_cfg["xgboost"]["colsample_bytree"][0],
                tuning_cfg["xgboost"]["colsample_bytree"][1]
            ),
        },
        "logistic_regression": {
            "C": uniform(
                tuning_cfg["logistic_regression"]["C"][0],
                tuning_cfg["logistic_regression"]["C"][1]
            ),
            "solver": tuning_cfg["logistic_regression"]["solver"],
            "penalty": tuning_cfg["logistic_regression"]["penalty"],
        },
        "random_forest": {
            "n_estimators": randint(
                tuning_cfg["random_forest"]["n_estimators"][0],
                tuning_cfg["random_forest"]["n_estimators"][1]
            ),
            "max_depth": randint(
                tuning_cfg["random_forest"]["max_depth"][0],
                tuning_cfg["random_forest"]["max_depth"][1]
            ),
            "min_samples_split": randint(
                tuning_cfg["random_forest"]["min_samples_split"][0],
                tuning_cfg["random_forest"]["min_samples_split"][1]
            ),
            "min_samples_leaf": randint(
                tuning_cfg["random_forest"]["min_samples_leaf"][0],
                tuning_cfg["random_forest"]["min_samples_leaf"][1]
            ),
            "max_features": tuning_cfg["random_forest"]["max_features"],
        },
    }

def tune_hyperparameter(
        model_name: str,
        model,
        param_dist: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: dict,
) -> tuple:
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    tuning_cfg = config["tuning"]
    n_iter = tuning_cfg["n_iter"]
    cv_folds = tuning_cfg["cv_folds"]
    scoring = tuning_cfg["scoring"]
    random_state = tuning_cfg["random_state"]

    logger.info(f"Starting hyperparameter tuning for {model_name}: {n_iter} iterations, {cv_folds}-fold CV, scoring={scoring}")
    logger.info(f"Scoring metric: {scoring}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=True,
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_cv_score = search.best_score_

    logger.info(f"  Best CV {scoring}: {best_cv_score:.4f}")
    logger.info(f"  Best params: {best_params}")

    cv_results = {
        f"cv_best_{scoring}": round(float(best_cv_score), 4),
        f"cv_mean_{scoring}": round(float(search.cv_results_["mean_test_score"].mean()), 4),
        f"cv_std_{scoring}": round(float(search.cv_results_["std_test_score"][search.best_index_]), 4),
    }

    return search.best_estimator_, best_params, cv_results

def train_and_track(
    model_name: str,
    model,
    best_params: dict,
    cv_results: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    config: dict
) -> tuple:
    
    mlflow.set_tracking_uri(resolve_path(config["mlflow"]["tracking_uri"]).as_uri())
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"{model_name}_tuned") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id[:8]}")

        mlflow.log_params({
            "model_name": model_name,
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "n_features": X_train.shape[1],
            "tunning_n_iter": config["tuning"]["n_iter"],
            "tuning_cv_folds": config["tuning"]["cv_folds"],
            "tuning_scoring": config["tuning"]["scoring"],
        })

        # Best params from tuning
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        # CV results
        mlflow.log_metrics(cv_results)

        # Predict and evaluate on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Generate and log evaluation report
        report_dir = str(resolve_path(f"reports/figures/{model_name}"))
        generate_evaluation_report(
            y_test, y_pred, y_prob,
            model=model if hasattr(model, "feature_importances_") else None,
            feature_names=feature_names,
            output_dir=report_dir
        )
        mlflow.log_artifacts(report_dir, artifact_path=f"evaluation_reports/{model_name}")

        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")

        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
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
            f"prepare_data() returned {len(result)} values, expected 6."
        )

    X_train, X_test, y_train, y_test, preprocessor, feature_names = result

    # Sanity check: dung shapes
    assert X_train.ndim == 2, f"X_train should be 2D, got {X_train.shape}"
    assert y_train.ndim == 1, f"y_train should be 1D, got {y_train.shape}"

    logger.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    logger.info(f"Features: {len(feature_names)}")

    base_models = get_base_models(config)
    param_dists = get_param_distribution(config)
    results = {}

    for name, model in base_models.items():
        logger.info(f"[{name.upper()}]")
        logger.info("Hyperparameter tuning")

        best_model, best_params, cv_results = tune_hyperparameter(
            name, model, param_dists[name],
            X_train, y_train, config
        )
        metric, run_id = train_and_track(
            name, best_model, best_params, cv_results,
            X_train, X_test, y_train, y_test, feature_names, config
        )

        results[name] = {
            "model": best_model, 
            "best_params": best_params, 
            "cv_results": cv_results,
            "metrics": metric,
            "run_id": run_id
            }

    # Select and save best
    best_name = select_best_model(results)
    best_model = results[best_name]["model"]
    save_best_model(best_model, preprocessor, config)

    # Summary table
    logger.info("Model Comparison:")
    logger.info(f"  {'Model':<25} {'CV AUC':>8} {'ROC-AUC':>9} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    logger.info("  " + "-" * 65)

    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]["metrics"]["roc_auc"], reverse=True
        )
    for name, r in sorted_results:
        m = r["metrics"]
        cv = r["cv_results"].get(f"cv_best_{config['tuning']['scoring']}", "N/A")
        cv = f"{cv:.4f}" if isinstance(cv, (int, float)) else str(cv)
        tag = " <- BEST" if name == best_name else ""
        logger.info(
            f"  {name:<25} {cv:>8} {m['roc_auc']:>9.4f} "
            f"{m['f1_score']:>7.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f}{tag}"
        )

    logger.info("Training pipeline complete!")
    logger.info("Run: mlflow ui --port 5000  ->  http://localhost:5000")
    logger.info("=" * 60)

def hyperparameter_tuning():
    """Optional: implement hyperparameter tuning logic here."""
    pass

if __name__ == "__main__":
    run_training_pipeline()