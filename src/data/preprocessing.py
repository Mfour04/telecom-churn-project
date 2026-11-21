
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/data/../../  =  project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Các cột dịch vụ dùng để đếm num_services
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Giá trị "không dùng" — không phải "No" thông thường
NON_SERVICE_VALUES = {"No", "No internet service", "No phone service"}


# ── Config

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    """Resolve a relative path against the project root (absolute paths are returned as-is)."""
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


# Data Loading 

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV to DataFrame.

    Args:
        filepath: Path to the CSV file

    Returns:
        Unprocessed Raw DataFrame
    """
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# Data Cleaning

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
   Handling raw data quality issues.

    Issues addressed:
    1. TotalCharges is a string, with a value of " " (space) when tenure=0
    2. Customer ID has no predicted value → drop
    3. Churn "Yes"/"No" → 1/0

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    df = df.copy()

    # TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_mask = df["TotalCharges"].isna()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        # Điền bằng công thức nghiệp vụ: tổng tiền = tiền tháng × số tháng
        # Vì tenure=0 nên kết quả = 0.0 — đúng với thực tế
        df.loc[missing_mask, "TotalCharges"] = (
            df.loc[missing_mask, "MonthlyCharges"] * df.loc[missing_mask, "tenure"]
        )
        logger.info(
            f"Imputed {n_missing} missing TotalCharges "
            f"using MonthlyCharges x tenure (all were tenure=0 -> result=0.0)"
        )

    # customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped customerID (not predictive)")

    # Target encoding
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Check after cleaning
    remaining_nulls = df.isnull().sum().sum()
    churn_rate = df["Churn"].mean()
    logger.info(f"After cleaning - nulls: {remaining_nulls} | churn rate: {churn_rate:.1%}")

    return df


# Feature Thresholds

def compute_feature_thresholds(df_train: pd.DataFrame) -> dict:
    """
    Calculate thresholds from the train set.
    Call this function only once on the train set, then save the result.
    Do NOT call it again on the test set or during inference.

    Args:
    df_train: DataFrame of the train set (after clean, before feature engineering)

    Returns:
    Dict containing the calculated thresholds
    """
    thresholds = {
        "monthly_charges_median": float(df_train["MonthlyCharges"].median()),
    }

    logger.info("Computed feature thresholds from train set:")
    for key, val in thresholds.items():
        logger.info(f"  {key}: {val}")

    return thresholds


def save_feature_thresholds(thresholds: dict, path: str) -> None:
    """
    Save thresholds to a YAML file for reuse during inference.

    Args:
    thresholds: Dict from compute_feature_thresholds()
    path: Path to the output file (e.g., models/feature_thresholds.yaml)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"feature_thresholds": thresholds}, f, default_flow_style=False)
    logger.info(f"Feature thresholds saved -> {path}")


def load_feature_thresholds(path: str) -> dict:
    """
    Load saved thresholds from the YAML file.

    Args:
    Path: File path (e.g., models/feature_thresholds.yaml)

    Returns:
    Dict thresholds

    Raises:
    FileNotFoundError: If the file does not exist (not trained)
    """
    threshold_path = Path(path)
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Thresholds file not found: {path}\n"
            "Run training pipeline first: python src/models/train.py"
        )

    with open(threshold_path, "r") as f:
        data = yaml.safe_load(f)

    thresholds = data["feature_thresholds"]
    logger.info(f"Loaded feature thresholds from {path}: {thresholds}")
    return thresholds


# ── Feature Engineering ───────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Create new features from existing data.

    Requires thresholds as a parameter — DO NOT recalculate from df.
    This ensures that train/test/inference use the same threshold.

    Args:
    df: Cleaned DataFrame
    thresholds: Dict from compute_feature_thresholds() or load_feature_thresholds()
    Required key: "monthly_charges_median"

    Returns:
    DataFrame with newly added features
    """
    logger.info("Engineering features...")
    df = df.copy()

    # ── Feature 1: tenure_group ───────────────────────────────────────────────
    df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    include_lowest=True,
    )

    df["tenure_group"] = (
        df["tenure_group"]
        .cat.add_categories(["Unknown"])
        .fillna("Unknown")
        .astype(str)
    )
    # ── Feature 2: high_charges ───────────────────────────────────────────────
    monthly_charges_median = thresholds["monthly_charges_median"]
    df["high_charges"] = (df["MonthlyCharges"] > monthly_charges_median).astype(int)

    logger.info(
        f"high_charges threshold (train median): {monthly_charges_median:.2f}$ "
        f"-> {df['high_charges'].mean():.1%} of records flagged as high"
    )

    # ── Feature 3: num_services ───────────────────────────────────────────────
    available_service_cols = [c for c in SERVICE_COLS if c in df.columns]
    df["num_services"] = df[available_service_cols].apply(
        lambda row: sum(1 for v in row if v not in NON_SERVICE_VALUES),
        axis=1,
    )

    # ── Feature 4: charge_per_service ─────────────────────────────────────────
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

    logger.info(
        f"Features added: tenure_group, high_charges, num_services, charge_per_service "
        f"| Total columns: {df.shape[1]}"
    )
    return df


# ── Sklearn Preprocessor ──────────────────────────────────────────────────────

def build_preprocessor(
    numerical_features: list,
    categorical_features: list,
) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer to handle 2 types of features.

    Numerical pipeline:
    SimpleImputer(median) → StandardScaler

    Categorical pipeline:
    SimpleImputer(most_frequent) → OneHotEncoder

    Args:
    numerical_features: List of numeric column names
    categorical_features: List of categorical column names

    Returns:
    ColumnTransformer not yet fitted
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",  # giá trị lạ lúc inference → vector 0
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # bỏ qua các cột không được khai báo
    )

    return preprocessor


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def prepare_data(config: dict) -> tuple:
    """
    Pipeline đầy đủ từ raw data đến dữ liệu sẵn sàng train.

    Thứ tự:
        1. load_data
        2. clean_data
        3. train_test_split  ← PHẢI chia trước khi tính threshold
        4. compute_feature_thresholds(X_train only)
        5. save_feature_thresholds
        6. feature_engineering(train, thresholds)
        7. feature_engineering(test, thresholds)   ← dùng cùng threshold
        8. build_preprocessor → fit(train) → transform(train + test)
        9. save preprocessor

    Args:
        config: Dict từ load_config()

    Returns:
        (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    # ── 1. Load & clean ───────────────────────────────────────────────────────
    df = load_data(config["paths"]["raw_data"])
    df = clean_data(df)

    # ── 2. Tách target ────────────────────────────────────────────────────────
    target = config["data"]["target_column"]
    X_raw = df.drop(columns=[target])
    y = df[target]

    # ── 3. Train/test split TRƯỚC feature engineering ─────────────────────────
    # Quan trọng: tính threshold trên train set → split phải đến trước
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    logger.info(f"Split -> train: {len(X_train_raw):,} | test: {len(X_test_raw):,}")

    # ── 4. Tính threshold CHỈ từ train set ───────────────────────────────────
    thresholds = compute_feature_thresholds(X_train_raw)

    # ── 5. Lưu thresholds ────────────────────────────────────────────────────
    save_feature_thresholds(thresholds, config["api"]["thresholds_path"])

    # ── 6. Feature engineering với threshold đã tính ─────────────────────────
    X_train_eng = feature_engineering(X_train_raw, thresholds)
    X_test_eng  = feature_engineering(X_test_raw,  thresholds)  # dùng cùng threshold

    # ── 7. Lưu processed data (train + test gộp lại) ─────────────────────────
    processed_path = Path(config["paths"]["processed_data"])
    processed_path.mkdir(parents=True, exist_ok=True)

    df_processed = pd.concat([X_train_eng, X_test_eng])
    df_processed[target] = pd.concat([y_train, y_test])
    df_processed.to_csv(processed_path / "processed_data.csv", index=False)
    logger.info(f"Processed data saved -> {processed_path / 'processed_data.csv'}")

    # ── 8. Định nghĩa feature lists ───────────────────────────────────────────
    numerical_features = config["data"]["numerical_features"] + [
        "high_charges", "num_services", "charge_per_service"
    ]
    categorical_features = config["data"]["categorical_features"] + ["tenure_group"]

    # Chỉ giữ các cột thực sự tồn tại trong DataFrame
    numerical_features   = [f for f in numerical_features   if f in X_train_eng.columns]
    categorical_features = [f for f in categorical_features if f in X_train_eng.columns]

    X_train_feat = X_train_eng[numerical_features + categorical_features]
    X_test_feat  = X_test_eng[numerical_features  + categorical_features]

    # ── 9. Build & fit preprocessor ──────────────────────────────────────────
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    # fit CHỈ trên train, transform trên cả hai
    X_train_processed = preprocessor.fit_transform(X_train_feat)
    X_test_processed  = preprocessor.transform(X_test_feat)

    # ── 10. Lấy tên feature sau OneHotEncoder ─────────────────────────────────
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]["encoder"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    except Exception:
        n_cat = X_train_processed.shape[1] - len(numerical_features)
        cat_feature_names = [f"cat_{i}" for i in range(n_cat)]

    feature_names = numerical_features + cat_feature_names
    logger.info(f"Total features after encoding: {len(feature_names)}")

    # ── 11. Lưu preprocessor ─────────────────────────────────────────────────
    model_dir = Path(config["api"]["model_path"]).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config["api"]["preprocessor_path"])
    logger.info(f"Preprocessor saved -> {config['api']['preprocessor_path']}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


# ── Inference Helper ──────────────────────────────────────────────────────────

def preprocess_single(data: dict, preprocessor, config: dict) -> np.ndarray:
    """
    Process a record for inference (used in the API).

    Reuse the correct threshold saved from the ship's initial training.

    The most complete feature at the time the model is trained.

    Inference:
    data: A customer's Dict feature (from the API request)
    preprocessor: Matched ColumnTransformer (loaded from preprocessor.pkl)
    config: Dict config (to know the path threshold)

    Return:
    A numpy array (1, n_features) available to model.predict()
    """
    df = pd.DataFrame([data])

    # ── Fix TotalCharges nếu cần ──────────────────────────────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(
        df["MonthlyCharges"] * df["tenure"]
    )

    # ── Load threshold đã lưu từ train — KHÔNG tính lại ─────────────────────
    thresholds = load_feature_thresholds(config["api"]["thresholds_path"])

    # ── Áp dụng đúng các bước feature engineering như lúc train ──────────────
    df = feature_engineering(df, thresholds)

    return preprocessor.transform(df)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(config)
    logger.info(f"Preprocessing complete. Feature count: {len(feature_names)}")