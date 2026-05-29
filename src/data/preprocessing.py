import pandas as pd
import numpy as np 
import yaml
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.utils.logger import get_logger

logger = get_logger(__name__)

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Giá trị "không dùng" — không phải "No" thông thường
NON_SERVICE_VALUES = {"No", "No internet service", "No phone service"}

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file. 
    
    Returns:
        Raw DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame by handling missing values and encoding categorical variables.

    Args:
        df (pd.DataFrame): The raw DataFrame to be cleaned.

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    df = df.copy()

    # Fix TotalCharges: convert to numeric (some values are whitespace strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Impute missing values in TotalCharges with median
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
            f"using MonthlyCharges × tenure (all were tenure=0 → result=0.0)"
        )
    # drop customerID as it's not useful for modeling
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped customerID (not predictive)")

    # Encode target variable: Yes -> 1, No -> 0
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    remaining_nulls = df.isnull().sum().sum()
    churn_rate = df["Churn"].mean()
    logger.info(f"After cleaning — nulls: {remaining_nulls} | churn rate: {churn_rate:.1%}")

    return df

# ── Feature Thresholds ────────────────────────────────────────────────────────

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
        # Ngưỡng phân loại high_charges
        # Dùng median thay vì hardcode 65 → tự thích nghi với dữ liệu
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
    logger.info(f"Feature thresholds saved → {path}")


def load_feature_thresholds(path: str) -> dict:
    """
    Load thresholds saved from the YAML file.

    Args:
    Path: File path (e.g., models/feature_thresholds.yaml)

    Returns:
    Dict thresholds

    Raises:
    FileNotFoundError: If the file does not exist (not yet trained)
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

def feature_engineering(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Perform feature engineering on the cleaned DataFrame.

    Args:
        df (pd.DataFrame): The cleaned DataFrame to be transformed.
        thresholds (dict): Dictionary containing the feature thresholds.

    Returns:
        DataFrame with engineered features
    """
    logger.info("Performing feature engineering...")

    #Tenure groups
    df['tenure_group'] = pd.cut(
        df['tenure'], 
        bins = [0, 12, 24, 48, 72],
        labels = ['0-1yr', '1-2yr', '2-4yr', '4-6yr'],
        include_lowest=True,
    ).astype(str)

    #High monthly charges flag 
    monthly_charges_median = thresholds["monthly_charges_median"]
    df["high_charges"] = (df["MonthlyCharges"] > monthly_charges_median).astype(int)

    logger.info(
        f"high_charges threshold (train median): {monthly_charges_median:.2f}$ "
        f"→ {df['high_charges'].mean():.1%} of records flagged as high"
    )

    #Number of services subscribed
    available_services = [c for c in SERVICE_COLS if c in df.columns]
    df['num_services'] = df[available_services].apply(
        lambda row: sum(1 for v in row if v not in NON_SERVICE_VALUES),
        axis=1
    )
    #Monthly charges per service
    df['charges_per_service'] = df['MonthlyCharges'] / (df['num_services'] + 1)

    logger.info(
        f"Features added: tenure_group, high_charges, num_services, charge_per_service "
        f"| Total columns: {df.shape[1]}"
    )
    return df

def build_preprocessor (numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a preprocessor for numerical and categorical features.

    Args:
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        A ColumnTransformer object that can be used in a pipeline.
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')

    return preprocessor

def prepare_data(config: dict) -> tuple:
    """
    Load, clean, and engineer features from the raw data, then split into training and testing sets.

    Args:
        config (dict): Configuration dictionary containing file paths and parameters.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    #Load and clean
    df = load_data(config['paths']['raw_data'])
    df = clean_data(df)

    #Define features and target
    target = config['data']['target_column']
    X_raw = df.drop(columns=[target])
    y = df[target]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    logger.info(f"Initial train-test split: {X_train_raw.shape[0]} train samples, {X_test_raw.shape[0]} test samples")

    #Caculate thresholds from train set and save for later use
    thresholds = compute_feature_thresholds(X_train_raw)
    save_feature_thresholds(thresholds, config['api']['feature_thresholds_path'])

    #Feature engineering with thresholds
    X_train_eng = feature_engineering(X_train_raw, thresholds)
    X_test_eng = feature_engineering(X_test_raw, thresholds)

    #Save processed data (train + test) for EDA and future reference
    processed_path = Path(config['paths']['processed_data'])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df_processed = pd.concat([X_train_eng, X_test_eng])
    df_processed[target] = pd.concat([y_train, y_test])
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path} (train + test combined)")

    #Define feature lists
    numerical_features = config['data']['numerical_features'] + [
        'high_charges', 'num_services', 'charges_per_service'
    ]

    categorical_features = config['data']['categorical_features'] + ['tenure_group']

    #Filter to only existing colums
    numerical_features = [f for f in numerical_features if f in X_train_eng.columns]
    categorical_features = [f for f in categorical_features if f in X_train_eng.columns]

    X_train_feat = X_train_eng[numerical_features + categorical_features]
    X_test_feat = X_test_eng[numerical_features + categorical_features]
    y = y_train

    #Build and fit preprocessor 
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train_feat)
    X_test_processed = preprocessor.transform(X_test_feat)  

    #Get feature names after encoding

    try: 
        cat_encoded = preprocessor.named_transformers_['cat']['encoder']
        cat_feature_names = cat_encoded.get_feature_names_out(categorical_features).tolist()
    except Exception:
        cat_feature_names = [f"cat{i}" for i in range(X_train_processed.shape[1] - len(numerical_features))]
    
    feature_names = numerical_features + cat_feature_names
    logger.info(f"Total features after encoding: {len(feature_names)}")
                
    #Save preprocessor 
    model_dir = Path(config['api']['model_path']).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config['api']['preprocessor_path'])
    logger.info(f"Preprocessor saved to {config['api']['preprocessor_path']}")

    return X_train_processed, X_test_processed, y_train.values, y_test.values, feature_names

def preprocess_single(data: dict, preprocessor, config:dict) -> np.array:
    """
    Preprocess a single data point for prediction.

    Args:
        data (dict): A dictionary containing the raw input data.
        preprocessor: The fitted preprocessor to transform the input data.
        config (dict): Configuration dictionary containing feature information.

    Returns:
        A numpy array of the preprocessed input data ready for prediction.
    """
    #Convert to DataFrame
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

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test, feature_names = prepare_data(config)
    logger.info(f"Data preprocessing completed successfully. Feature count: {len(feature_names)}")