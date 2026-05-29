import pandas as pd
import numpy as np 
import yaml
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
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
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = (
        df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']
    )

    # drop customerID as it's not useful for modeling
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Encode target variable: Yes -> 1, No -> 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    logger.info(f'Missing values after cleaning: {df.isnull().sum().sum()}')
    logger.info(f"Churn distribution: {df['Churn'].value_counts().to_dict()}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the cleaned DataFrame.

    Args:
        df (pd.DataFrame): The cleaned DataFrame to be transformed.

    Returns:
        DataFrame with engineered features
    """
    logger.info("Performing feature engineering...")

    #Tenure groups
    df['tenure_group'] = pd.cut(
        df['tenure'], 
        bins = [0, 12, 24, 48, 72],
        labels = ['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    ).astype(str)

    #High monthly charges flag 
    df['high_monthly_charges'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

    #Number of services subscribed
    service_cols = [
        'PhoneService', 'Multiplelines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['num_services'] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]), axis=1
    )

    #Monthly charges per service
    df['charges_per_service'] = df['MonthlyCharges'] / (df['num_services'] + 1)

    logger.info(f"Feature after engineering: {df.shape[1]}")
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
    df = feature_engineering(df)

    #Save processed data
    processed_path = Path(config['paths']['processed_data'])
    processed_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path / "processed_data.csv", index=False)
    logger.info(f"Processed data saved to {processed_path / 'processed_data.csv'}")

    #Define features and target
    target = config['data']['target_column']

    numerical_features = config['data']['numerical_features'] + [
        'high_charges', 'num_services', 'charges_per_service'
    ]

    categorical_features = config['data']['categorical_features'] + ['tenure_group']

    #Filter to only existing colums
    numerical_features = [f for f in numerical_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    X = df[numerical_features + categorical_features]
    y = df[target]

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )

    logger.info(f"Data split into train and test sets: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

    #Build and fit preprocessor 
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_train_processed = preprocessor.fit.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)  

    #Get feature names after encoding

    try: 
        cat_encoded = preprocessor.named_transformers_['cat']['encoder']
        cat_feature_names = cat_encoded.get_feature_names_out(categorical_features).tolist()
    except Exception:
        cat_feature_names = [f"cat{i}" for i in range(X_train_processed.shape[1] - len(numerical_features))]
    
    feature_names = numerical_features + cat_feature_names

    #Save preprocessor 
    model_dir = Path(config['paths']['model']).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config['api']['preprocessor'])
    logger.info(f"Preprocessor saved to {config['api']['preprocessor']}")

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

    #Feature engineering 
    service_cols = [
        'PhoneService', 'Multiplelines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies']

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins = [0, 12, 24, 48, 72],
        labels = ['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    ).astype(str)

    df['high_monthly_charges'] = (df['MonthlyCharges'] > 65).astype(int)

    df['num_services'] = df[[c for c in service_cols if c in df.columns]].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]), axis=1
    )
    df['charges_per_service'] = df['MonthlyCharges'] / (df['num_services'] + 1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(
        df['MonthlyCharges'] * df['tenure']
    )

    return preprocessor.transform(df)

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test, feature_names = prepare_data(config)
    logger.info(f"Data preprocessing completed successfully. Feature count: {len(feature_names)}")