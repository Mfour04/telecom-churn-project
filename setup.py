from setuptools import setup, find_packages

setup(
    name="telco-churn-prediction",
    version="1.0.0",
    description="End-to-end ML pipeline for Telco customer churn prediction",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "pydantic>=2.6.0",
        "mlflow>=2.12.0",
        "PyYAML>=6.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "httpx>=0.27.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "isort>=5.13.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "churn-train=src.models.train:run_training_pipeline",
        ]
    }
)
