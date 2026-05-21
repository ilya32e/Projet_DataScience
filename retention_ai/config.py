from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"

DATASET_FILENAME = "customer_churn_business_dataset.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME
KAGGLE_DATASET_SLUG = "miadul/customer-churn-prediction-business-dataset"

TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25
CV_FOLDS = 4

LEADERBOARD_PATH = METRICS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = METRICS_DIR / "feature_importance.csv"
SCHEMA_PATH = ARTIFACTS_DIR / "schema.json"
OVERVIEW_PATH = ARTIFACTS_DIR / "training_overview.json"
SCORED_CUSTOMERS_PATH = ARTIFACTS_DIR / "scored_customers.csv"
FINAL_MODEL_BUNDLE_PATH = MODELS_DIR / "final_model_bundle.joblib"

SECONDARY_METRICS_DIR = METRICS_DIR / "secondary"
RAR_LEADERBOARD_PATH = SECONDARY_METRICS_DIR / "revenue_at_risk.csv"
CLV_LEADERBOARD_PATH = SECONDARY_METRICS_DIR / "clv.csv"
ENG_REG_LEADERBOARD_PATH = SECONDARY_METRICS_DIR / "engagement_regression.csv"
ENG_CLF_LEADERBOARD_PATH = SECONDARY_METRICS_DIR / "engagement_classification.csv"
RAR_IMPORTANCE_PATH = SECONDARY_METRICS_DIR / "rar_importance.csv"
CLV_IMPORTANCE_PATH = SECONDARY_METRICS_DIR / "clv_importance.csv"
ENG_IMPORTANCE_PATH = SECONDARY_METRICS_DIR / "engagement_importance.csv"

