"""Konfigurasi global untuk project stroke prediction."""
from pathlib import Path

# ===== Paths =====
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "healthcare-dataset-stroke-data.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "stroke_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.txt"

# ===== Columns =====
TARGET = "stroke"
DROP_COLS = ["id"]

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CATEGORICAL_FEATURES = [
    "gender", "ever_married", "work_type",
    "Residence_type", "smoking_status",
]

# ===== Training =====
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# ===== Model Hyperparameters (tuned) =====
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,   # handle imbalance
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}

LGBM_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 5,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}