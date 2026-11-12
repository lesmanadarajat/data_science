"""Preprocessing: cleaning, imputation, encoding."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

from src.config import (
    RAW_DATA, TRAIN_PATH, TEST_PATH, PROCESSED_DIR,
    TARGET, DROP_COLS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TEST_SIZE, RANDOM_STATE, SCALER_PATH,
)
from src.data.load_data import load_raw_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data mentah."""
    df = df.copy()
    # Drop kolom yang tidak berguna
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Hapus kategori 'Other' pada gender (hanya 1 row)
    df = df[df["gender"] != "Other"]

    # Imputasi BMI dengan median per group (gender + age bins)
    df["age_bin"] = pd.cut(df["age"], bins=[0, 15, 30, 50, 70, 100], labels=False)
    df["bmi"] = df.groupby(["gender", "age_bin"])["bmi"].transform(
        lambda x: x.fillna(x.median())
    )
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df = df.drop(columns=["age_bin"])

    # Ganti 'Unknown' pada smoking_status -> mode per age group
    df["smoking_status"] = df["smoking_status"].replace("Unknown", np.nan)
    df["smoking_status"] = df["smoking_status"].fillna(
        df.groupby(pd.cut(df["age"], bins=[0, 20, 40, 60, 100]))["smoking_status"]
        .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else "never smoked")
    )

    # Hapus duplikat
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def get_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    df = df.copy()
    df["glucose_bmi_ratio"] = df["avg_glucose_level"] / (df["bmi"] + 1e-6)
    df["age_glucose"] = df["age"] * df["avg_glucose_level"] / 100
    df["hypertension_hd"] = df["hypertension"] + df["heart_disease"]
    df["is_senior"] = (df["age"] >= 60).astype(int)
    df["high_glucose"] = (df["avg_glucose_level"] >= 140).astype(int)
    df["obese"] = (df["bmi"] >= 30).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer untuk numeric + categorical."""
    numeric_cols = NUMERIC_FEATURES + [
        "glucose_bmi_ratio", "age_glucose", "hypertension_hd",
        "is_senior", "high_glucose", "obese",
    ]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])
    return pre


def preprocess_and_split(save: bool = True):
    df = load_raw_data(RAW_DATA)
    df = clean_data(df)
    df = get_feature_engineer(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Ambil nama fitur
    num_cols = NUMERIC_FEATURES + [
        "glucose_bmi_ratio", "age_glucose", "hypertension_hd",
        "is_senior", "high_glucose", "obese",
    ]
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_cols = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = num_cols + cat_cols

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_df = pd.DataFrame(X_train_t, columns=feature_names)
        train_df[TARGET] = y_train.values
        test_df = pd.DataFrame(X_test_t, columns=feature_names)
        test_df[TARGET] = y_test.values
        train_df.to_csv(TRAIN_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)
        joblib.dump(preprocessor, SCALER_PATH)
        joblib.dump(feature_names, SCALER_PATH.parent / "feature_names.pkl")
        print(f"[preprocess] Train: {train_df.shape}, Test: {test_df.shape}")

    return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    preprocess_and_split(save=True)