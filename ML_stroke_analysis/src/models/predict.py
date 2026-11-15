"""Prediksi untuk data baru."""
import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, SCALER_PATH, FEATURES_PATH
from src.data.preprocess import get_feature_engineer


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(SCALER_PATH)
    return model, preprocessor


def predict(input_dict: dict, threshold: float = 0.5):
    """input_dict: raw features (sama seperti CSV kolom)."""
    model, preprocessor = load_artifacts()
    df = pd.DataFrame([input_dict])
    df = get_feature_engineer(df)
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= threshold)
    return {"stroke_probability": float(prob), "prediction": pred}


if __name__ == "__main__":
    sample = {
        "gender": "Male", "age": 67.0, "hypertension": 1, "heart_disease": 0,
        "ever_married": "Yes", "work_type": "Private",
        "Residence_type": "Urban", "avg_glucose_level": 228.69,
        "bmi": 36.6, "smoking_status": "formerly smoked",
    }
    print(predict(sample))