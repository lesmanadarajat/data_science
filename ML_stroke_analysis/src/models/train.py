"""Training pipeline dengan XGBoost-only + scale_pos_weight besar."""
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.config import MODEL_PATH, RANDOM_STATE, CV_FOLDS
from src.data.preprocess import preprocess_and_split


def train():
    X_train, X_test, y_train, y_test, preprocessor, feat_names = preprocess_and_split(save=True)

    # SMOTE dengan rasio agresif — perbanyak kelas minoritas
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3, sampling_strategy=0.5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[train] After SMOTE: {X_train_res.shape}, dist={np.bincount(y_train_res)}")

    # XGBoost dengan scale_pos_weight=20 (boost kelas minoritas 20x)
    model = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.8,
        scale_pos_weight=20,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.5,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        tree_method="hist",
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_res, y_train_res, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    print(f"[train] CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Fit final
    model.fit(X_train_res, y_train_res)

    # Simpan
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[train] Model saved -> {MODEL_PATH}")

    return model, X_test, y_test, feat_names


if __name__ == "__main__":
    train()