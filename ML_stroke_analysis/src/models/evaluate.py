"""Evaluasi model pada test set."""
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)

from src.config import MODEL_PATH, METRICS_PATH, REPORTS_DIR
from src.data.preprocess import preprocess_and_split


def find_best_threshold(y_true, y_prob):
    """Cari threshold terbaik berdasarkan F1."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate():
    # ✅ Unpacking dengan urutan yang benar
    # return: X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names
    #            ↑1        ↑2        ↑3      ↑4        ↑5            ↑6
    _, X_test, _, y_test, _, feat_names = preprocess_and_split(save=False)

    model = joblib.load(MODEL_PATH)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    best_t, best_f1 = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_t).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    lines = [
        "=" * 60,
        "STROKE PREDICTION — EVALUATION REPORT",
        "=" * 60,
        f"ROC-AUC       : {auc:.4f}",
        f"PR-AUC (AP)   : {ap:.4f}",
        f"Best threshold: {best_t:.2f}",
        f"F1 (best)     : {best_f1:.4f}",
        f"Precision     : {precision_score(y_test, y_pred, zero_division=0):.4f}",
        f"Recall        : {recall_score(y_test, y_pred, zero_division=0):.4f}",
        "",
        "Confusion Matrix:",
        str(cm),
        "",
        "Classification Report:",
        report,
    ]
    output = "\n".join(lines)
    print(output)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(output)
    print(f"[evaluate] Saved -> {METRICS_PATH}")

    return {"roc_auc": auc, "pr_auc": ap, "f1": best_f1, "threshold": best_t}


if __name__ == "__main__":
    evaluate()