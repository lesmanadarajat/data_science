"""Benchmark 4 algoritma ML untuk prediksi stroke.
Hasil: Tabel 2.3 Perbandingan Performa Algoritma Machine Learning.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              recall_score, precision_score, f1_score,
                              accuracy_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_STATE, CV_FOLDS
from src.data.preprocess import preprocess_and_split


def benchmark():
    print("=" * 70)
    print("BENCHMARK 4 ALGORITMA ML UNTUK PREDIKSI STROKE")
    print("=" * 70 + "\n")

    # Load data
    X_train, X_test, y_train, y_test, pre, feats = preprocess_and_split(save=False)
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}\n")

    # SMOTE
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3, sampling_strategy=0.5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] Setelah SMOTE: {X_res.shape}\n")

    # ========== DEFINISIKAN 4 MODEL ==========
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            num_leaves=31, class_weight="balanced",
            random_state=RANDOM_STATE, verbose=-1
        ),
        "XGBoost + SMOTE": XGBClassifier(
            n_estimators=800, max_depth=5, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.8,
            scale_pos_weight=20, min_child_weight=2,
            gamma=0.1, reg_alpha=0.5, reg_lambda=1.5,
            eval_metric="aucpr",
            random_state=RANDOM_STATE, tree_method="hist"
        ),
    }

    # ========== TRAIN & EVALUASI ==========
    results = []

    for name, model in models.items():
        print(f"🔄 Training {name}...")

        model.fit(X_res, y_res)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.10).astype(int)

        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        results.append({
            "Algoritma": name,
            "ROC-AUC": round(roc, 2),
            "PR-AUC": round(pr, 2),
            "Recall": round(rec, 2),
            "Precision": round(prec, 2),
            "F1-Score": round(f1, 2),
            "Accuracy": round(acc, 2),
        })
        print(f"   ✅ ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Recall: {rec:.4f}\n")

    # ========== TABEL HASIL ==========
    df_results = pd.DataFrame(results)
    print("=" * 70)
    print("TABEL 2.3 PERBANDINGAN PERFORMA ALGORITMA MACHINE LEARNING")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)

    # Simpan CSV
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "Tabel_2.3_Benchmark.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # ========== GAMBAR TABEL PNG ==========
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=200)
        ax.axis("off")

        ax.text(0.5, 0.98,
                "Tabel 2.3  Perbandingan Performa Algoritma Machine Learning",
                ha="center", va="top", fontsize=13, fontweight="bold",
                family="Times New Roman", color="#1E293B",
                transform=ax.transAxes)

        headers = list(df_results.columns)
        rows = df_results.values.tolist()
        cell_data = [headers] + [[str(x) for x in row] for row in rows]

        n_cols = len(headers)
        col_widths = [0.22] + [(0.78 / (n_cols - 1))] * (n_cols - 1)

        table = ax.table(
            cellText=cell_data,
            colWidths=col_widths,
            cellLoc="center",
            loc="center",
            bbox=[0.02, 0.02, 0.96, 0.82],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for i in range(len(cell_data)):
            for j in range(n_cols):
                cell = table[(i, j)]
                cell.set_edgecolor("#CBD5E1")
                cell.set_linewidth(0.8)
                if i == 0:
                    cell.set_facecolor("#4F46E5")
                    cell.set_text_props(color="white", weight="bold",
                                        family="Times New Roman")
                    cell.set_height(0.15)
                elif cell_data[i][0] == "XGBoost + SMOTE":
                    cell.set_facecolor("#FEF2F2")
                    cell.set_text_props(color="#991B1B", weight="bold",
                                        family="Times New Roman")
                    cell.set_height(0.16)
                else:
                    cell.set_facecolor("#FFFFFF" if i % 2 else "#F8FAFC")
                    cell.set_text_props(color="#1E293B",
                                        family="Times New Roman")
                    cell.set_height(0.16)

        fig_dir = Path("reports/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_png = fig_dir / "Tabel_2.3_Benchmark.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close()
        print(f"[SAVED] {out_png}")
    except Exception as e:
        print(f"[WARN] Gagal buat PNG: {e}")

    return df_results


if __name__ == "__main__":
    benchmark()