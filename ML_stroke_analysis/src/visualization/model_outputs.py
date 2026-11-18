"""Generate 4 output model untuk Lampiran 3:
1. ROC Curve
2. Precision-Recall Curve
3. Feature Importance Chart
4. Confusion Matrix (versi standalone)
"""
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report,
)

from src.config import MODEL_PATH
from src.data.preprocess import preprocess_and_split

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL & PREDIKSI
# ============================================================
def load_and_predict(threshold=0.10):
    """Load model & hitung probabilitas prediksi."""
    print("[INFO] Load model & hitung prediksi...")
    _, X_test, _, y_test, _, feat_names = preprocess_and_split(save=False)
    model = joblib.load(MODEL_PATH)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return model, y_test, y_prob, y_pred, feat_names


# ============================================================
# 1. ROC CURVE
# ============================================================
def plot_roc_curve(y_test, y_prob):
    """Plot ROC Curve dengan styling cantik."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)

    # Plot utama
    ax.plot(
        fpr, tpr,
        color="#4F46E5", linewidth=3,
        label=f"ROC Curve (AUC = {roc_auc:.4f})",
        zorder=3,
    )

    # Area di bawah kurva
    ax.fill_between(
        fpr, tpr, alpha=0.15, color="#4F46E5", zorder=2,
    )

    # Garis diagonal (random classifier)
    ax.plot(
        [0, 1], [0, 1],
        color="#94A3B8", linewidth=2, linestyle="--",
        label="Random Classifier (AUC = 0.50)",
        zorder=2,
    )

    # Titik optimal (best threshold)
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        s=180, color="#EF4444", zorder=5,
        edgecolor="white", linewidth=2,
        label=f"Best Threshold ({tpr[optimal_idx]-fpr[optimal_idx]:.3f})",
    )

    # Setup
    ax.set_xlabel("False Positive Rate (1 - Specificity)",
                  fontsize=12, family="Times New Roman")
    ax.set_ylabel("True Positive Rate (Sensitivity)",
                  fontsize=12, family="Times New Roman")
    ax.set_title("ROC Curve — Model XGBoost Prediksi Stroke",
                 fontsize=13, fontweight="bold",
                 family="Times New Roman", pad=15)

    ax.legend(loc="lower right", fontsize=10, frameon=True,
              facecolor="white", edgecolor="#CBD5E1")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontsize(10)

    plt.tight_layout()
    out = OUTPUT_DIR / "Lampiran_ROC_Curve.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


# ============================================================
# 2. PRECISION-RECALL CURVE
# ============================================================
def plot_precision_recall_curve(y_test, y_prob):
    """Plot Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    # Baseline (proporsi kelas positif)
    baseline = y_test.sum() / len(y_test)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)

    # Plot PR Curve
    ax.plot(
        recall, precision,
        color="#10B981", linewidth=3,
        label=f"PR Curve (AP = {ap:.4f})",
        zorder=3,
    )

    ax.fill_between(
        recall, precision, alpha=0.15, color="#10B981", zorder=2,
    )

    # Baseline
    ax.axhline(
        y=baseline, color="#EF4444", linestyle="--",
        linewidth=2, label=f"Baseline ({baseline:.3f})",
        zorder=2,
    )

    # Setup
    ax.set_xlabel("Recall (Sensitivity)",
                  fontsize=12, family="Times New Roman")
    ax.set_ylabel("Precision",
                  fontsize=12, family="Times New Roman")
    ax.set_title("Precision-Recall Curve — Model XGBoost Prediksi Stroke",
                 fontsize=13, fontweight="bold",
                 family="Times New Roman", pad=15)

    ax.legend(loc="upper right", fontsize=10, frameon=True,
              facecolor="white", edgecolor="#CBD5E1")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontsize(10)

    plt.tight_layout()
    out = OUTPUT_DIR / "Lampiran_Precision_Recall_Curve.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


# ============================================================
# 3. FEATURE IMPORTANCE CHART
# ============================================================
def plot_feature_importance(model, feat_names, top_n=15):
    """Plot Feature Importance (horizontal bar chart)."""
    # Ambil feature importance dari model
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("[WARN] Model tidak punya feature_importances_")
        return

    # Buat DataFrame & sort
    import pandas as pd
    df_imp = pd.DataFrame({
        "Fitur": feat_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=200)

    # Horizontal bar dengan gradient warna
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(df_imp)))[::-1]
    bars = ax.barh(
        df_imp["Fitur"][::-1], df_imp["Importance"][::-1],
        color=colors, edgecolor="white", linewidth=1.5,
    )

    # Label nilai di ujung bar
    for bar, val in zip(bars, df_imp["Importance"][::-1]):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}",
            ha="left", va="center",
            fontsize=9, family="Times New Roman",
            color="#1E293B",
        )

    # Setup
    ax.set_xlabel("Feature Importance Score",
                  fontsize=12, family="Times New Roman")
    ax.set_title(f"Feature Importance — Top {top_n} Fitur Model XGBoost",
                 fontsize=13, fontweight="bold",
                 family="Times New Roman", pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontsize(10)

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = OUTPUT_DIR / "Lampiran_Feature_Importance.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


# ============================================================
# 4. CONFUSION MATRIX (standalone untuk Lampiran)
# ============================================================
def plot_confusion_matrix_lampiran(y_test, y_pred):
    """Confusion matrix versi ringkas untuk Lampiran."""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # Warna 2x2 grid
    colors = [
        ["#D1FAE5", "#FEE2E2"],
        ["#FEE2E2", "#A7F3D0"],
    ]
    text_colors = [
        ["#065F46", "#991B1B"],
        ["#991B1B", "#065F46"],
    ]

    for i in range(2):
        for j in range(2):
            rect = mpatches.Rectangle(
                (j - 0.5, 1 - i - 0.5), 1, 1,
                facecolor=colors[i][j],
                edgecolor="white", linewidth=3,
            )
            ax.add_patch(rect)

            ax.text(
                j, 1 - i,
                f"{cm[i, j]:,}".replace(",", "."),
                ha="center", va="center",
                fontsize=34, fontweight="bold",
                family="Times New Roman",
                color=text_colors[i][j],
            )

            pct = cm[i, j] / total * 100
            ax.text(
                j, 1 - i - 0.3,
                f"({pct:.1f}%)",
                ha="center", va="center",
                fontsize=13,
                family="Times New Roman",
                color=text_colors[i][j],
                alpha=0.85,
            )

    # Label
    ax.text(0, 1.55, "Prediksi: Tidak Stroke",
            ha="center", va="center", fontsize=11,
            family="Times New Roman", color="#475569")
    ax.text(1, 1.55, "Prediksi: Stroke",
            ha="center", va="center", fontsize=11,
            family="Times New Roman", color="#475569")
    ax.text(-0.85, 1, "Aktual: Tidak Stroke",
            ha="center", va="center", fontsize=11,
            family="Times New Roman", color="#475569", rotation=90)
    ax.text(-0.85, 0, "Aktual: Stroke",
            ha="center", va="center", fontsize=11,
            family="Times New Roman", color="#475569", rotation=90)

    # Label TN/FP/FN/TP
    labels = [
        ["TN", "FP"],
        ["FN", "TP"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j - 0.4, 1 - i - 0.4, labels[i][j],
                    ha="center", va="center",
                    fontsize=10, style="italic",
                    family="Times New Roman",
                    color=text_colors[i][j], alpha=0.7)

    ax.set_xlim(-1.2, 1.5)
    ax.set_ylim(-0.6, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.5, 1.75,
            "Confusion Matrix — Model XGBoost",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            family="Times New Roman", color="#1E293B")

    plt.tight_layout()
    out = OUTPUT_DIR / "Lampiran_Confusion_Matrix.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("GENERATE 4 OUTPUT MODEL UNTUK LAMPIRAN 3")
    print("=" * 60 + "\n")

    # Load & prediksi
    model, y_test, y_prob, y_pred, feat_names = load_and_predict(threshold=0.10)

    # Generate 4 chart
    print("\n📊 1. ROC Curve...")
    plot_roc_curve(y_test, y_prob)

    print("\n📊 2. Precision-Recall Curve...")
    plot_precision_recall_curve(y_test, y_prob)

    print("\n📊 3. Feature Importance Chart...")
    plot_feature_importance(model, feat_names)

    print("\n📊 4. Confusion Matrix...")
    plot_confusion_matrix_lampiran(y_test, y_pred)

    print("\n" + "=" * 60)
    print("✅ SELESAI! 4 gambar tersimpan di reports/figures/:")
    print("   📈 Lampiran_ROC_Curve.png")
    print("   📈 Lampiran_Precision_Recall_Curve.png")
    print("   📊 Lampiran_Feature_Importance.png")
    print("   📊 Lampiran_Confusion_Matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    main()