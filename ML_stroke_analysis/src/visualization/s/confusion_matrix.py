"""Generate Confusion Matrix Model XGBoost (Gambar 2.2) — PNG cantik."""
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report

from src.config import MODEL_PATH
from src.data.preprocess import preprocess_and_split

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_confusion_matrix(threshold=0.10):
    """Hitung confusion matrix dari model di test set."""
    print("[INFO] Menghitung confusion matrix dari model...")

    _, X_test, _, y_test, _, _ = preprocess_and_split(save=False)
    model = joblib.load(MODEL_PATH)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n[INFO] Confusion Matrix (raw):\n{cm}")
    print(f"\n[INFO] Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                  target_names=["Tidak Stroke", "Stroke"],
                                  digits=4))
    return cm, y_test, y_pred


def draw_confusion_matrix(cm):
    """Gambar confusion matrix cantik."""
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    # Figure
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=200)

    # ========== HEATMAP MANUAL ==========
    cell_colors = [
        ["#D1FAE5", "#FEE2E2"],   # Baris 1 (TN, FP)
        ["#FEE2E2", "#A7F3D0"],   # Baris 2 (FN, TP)
    ]
    text_colors = [
        ["#065F46", "#991B1B"],
        ["#991B1B", "#065F46"],
    ]

    # Plot 2x2 grid
    for i in range(2):
        for j in range(2):
            rect = mpatches.Rectangle(
                (j - 0.5, 1 - i - 0.5), 1, 1,
                facecolor=cell_colors[i][j],
                edgecolor="white", linewidth=3,
            )
            ax.add_patch(rect)

            # Nilai di tengah cell
            ax.text(
                j, 1 - i,
                f"{cm[i, j]:,}".replace(",", "."),
                ha="center", va="center",
                fontsize=32, fontweight="bold",
                family="Times New Roman",
                color=text_colors[i][j],
            )

            # Persentase di bawah nilai
            pct = cm[i, j] / total * 100
            ax.text(
                j, 1 - i - 0.28,
                f"({pct:.1f}%)",
                ha="center", va="center",
                fontsize=13,
                family="Times New Roman",
                color=text_colors[i][j],
                alpha=0.8,
            )

    # ========== LABEL SUMBU ==========
    ax.text(0, 1.65, "Tidak Stroke", ha="center", va="center",
            fontsize=13, fontweight="bold", family="Times New Roman",
            color="#1E293B")
    ax.text(1, 1.65, "Stroke", ha="center", va="center",
            fontsize=13, fontweight="bold", family="Times New Roman",
            color="#1E293B")

    ax.text(-0.75, 1, "Tidak Stroke", ha="center", va="center",
            fontsize=13, fontweight="bold", family="Times New Roman",
            color="#1E293B", rotation=90)
    ax.text(-0.75, 0, "Stroke", ha="center", va="center",
            fontsize=13, fontweight="bold", family="Times New Roman",
            color="#1E293B", rotation=90)

    # Judul sumbu
    ax.text(0.5, 1.95, "PREDIKSI", ha="center", va="center",
            fontsize=14, fontweight="bold", family="Times New Roman",
            color="#4F46E5")
    ax.text(-1.05, 0.5, "AKTUAL", ha="center", va="center",
            fontsize=14, fontweight="bold", family="Times New Roman",
            color="#4F46E5", rotation=90)

    # ========== LABEL SETIAP SEL ==========
    labels = [
        ["TN (True Negative)", "FP (False Positive)"],
        ["FN (False Negative)", "TP (True Positive)"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(
                j, 1 - i + 0.32,
                labels[i][j],
                ha="center", va="center",
                fontsize=9, style="italic",
                family="Times New Roman",
                color=text_colors[i][j],
                alpha=0.9,
            )

    # ========== SETUP AXES ==========
    ax.set_xlim(-1.1, 1.5)
    ax.set_ylim(-0.6, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # ========== JUDUL — DIUBAH KE 2.2 ==========
    ax.text(
        0.5, 1.98,
        "Gambar 2.2  Confusion Matrix Model XGBoost",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold",
        family="Times New Roman",
        color="#1E293B",
        transform=ax.transData,
    )

    # ========== SUMMARY ==========
    summary = (
        f"Total Test: {total:,}  |  "
        f"Benar: {tn + tp:,} ({(tn+tp)/total*100:.1f}%)  |  "
        f"Salah: {fp + fn:,} ({(fp+fn)/total*100:.1f}%)"
    ).replace(",", ".")

    ax.text(
        0.5, -0.45,
        summary,
        ha="center", va="center",
        fontsize=11, family="Times New Roman",
        color="#475569", style="italic",
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "Confusion_Matrix.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"\n[SAVED] {out}")

    # ========== BONUS: VERSI SKLEARN STANDAR ==========
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        _, X_test, _, y_test, _, _ = preprocess_and_split(save=False)
        model = joblib.load(MODEL_PATH)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.10).astype(int)

        fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Tidak Stroke", "Stroke"]
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title("Gambar 2.2 Confusion Matrix Model XGBoost",
                     fontsize=13, fontweight="bold",
                     family="Times New Roman", pad=15)
        plt.tight_layout()
        out2 = OUTPUT_DIR / "Confusion_Matrix_v2.png"
        plt.savefig(out2, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close()
        print(f"[SAVED] {out2} (versi sklearn)")
    except Exception as e:
        print(f"[WARN] Versi sklearn gagal: {e}")


def main():
    print("=" * 60)
    print("GENERATE CONFUSION MATRIX — GAMBAR 2.2")
    print("=" * 60 + "\n")

    cm, y_test, y_pred = get_confusion_matrix(threshold=0.10)
    draw_confusion_matrix(cm)

    print("\n" + "=" * 60)
    print("✅ SELESAI! File tersimpan di:")
    print("   reports/figures/Confusion_Matrix.png")
    print("   reports/figures/Confusion_Matrix_v2.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
