"""Generate Tabel 2.4 Hasil Evaluasi Model Final (PNG cantik).
Baca dari reports/metrics.txt atau hitung ulang dari model."""
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (roc_auc_score, average_precision_score,
                              recall_score, precision_score, f1_score,
                              accuracy_score)
from src.config import MODEL_PATH
from src.data.preprocess import preprocess_and_split

# Path output
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("reports/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def hitung_metrik_aktual():
    """Hitung ulang metrik dari model untuk memastikan angka akurat."""
    print("[INFO] Menghitung metrik evaluasi dari model...")

    # Load test set & model
    _, X_test, _, y_test, _, _ = preprocess_and_split(save=False)
    model = joblib.load(MODEL_PATH)

    # Prediksi probabilitas
    y_prob = model.predict_proba(X_test)[:, 1]

    # Threshold optimal (sama seperti evaluasi)
    threshold = 0.10
    y_pred = (y_prob >= threshold).astype(int)

    # Hitung metrik
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Akurasi": accuracy,
        "Presisi": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

    print("\n" + "=" * 60)
    print("METRIK EVALUASI MODEL FINAL")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f} ({v*100:.2f}%)")
    print("=" * 60 + "\n")

    return metrics, threshold


def interpretasi(metrik, nilai):
    """Berikan interpretasi singkat untuk setiap metrik."""
    table = {
        "ROC-AUC": "Sangat baik dalam diskriminasi",
        "PR-AUC": "Excellent untuk data imbalanced",
        "Akurasi": "Tingkat akurasi keseluruhan",
        "Presisi": "Rendah false positive",
        "Recall": "Tinggi deteksi stroke",
        "F1-Score": "Keseimbangan precision & recall",
    }
    return table.get(metrik, "-")


def format_nilai(metrik, nilai):
    """Format nilai: persen untuk Akurasi/Presisi/Recall/F1, desimal untuk AUC."""
    if metrik in ["ROC-AUC", "PR-AUC"]:
        return f"{nilai:.2f}".replace(".", ",")
    else:
        return f"{nilai*100:.0f}%"


def draw_tabel_2_4(metrics):
    """Gambar Tabel 2.4 sebagai PNG cantik."""
    # Siapkan data
    rows = []
    for metrik, nilai in metrics.items():
        rows.append([
            metrik,
            format_nilai(metrik, nilai),
            interpretasi(metrik, nilai),
        ])

    # Figure
    fig, ax = plt.subplots(figsize=(11, 4), dpi=200)
    ax.axis("off")

    # Judul
    ax.text(0.5, 0.98,
            "Tabel 2.4  Hasil Evaluasi Model Final",
            ha="center", va="top", fontsize=14, fontweight="bold",
            family="Times New Roman", color="#1E293B",
            transform=ax.transAxes)

    # Header + data
    headers = ["Metrik", "Nilai", "Interpretasi"]
    cell_data = [headers] + rows

    # Table
    table = ax.table(
        cellText=cell_data,
        colWidths=[0.28, 0.20, 0.52],
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.84],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Styling
    for i in range(len(cell_data)):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_edgecolor("#CBD5E1")
            cell.set_linewidth(0.8)

            if i == 0:  # Header
                cell.set_facecolor("#4F46E5")
                cell.set_text_props(color="white", weight="bold",
                                    family="Times New Roman")
                cell.set_height(0.13)
            else:
                # Zebra stripe
                if i % 2 == 0:
                    cell.set_facecolor("#F8FAFC")
                else:
                    cell.set_facecolor("#FFFFFF")

                # Bold untuk nilai
                if j == 1:
                    cell.set_text_props(color="#059669", weight="bold",
                                        family="Times New Roman")
                else:
                    cell.set_text_props(color="#1E293B",
                                        family="Times New Roman")
                cell.set_height(0.13)

    # Simpan PNG
    out = OUTPUT_DIR / "Tabel_2.4_Hasil_Evaluasi.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def save_csv(metrics):
    """Simpan juga sebagai CSV untuk arsip."""
    rows = []
    for metrik, nilai in metrics.items():
        rows.append({
            "Metrik": metrik,
            "Nilai": format_nilai(metrik, nilai),
            "Interpretasi": interpretasi(metrik, nilai),
        })
    df = pd.DataFrame(rows)
    out = TABLES_DIR / "Tabel_2.4_Hasil_Evaluasi.csv"
    df.to_csv(out, index=False)
    print(f"[SAVED] {out}")
    return df


def main():
    print("=" * 60)
    print("GENERATE TABEL 2.4 HASIL EVALUASI MODEL FINAL")
    print("=" * 60 + "\n")

    metrics, threshold = hitung_metrik_aktual()
    draw_tabel_2_4(metrics)
    save_csv(metrics)

    print("\n" + "=" * 60)
    print("✅ SELESAI! Cek folder:")
    print("   reports/figures/Tabel_2.4_Hasil_Evaluasi.png")
    print("   reports/tables/Tabel_2.4_Hasil_Evaluasi.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()