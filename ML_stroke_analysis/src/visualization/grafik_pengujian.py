"""Generate 1 GAMBAR GABUNGAN pengujian skenario untuk Lampiran 5.1.
Layout: 1 gambar dengan 4 panel (Tabel, Bar, Trend, Pie).
Judul tanpa prefix nomor lampiran.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

from src.config import MODEL_PATH
from src.data.preprocess import get_feature_engineer

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    print("[INFO] Load model & preprocessor...")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load("models/scaler.pkl")
    return model, preprocessor


def predict_one(model, preprocessor, data: dict, threshold=0.10):
    df = pd.DataFrame([data])
    df = get_feature_engineer(df)
    X = preprocessor.transform(df)
    return model.predict_proba(X)[0, 1]


# ============================================================
# 10 SKENARIO
# ============================================================
def get_scenarios():
    return [
        {"gender": "Male", "age": 30, "hypertension": 0, "heart_disease": 0,
         "ever_married": "No", "work_type": "Private", "Residence_type": "Urban",
         "avg_glucose_level": 85, "bmi": 22, "smoking_status": "never smoked",
         "label": "Muda, Sehat"},
        {"gender": "Male", "age": 45, "hypertension": 0, "heart_disease": 0,
         "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
         "avg_glucose_level": 100, "bmi": 25, "smoking_status": "never smoked",
         "label": "Dewasa, Sehat"},
        {"gender": "Female", "age": 55, "hypertension": 1, "heart_disease": 0,
         "ever_married": "Yes", "work_type": "Govt_job", "Residence_type": "Rural",
         "avg_glucose_level": 120, "bmi": 28, "smoking_status": "formerly smoked",
         "label": "Pra-Lansia, Hipertensi"},
        {"gender": "Male", "age": 65, "hypertension": 1, "heart_disease": 0,
         "ever_married": "Yes", "work_type": "Self-employed", "Residence_type": "Urban",
         "avg_glucose_level": 150, "bmi": 30, "smoking_status": "smokes",
         "label": "Lansia, Hipertensi, Merokok"},
        {"gender": "Male", "age": 70, "hypertension": 1, "heart_disease": 1,
         "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
         "avg_glucose_level": 180, "bmi": 32, "smoking_status": "smokes",
         "label": "Lansia, Multi Risiko"},
        {"gender": "Male", "age": 80, "hypertension": 1, "heart_disease": 1,
         "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
         "avg_glucose_level": 228, "bmi": 38, "smoking_status": "smokes",
         "label": "Lansia, Semua Risiko"},
        {"gender": "Female", "age": 75, "hypertension": 1, "heart_disease": 0,
         "ever_married": "Yes", "work_type": "Private", "Residence_type": "Rural",
         "avg_glucose_level": 160, "bmi": 29, "smoking_status": "formerly smoked",
         "label": "Lansia Perempuan"},
        {"gender": "Female", "age": 60, "hypertension": 0, "heart_disease": 0,
         "ever_married": "Yes", "work_type": "Govt_job", "Residence_type": "Urban",
         "avg_glucose_level": 130, "bmi": 27, "smoking_status": "never smoked",
         "label": "Lansia, Tanpa Hipertensi"},
        {"gender": "Male", "age": 68, "hypertension": 1, "heart_disease": 0,
         "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
         "avg_glucose_level": 175, "bmi": 31, "smoking_status": "smokes",
         "label": "Lansia, Hipertensi, Merokok"},
        {"gender": "Female", "age": 40, "hypertension": 0, "heart_disease": 0,
         "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
         "avg_glucose_level": 95, "bmi": 24, "smoking_status": "never smoked",
         "label": "Dewasa, Sehat"},
    ]


# ============================================================
# WARNA
# ============================================================
def get_color(prob):
    if prob >= 0.50:
        return "#EF4444"
    elif prob >= 0.20:
        return "#F59E0B"
    return "#10B981"


def get_kat(prob):
    if prob >= 0.50:
        return "Tinggi"
    elif prob >= 0.20:
        return "Sedang"
    return "Rendah"


# ============================================================
# GAMBAR GABUNGAN (4 PANEL DALAM 1)
# ============================================================
def plot_all_in_one(scenarios, probs):
    """1 gambar dengan 4 panel: Tabel, Bar, Trend, Pie."""

    fig = plt.figure(figsize=(16, 13), dpi=200)

    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.3, 1.2, 0.9],
        hspace=0.35, wspace=0.22,
        left=0.06, right=0.97,
        top=0.94, bottom=0.05,
    )

    # ========== PANEL 1: TABEL ==========
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")

    headers = ["No", "Skenario", "Usia", "HTN", "HD",
               "Glukosa", "BMI", "Rokok", "Probabilitas", "Kategori"]

    rows = []
    for i, (s, p) in enumerate(zip(scenarios, probs), 1):
        htn = "Ya" if s["hypertension"] == 1 else "Tidak"
        hd = "Ya" if s["heart_disease"] == 1 else "Tidak"

        r = s["smoking_status"]
        if r == "never smoked":
            rokok = "Tidak Pernah"
        elif r == "formerly smoked":
            rokok = "Mantan"
        else:
            rokok = "Perokok"

        rows.append([
            str(i), s["label"], str(s["age"]), htn, hd,
            str(s["avg_glucose_level"]), str(s["bmi"]), rokok,
            f"{p*100:.1f}%", get_kat(p),
        ])

    table = ax_table.table(
        cellText=[headers] + rows,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for i in range(len(rows) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_edgecolor("#CBD5E1")
            cell.set_linewidth(0.8)

            if i == 0:
                cell.set_facecolor("#4F46E5")
                cell.set_text_props(color="white", weight="bold",
                                    family="Times New Roman", fontsize=10)
                cell.set_height(0.10)
            else:
                p = probs[i - 1]
                if p >= 0.50:
                    bg, color = "#FEE2E2", "#991B1B"
                elif p >= 0.20:
                    bg, color = "#FEF3C7", "#92400E"
                else:
                    bg, color = "#D1FAE5", "#065F46"

                cell.set_facecolor(bg)
                cell.set_text_props(color=color, weight="bold",
                                    family="Times New Roman", fontsize=9)
                cell.set_height(0.075)

    # Judul panel tabel — TANPA nomor lampiran
    ax_table.text(0.5, 1.02,
                  "Tabel Pengujian — 10 Skenario Input",
                  ha="center", va="bottom",
                  fontsize=12, fontweight="bold",
                  family="Times New Roman", color="#1E293B",
                  transform=ax_table.transAxes)

    # ========== PANEL 2: BAR CHART ==========
    ax_bar = fig.add_subplot(gs[1, 0])

    x = np.arange(1, len(probs) + 1)
    colors = [get_color(p) for p in probs]
    bars = ax_bar.bar(x, [p * 100 for p in probs],
                      color=colors, edgecolor="white",
                      linewidth=1.8, width=0.65)

    for bar, p in zip(bars, probs):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{p*100:.1f}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            family="Times New Roman", color="#1E293B",
        )

    ax_bar.axhline(y=10, color="#6366F1", linestyle="--",
                   linewidth=1.2, alpha=0.8)
    ax_bar.axhline(y=50, color="#EF4444", linestyle="--",
                   linewidth=1.2, alpha=0.7)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"S{i}" for i in x],
                           fontsize=9, family="Times New Roman")
    ax_bar.set_ylabel("Probabilitas (%)",
                      fontsize=10, fontweight="bold",
                      family="Times New Roman")
    ax_bar.set_ylim(0, 105)
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#CBD5E1")
    ax_bar.spines["bottom"].set_color("#CBD5E1")

    for label in ax_bar.get_xticklabels() + ax_bar.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    ax_bar.set_title("Bar Chart — Probabilitas per Skenario",
                     fontsize=11, fontweight="bold",
                     family="Times New Roman", pad=10)

    legend_elements = [
        mpatches.Patch(facecolor="#10B981", edgecolor="white",
                       label="Rendah (<20%)"),
        mpatches.Patch(facecolor="#F59E0B", edgecolor="white",
                       label="Sedang (20-50%)"),
        mpatches.Patch(facecolor="#EF4444", edgecolor="white",
                       label="Tinggi (≥50%)"),
    ]
    ax_bar.legend(handles=legend_elements, loc="upper left",
                  fontsize=8, frameon=True, facecolor="white",
                  edgecolor="#CBD5E1")

    # ========== PANEL 3: PIE CHART ==========
    ax_pie = fig.add_subplot(gs[1, 1])

    rendah = sum(1 for p in probs if p < 0.20)
    sedang = sum(1 for p in probs if 0.20 <= p < 0.50)
    tinggi = sum(1 for p in probs if p >= 0.50)

    sizes = [rendah, sedang, tinggi]
    labels = [f"Rendah\n({rendah})",
              f"Sedang\n({sedang})",
              f"Tinggi\n({tinggi})"]
    colors_pie = ["#10B981", "#F59E0B", "#EF4444"]
    explode = (0.05, 0.05, 0.08)

    wedges, texts, autotexts = ax_pie.pie(
        sizes, explode=explode,
        colors=colors_pie,
        autopct="%1.0f%%",
        shadow=True, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
        textprops={"fontsize": 10, "fontweight": "bold",
                   "family": "Times New Roman"},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(13)
        autotext.set_fontweight("bold")

    ax_pie.set_title("Distribusi Kategori Risiko",
                     fontsize=11, fontweight="bold",
                     family="Times New Roman", pad=10)

    ax_pie.legend(
        wedges, labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, frameon=True,
        facecolor="white", edgecolor="#CBD5E1",
        prop={"family": "Times New Roman", "size": 9},
    )

    # ========== PANEL 4: TREND LINE ==========
    ax_line = fig.add_subplot(gs[2, :])

    probs_pct = [p * 100 for p in probs]

    ax_line.fill_between(x, probs_pct, alpha=0.15, color="#6366F1")
    ax_line.plot(x, probs_pct,
                 marker="o", markersize=9,
                 linewidth=2.5, color="#6366F1",
                 markerfacecolor="white", markeredgewidth=2.5,
                 markeredgecolor="#6366F1", zorder=3)

    for xi, pi in zip(x, probs_pct):
        ax_line.annotate(
            f"{pi:.1f}%",
            (xi, pi), textcoords="offset points",
            xytext=(0, 12), ha="center",
            fontsize=9, fontweight="bold",
            family="Times New Roman", color="#1E293B",
        )

    ax_line.axhline(y=10, color="#6366F1", linestyle="--",
                    linewidth=1.2, alpha=0.7, label="Threshold (10%)")
    ax_line.axhline(y=50, color="#EF4444", linestyle="--",
                    linewidth=1.2, alpha=0.7, label="Risiko Tinggi (50%)")

    ax_line.set_xticks(x)
    ax_line.set_xticklabels([f"S{i}" for i in x],
                            fontsize=9.5, family="Times New Roman")
    ax_line.set_xlabel("Skenario Pengujian",
                       fontsize=10, fontweight="bold",
                       family="Times New Roman")
    ax_line.set_ylabel("Probabilitas (%)",
                       fontsize=10, fontweight="bold",
                       family="Times New Roman")
    ax_line.set_ylim(-8, 105)
    ax_line.grid(alpha=0.3, linestyle="--")
    ax_line.set_axisbelow(True)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)

    for label in ax_line.get_xticklabels() + ax_line.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    ax_line.legend(loc="upper left", fontsize=8.5,
                   frameon=True, facecolor="white",
                   edgecolor="#CBD5E1", ncol=2)

    ax_line.set_title("Trend Probabilitas Antar Skenario",
                      fontsize=11, fontweight="bold",
                      family="Times New Roman", pad=8)

    # ========== JUDUL UTAMA — TANPA nomor lampiran ==========
    fig.suptitle(
        "Log Pengujian Sistem — 10 Skenario Input\n"
        "Sistem Prediksi Risiko Penyakit Stroke Berbasis Machine Learning",
        fontsize=14, fontweight="bold",
        family="Times New Roman", color="#1E293B",
        y=0.985,
    )

    # ========== FOOTER ==========
    fig.text(
        0.5, 0.01,
        "Dicetak oleh: Lesmana Luhur Darajat  |  "
        "Tanggal: 7 Oktober 2026  |  "
        "Model: XGBoost + SMOTE  |  Threshold: 0.10  |  v1.0.0",
        ha="center", va="bottom",
        fontsize=9, style="italic",
        family="Times New Roman", color="#64748B",
    )

    # ========== SIMPAN ==========
    out = OUTPUT_DIR / "Pengujian_Skenario.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("GENERATE GAMBAR GABUNGAN PENGUJIAN SKENARIO")
    print("=" * 60 + "\n")

    model, preprocessor = load_model()
    scenarios = get_scenarios()

    probs = []
    for s in scenarios:
        data = {k: v for k, v in s.items() if k != "label"}
        probs.append(predict_one(model, preprocessor, data))

    print("\n[INFO] Hasil Prediksi:")
    for i, (s, p) in enumerate(zip(scenarios, probs), 1):
        print(f"  S{i:02d} ({s['label']:30s}): {p*100:.2f}%")

    print("\n📊 Generate gambar gabungan (4 panel)...")
    plot_all_in_one(scenarios, probs)

    print("\n" + "=" * 60)
    print("✅ SELESAI! 1 gambar gabungan tersimpan di:")
    print("   📊 reports/figures/Pengujian_Skenario.png")
    print("=" * 60)


if __name__ == "__main__":
    main()