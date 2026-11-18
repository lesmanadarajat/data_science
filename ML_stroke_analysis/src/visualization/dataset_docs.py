"""Generate dokumentasi dataset untuk Lampiran 4:
1. Statistik Deskriptif Dataset (tabel PNG)
2. Distribusi Kelas Stroke (bar chart + pie chart)
3. Heatmap Korelasi Antar Fitur
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import RAW_DATA

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================
def load_dataset():
    """Load dataset mentah."""
    print(f"[INFO] Load dataset dari: {RAW_DATA}")
    df = pd.read_csv(RAW_DATA)
    print(f"[INFO] Shape: {df.shape}")
    return df


# ============================================================
# 1. STATISTIK DESKRIPTIF (Tabel PNG)
# ============================================================
def plot_statistik_deskriptif(df):
    """Generate tabel statistik deskriptif dataset."""
    print("\n📊 1. Statistik Deskriptif...")

    num_cols = ["age", "avg_glucose_level", "bmi", "hypertension",
                "heart_disease", "stroke"]

    stats = []
    for col in num_cols:
        s = df[col].dropna()
        stats.append({
            "Variabel": col,
            "N": len(s),
            "Mean": f"{s.mean():.2f}",
            "Std Dev": f"{s.std():.2f}",
            "Min": f"{s.min():.2f}",
            "Q1": f"{s.quantile(0.25):.2f}",
            "Median": f"{s.median():.2f}",
            "Q3": f"{s.quantile(0.75):.2f}",
            "Max": f"{s.max():.2f}",
        })

    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False))

    # ========== GAMBAR TABEL PNG ==========
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=200)
    ax.axis("off")

    # Judul — TANPA "Lampiran 4.1"
    ax.text(
        0.5, 0.98,
        "Statistik Deskriptif Dataset Stroke",
        ha="center", va="top",
        fontsize=14, fontweight="bold",
        family="Times New Roman", color="#1E293B",
        transform=ax.transAxes,
    )

    headers = list(df_stats.columns)
    rows = df_stats.values.tolist()
    cell_data = [headers] + rows

    n_cols = len(headers)
    col_widths = [0.16] + [(0.84 / (n_cols - 1))] * (n_cols - 1)

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
                cell.set_height(0.14)
            else:
                if cell_data[i][0] == "stroke":
                    cell.set_facecolor("#FEF2F2")
                    cell.set_text_props(color="#991B1B", weight="bold",
                                        family="Times New Roman")
                else:
                    cell.set_facecolor("#FFFFFF" if i % 2 else "#F8FAFC")
                    cell.set_text_props(color="#1E293B",
                                        family="Times New Roman")
                cell.set_height(0.14)

    plt.tight_layout()
    out = OUTPUT_DIR / "Statistik_Deskriptif_Dataset.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")

    return df_stats


# ============================================================
# 2. DISTRIBUSI KELAS STROKE
# ============================================================
def plot_distribusi_kelas(df):
    """Generate bar chart + pie chart distribusi kelas stroke."""
    print("\n📊 2. Distribusi Kelas Stroke...")

    counts = df["stroke"].value_counts().sort_index()
    labels = ["Tidak Stroke (0)", "Stroke (1)"]
    values = [counts[0], counts[1]]
    colors = ["#10B981", "#EF4444"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # ========== BAR CHART ==========
    ax1 = axes[0]
    bars = ax1.bar(
        labels, values, color=colors,
        edgecolor="white", linewidth=2, width=0.55,
    )

    for bar, val in zip(bars, values):
        pct = val / sum(values) * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{val:,}".replace(",", "."),
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
            family="Times New Roman", color="#1E293B",
        )
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="center", va="center",
            fontsize=13, fontweight="bold",
            family="Times New Roman", color="white",
        )

    ax1.set_title("Distribusi Kelas (Bar Chart)",
                  fontsize=13, fontweight="bold",
                  family="Times New Roman", pad=12)
    ax1.set_ylabel("Jumlah Pasien", fontsize=11,
                   family="Times New Roman")
    ax1.set_ylim(0, max(values) * 1.15)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#CBD5E1")
    ax1.spines["bottom"].set_color("#CBD5E1")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontsize(11)

    # ========== PIE CHART ==========
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        values,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        explode=(0, 0.08),
        shadow=True,
        wedgeprops={"edgecolor": "white", "linewidth": 3},
        textprops={
            "fontsize": 13,
            "fontweight": "bold",
            "family": "Times New Roman",
            "color": "white",
        },
    )

    ax2.set_title("Distribusi Kelas (Pie Chart)",
                  fontsize=13, fontweight="bold",
                  family="Times New Roman", pad=12)

    ax2.legend(
        wedges,
        [f"{l}: {v:,}".replace(",", ".") for l, v in zip(labels, values)],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        prop={"family": "Times New Roman", "size": 11},
    )

    # Judul utama — TANPA "Lampiran 4.2"
    plt.suptitle(
        "Distribusi Kelas Stroke pada Dataset",
        fontsize=14, fontweight="bold",
        family="Times New Roman", y=1.02,
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "Distribusi_Kelas_Stroke.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


# ============================================================
# 3. HEATMAP KORELASI ANTAR FITUR
# ============================================================
def plot_heatmap_korelasi(df):
    """Generate heatmap korelasi antar fitur numerik."""
    print("\n📊 3. Heatmap Korelasi Antar Fitur...")

    num_cols = [
        "age", "hypertension", "heart_disease",
        "avg_glucose_level", "bmi", "stroke",
    ]

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=200)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=2,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Koefisien Korelasi"},
        annot_kws={
            "fontsize": 11,
            "fontweight": "bold",
            "family": "Times New Roman",
        },
        ax=ax,
        vmin=-0.5, vmax=0.5,
    )

    # Judul — TANPA "Lampiran 4.3"
    ax.set_title(
        "Heatmap Korelasi Antar Fitur Numerik",
        fontsize=13, fontweight="bold",
        family="Times New Roman", pad=15,
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45, ha="right",
        fontsize=11, family="Times New Roman",
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=11, family="Times New Roman",
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(
        "Koefisien Korelasi",
        fontsize=11, family="Times New Roman",
    )
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    plt.tight_layout()
    out = OUTPUT_DIR / "Heatmap_Korelasi_Fitur.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")

    print("\n[INFO] Korelasi dengan Stroke (sorted):")
    corr_stroke = corr["stroke"].drop("stroke").sort_values(ascending=False)
    for var, val in corr_stroke.items():
        print(f"   {var:20s}: {val:+.4f}")

    return corr


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("GENERATE DOKUMENTASI DATASET")
    print("=" * 60)

    df = load_dataset()

    plot_statistik_deskriptif(df)
    plot_distribusi_kelas(df)
    plot_heatmap_korelasi(df)

    print("\n" + "=" * 60)
    print("✅ SELESAI! 3 gambar tersimpan di reports/figures/:")
    print("   📊 Statistik_Deskriptif_Dataset.png")
    print("   📊 Distribusi_Kelas_Stroke.png")
    print("   📊 Heatmap_Korelasi_Fitur.png")
    print("=" * 60)


if __name__ == "__main__":
    main()