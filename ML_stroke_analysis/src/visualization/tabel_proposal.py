"""Generate tabel cantik (PNG) untuk proposal INOTEK dari dataset stroke."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Path
DATA_PATH = Path("data/raw/healthcare-dataset-stroke-data.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Warna tema (soft pastel, profesional)
COLOR_HEADER_BG = "#4F46E5"      # Ungu tua
COLOR_HEADER_TEXT = "#FFFFFF"    # Putih
COLOR_ROW_ALT = "#F8FAFC"        # Abu sangat muda
COLOR_ROW_WHITE = "#FFFFFF"      # Putih
COLOR_BORDER = "#CBD5E1"         # Abu netral
COLOR_TEXT = "#1E293B"           # Slate tua
COLOR_TARGET = "#FEF2F2"         # Merah sangat muda untuk baris "Stroke"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def get_feature_data(df):
    """Data Tabel 2.1 — hardcoded dari analisis dataset."""
    fitur_data = [
        ["1",  "Gender",            "Kategorikal",   "Jenis kelamin pasien",     "Male / Female"],
        ["2",  "Age",               "Numerik",       "Usia pasien",               "0,08 – 82 tahun"],
        ["3",  "Hypertension",      "Biner",         "Riwayat hipertensi",        "0 / 1"],
        ["4",  "Heart Disease",     "Biner",         "Riwayat penyakit jantung",  "0 / 1"],
        ["5",  "Ever Married",      "Kategorikal",   "Status pernikahan",         "Yes / No"],
        ["6",  "Work Type",         "Kategorikal",   "Jenis pekerjaan",           "5 kategori"],
        ["7",  "Residence Type",    "Kategorikal",   "Tipe tempat tinggal",       "Urban / Rural"],
        ["8",  "Avg Glucose Level", "Numerik",       "Kadar glukosa rata-rata",   "55 – 272 mg/dL"],
        ["9",  "BMI",               "Numerik",       "Indeks Massa Tubuh",        "10,3 – 97,6"],
        ["10", "Smoking Status",    "Kategorikal",   "Status merokok",            "3 kategori"],
        ["11", "Stroke",            "Biner (target)","Riwayat stroke",            "0 / 1"],
    ]
    return fitur_data


def draw_tabel_2_1(df, fitur_data):
    """Gambar Tabel 2.1 sebagai PNG cantik."""
    headers = ["No", "Fitur", "Tipe Data", "Deskripsi", "Rentang Nilai"]
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=200)
    ax.axis("off")
    
    # Judul di atas tabel
    ax.text(0.5, 0.97, "Tabel 2.1  Fitur Dataset dan Deskripsinya",
            ha="center", va="top", fontsize=14, fontweight="bold",
            family="Times New Roman", color=COLOR_TEXT,
            transform=ax.transAxes)
    
    # Buat tabel
    n_rows = len(fitur_data) + 1  # +1 untuk header
    n_cols = len(headers)
    
    table = ax.table(
        cellText=[headers] + fitur_data,
        colWidths=[0.06, 0.20, 0.15, 0.32, 0.24],
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.90],
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_edgecolor(COLOR_BORDER)
            cell.set_linewidth(0.8)
            
            # Header row
            if i == 0:
                cell.set_facecolor(COLOR_HEADER_BG)
                cell.set_text_props(color=COLOR_HEADER_TEXT,
                                    weight="bold", family="Times New Roman")
                cell.set_height(0.06)
            else:
                # Baris target "Stroke" → warna khusus
                if fitur_data[i-1][1] == "Stroke":
                    cell.set_facecolor(COLOR_TARGET)
                    cell.set_text_props(color="#991B1B",
                                        weight="bold", family="Times New Roman")
                else:
                    # Zebra striping
                    if i % 2 == 0:
                        cell.set_facecolor(COLOR_ROW_ALT)
                    else:
                        cell.set_facecolor(COLOR_ROW_WHITE)
                    cell.set_text_props(color=COLOR_TEXT,
                                        family="Times New Roman")
                cell.set_height(0.065)
    
    # Simpan
    out = OUTPUT_DIR / "Tabel_2.1_Fitur_Dataset.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def draw_tabel_2_2(df):
    """Gambar Tabel 2.2 Distribusi Kelas sebagai PNG cantik."""
    counts = df["stroke"].value_counts().sort_index()
    total = len(df)
    
    data = [
        ["Tidak Stroke (0)", f"{counts[0]:,}".replace(",", "."), f"{counts[0]/total*100:.1f}%"],
        ["Stroke (1)",       f"{counts[1]:,}".replace(",", "."), f"{counts[1]/total*100:.1f}%"],
        ["Total",            f"{total:,}".replace(",", "."),    "100%"],
    ]
    headers = ["Kelas", "Jumlah", "Persentase"]
    
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
    ax.axis("off")
    
    ax.text(0.5, 0.98, "Tabel 2.2  Distribusi Kelas pada Dataset",
            ha="center", va="top", fontsize=13, fontweight="bold",
            family="Times New Roman", color=COLOR_TEXT,
            transform=ax.transAxes)
    
    table = ax.table(
        cellText=[headers] + data,
        colWidths=[0.45, 0.28, 0.27],
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.82],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    for i in range(len(data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_edgecolor(COLOR_BORDER)
            cell.set_linewidth(0.8)
            
            if i == 0:
                cell.set_facecolor(COLOR_HEADER_BG)
                cell.set_text_props(color=COLOR_HEADER_TEXT,
                                    weight="bold", family="Times New Roman")
                cell.set_height(0.15)
            elif i == 3:  # Baris Total
                cell.set_facecolor("#F1F5F9")
                cell.set_text_props(color=COLOR_TEXT, weight="bold",
                                    family="Times New Roman")
                cell.set_height(0.16)
            else:
                if i == 2:  # Baris Stroke
                    cell.set_facecolor(COLOR_TARGET)
                    cell.set_text_props(color="#991B1B", weight="bold",
                                        family="Times New Roman")
                else:
                    cell.set_facecolor(COLOR_ROW_WHITE)
                    cell.set_text_props(color=COLOR_TEXT,
                                        family="Times New Roman")
                cell.set_height(0.18)
    
    out = OUTPUT_DIR / "Tabel_2.2_Distribusi_Kelas.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def draw_bar_chart_distribusi(df):
    """Bonus: Grafik batang distribusi kelas (untuk Latar Belakang)."""
    counts = df["stroke"].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    
    bars = ax.bar(
        ["Tidak Stroke", "Stroke"],
        [counts[0], counts[1]],
        color=["#10B981", "#EF4444"],
        edgecolor="white",
        linewidth=2,
        width=0.55,
    )
    
    # Label di atas bar
    for bar, val in zip(bars, [counts[0], counts[1]]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 80,
                f"{val:,}".replace(",", "."),
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                family="Times New Roman", color=COLOR_TEXT)
    
    ax.set_title("Distribusi Kelas Dataset Stroke",
                 fontsize=14, fontweight="bold", family="Times New Roman",
                 pad=15)
    ax.set_ylabel("Jumlah Pasien", fontsize=12, family="Times New Roman")
    ax.set_ylim(0, counts[0] * 1.15)
    
    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BORDER)
    ax.spines["bottom"].set_color(COLOR_BORDER)
    ax.tick_params(axis="both", labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
    
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    out = OUTPUT_DIR / "Grafik_Distribusi_Kelas.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def main():
    print("=" * 60)
    print("GENERATE TABEL & GRAFIK UNTUK PROPOSAL INOTEK")
    print("=" * 60 + "\n")
    
    df = load_data()
    fitur_data = get_feature_data(df)
    
    draw_tabel_2_1(df, fitur_data)
    draw_tabel_2_2(df)
    draw_bar_chart_distribusi(df)
    
    print("\n" + "=" * 60)
    print("✅ SELESAI! File tersimpan di reports/figures/:")
    print("   📊 Tabel_2.1_Fitur_Dataset.png")
    print("   📊 Tabel_2.2_Distribusi_Kelas.png")
    print("   📈 Grafik_Distribusi_Kelas.png")
    print("=" * 60)


if __name__ == "__main__":
    main()