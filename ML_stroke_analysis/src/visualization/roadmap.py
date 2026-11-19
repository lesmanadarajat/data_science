"""Generate Tabel 2.5 Roadmap Pengembangan 3 Tahun (PNG cantik)."""
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_tabel_2_5():
    """Gambar Tabel 2.5 Roadmap 3 Tahun (2026-2028)."""

    # ========== DATA TABEL ==========
    headers = ["Tahun", "Fokus", "Target Utama", "Indikator Keberhasilan"]
    data = [
        ["2026", "Pengembangan &\nPilot Project",
         "10 Puskesmas\nLampung Timur", "5.000\npengguna aktif"],
        ["2027", "Deployment &\nKolaborasi",
         "5 Kabupaten di\nProvinsi Lampung", "50.000\npengguna aktif"],
        ["2028", "Skalabilitas &\nModel Bisnis",
         "10 Provinsi\ndi Indonesia", "500.000\npengguna aktif"],
    ]

    # ========== FIGURE ==========
    fig, ax = plt.subplots(figsize=(13, 4.5), dpi=200)
    ax.axis("off")

    # Judul — Tabel 2.5
    ax.text(
        0.5, 0.98,
        "Tabel 2.5  Roadmap Pengembangan 3 Tahun (2026-2028)",
        ha="center", va="top",
        fontsize=14, fontweight="bold",
        family="Times New Roman", color="#1E293B",
        transform=ax.transAxes,
    )

    # ========== BUAT TABEL ==========
    col_widths = [0.10, 0.28, 0.32, 0.28]
    table = ax.table(
        cellText=[headers] + data,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.82],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # ========== STYLING ==========
    year_colors = {
        1: "#DBEAFE",   # 2026 → biru muda
        2: "#E0E7FF",   # 2027 → indigo muda
        3: "#F3E8FF",   # 2028 → ungu muda
    }
    year_border = {
        1: "#3B82F6",
        2: "#6366F1",
        3: "#8B5CF6",
    }

    for i in range(len(data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_edgecolor("#CBD5E1")
            cell.set_linewidth(0.8)

            if i == 0:
                # Header
                cell.set_facecolor("#4F46E5")
                cell.set_text_props(
                    color="white", weight="bold",
                    family="Times New Roman", fontsize=12,
                )
                cell.set_height(0.15)
            else:
                # Body
                cell.set_facecolor(year_colors[i])
                if j == 0:
                    cell.set_text_props(
                        color=year_border[i], weight="bold",
                        family="Times New Roman", fontsize=14,
                    )
                else:
                    cell.set_text_props(
                        color="#1E293B",
                        family="Times New Roman", fontsize=11,
                    )
                cell.set_height(0.20)

    # ========== SIMPAN PNG ==========
    out = OUTPUT_DIR / "Tabel_2.5_Roadmap.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def draw_roadmap_timeline():
    """Bonus: visual timeline roadmap."""
    fig, ax = plt.subplots(figsize=(13, 5), dpi=200)

    years = ["2026", "2027", "2028"]
    phases = ["Pengembangan &\nPilot Project",
              "Deployment &\nKolaborasi",
              "Skalabilitas &\nModel Bisnis"]
    targets = ["10 Puskesmas\nLampung Timur",
               "5 Kabupaten\nProvinsi Lampung",
               "10 Provinsi\ndi Indonesia"]
    users = ["5.000", "50.000", "500.000"]
    colors = ["#3B82F6", "#6366F1", "#8B5CF6"]

    # Timeline line
    ax.plot([0.5, 1.5, 2.5], [0.5, 0.5, 0.5],
            color="#CBD5E1", linewidth=4, zorder=1)

    for i, (x, year, phase, target, user, color) in enumerate(
        zip([0.5, 1.5, 2.5], years, phases, targets, users, colors)
    ):
        ax.scatter(x, 0.5, s=800, color=color, zorder=2,
                   edgecolor="white", linewidth=3)
        ax.text(x, 0.5, year[-2:], ha="center", va="center",
                fontsize=16, fontweight="bold",
                family="Times New Roman", color="white", zorder=3)
        ax.text(x, 0.15, year, ha="center", va="top",
                fontsize=14, fontweight="bold",
                family="Times New Roman", color=color)
        ax.text(x, 0.95, phase, ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                family="Times New Roman", color="#1E293B")
        ax.text(x, 0.80, target, ha="center", va="top",
                fontsize=10,
                family="Times New Roman", color="#475569")
        ax.text(x, -0.05, f"{user} pengguna", ha="center", va="top",
                fontsize=10, style="italic",
                family="Times New Roman", color="#64748B")

    ax.text(1.5, 1.25,
            "Gambar 2.6  Timeline Roadmap Pengembangan 2026-2028",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold",
            family="Times New Roman", color="#1E293B")

    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.5, 1.4)
    ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / "Gambar_2.6_Timeline_Roadmap.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def main():
    print("=" * 60)
    print("GENERATE TABEL 2.5 & TIMELINE ROADMAP")
    print("=" * 60 + "\n")

    draw_tabel_2_5()
    draw_roadmap_timeline()

    print("\n" + "=" * 60)
    print("✅ SELESAI! File tersimpan di reports/figures/:")
    print("   📊 Tabel_2.5_Roadmap.png")
    print("   📈 Gambar_2.6_Timeline_Roadmap.png")
    print("=" * 60)


if __name__ == "__main__":
    main()