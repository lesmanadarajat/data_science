"""Generate Gambar 2.9 Roadmap Timeline Pengembangan 3 Tahun (2026-2028).
Versi redesigned — lebih rapi, proporsional, dan profesional.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_roadmap_timeline():
    """Gambar timeline roadmap 3 tahun — versi redesigned."""

    # ========== DATA ==========
    roadmap = [
        {
            "tahun": "2026",
            "fokus": "Pengembangan\n& Pilot Project",
            "target": "10 Puskesmas\nLampung Timur",
            "user": "5.000 pengguna",
            "warna": "#3B82F6",
            "warna_bg": "#EFF6FF",
        },
        {
            "tahun": "2027",
            "fokus": "Deployment\n& Kolaborasi",
            "target": "5 Kabupaten\nProvinsi Lampung",
            "user": "50.000 pengguna",
            "warna": "#6366F1",
            "warna_bg": "#EEF2FF",
        },
        {
            "tahun": "2028",
            "fokus": "Skalabilitas\n& Model Bisnis",
            "target": "10 Provinsi\ndi Indonesia",
            "user": "500.000 pengguna",
            "warna": "#8B5CF6",
            "warna_bg": "#F5F3FF",
        },
    ]

    # ========== FIGURE ==========
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # ========== JUDUL ==========
    ax.text(
        6, 5.5,
        "Roadmap Pengembangan 3 Tahun (2026-2028)",
        ha="center", va="center",
        fontsize=15, fontweight="bold",
        family="Times New Roman", color="#1E293B",
    )

    # ========== GARIS TIMELINE UTAMA ==========
    y_timeline = 1.0
    ax.plot(
        [1.2, 10.8], [y_timeline, y_timeline],
        color="#CBD5E1", linewidth=3,
        zorder=1, solid_capstyle="round",
    )

    # ========== CARD SETUP ==========
    card_w = 3.0
    card_h = 3.2
    card_bottom = 1.6   # posisi bawah card di atas timeline
    card_top = card_bottom + card_h

    # Posisi X center tiap card
    x_positions = [2.5, 6.0, 9.5]

    for i, (x, item) in enumerate(zip(x_positions, roadmap)):
        card_left = x - card_w / 2
        card_bottom_y = card_bottom

        # ========== SHADOW CARD ==========
        shadow = FancyBboxPatch(
            (card_left + 0.06, card_bottom_y - 0.06),
            card_w, card_h,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            facecolor="#00000012", edgecolor="none", zorder=2,
        )
        ax.add_patch(shadow)

        # ========== CARD PUTIH ==========
        card = FancyBboxPatch(
            (card_left, card_bottom_y),
            card_w, card_h,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            facecolor="white",
            edgecolor=item["warna"],
            linewidth=2.5,
            zorder=3,
        )
        ax.add_patch(card)

        # ========== HEADER WARNA (rounded top) ==========
        header_h = 0.75
        header = FancyBboxPatch(
            (card_left, card_bottom_y + card_h - header_h),
            card_w, header_h,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            facecolor=item["warna"],
            edgecolor=item["warna"],
            linewidth=2.5,
            zorder=4,
        )
        ax.add_patch(header)

        # Tutup bagian bawah header biar flat
        header_flat = plt.Rectangle(
            (card_left, card_bottom_y + card_h - header_h),
            card_w, header_h * 0.5,
            facecolor=item["warna"], edgecolor="none", zorder=5,
        )
        ax.add_patch(header_flat)

        # ========== TEKS TAHUN ==========
        ax.text(
            x, card_bottom_y + card_h - header_h / 2,
            item["tahun"],
            ha="center", va="center",
            fontsize=22, fontweight="bold",
            family="Times New Roman", color="white",
            zorder=6,
        )

        # ========== TEKS FOKUS ==========
        ax.text(
            x, card_bottom_y + card_h - header_h - 0.55,
            item["fokus"],
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            family="Times New Roman", color=item["warna"],
            zorder=6,
        )

        # ========== GARIS PEMISAH ==========
        ax.plot(
            [card_left + 0.4, card_left + card_w - 0.4],
            [card_bottom_y + card_h - header_h - 0.95,
             card_bottom_y + card_h - header_h - 0.95],
            color="#E2E8F0", linewidth=1.5, zorder=5,
        )

        # ========== TEKS TARGET ==========
        ax.text(
            x, card_bottom_y + 1.35,
            item["target"],
            ha="center", va="center",
            fontsize=11,
            family="Times New Roman", color="#334155",
            zorder=6,
        )

        # ========== BOX USER (badge di bawah) ==========
        badge_w = 2.4
        badge_h = 0.45
        badge_left = x - badge_w / 2
        badge_bottom = card_bottom_y + 0.25

        badge = FancyBboxPatch(
            (badge_left, badge_bottom),
            badge_w, badge_h,
            boxstyle="round,pad=0.02,rounding_size=0.22",
            facecolor=item["warna_bg"],
            edgecolor=item["warna"],
            linewidth=1.5,
            zorder=5,
        )
        ax.add_patch(badge)

        ax.text(
            x, badge_bottom + badge_h / 2,
            item["user"],
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            family="Times New Roman", color=item["warna"],
            zorder=6,
        )

        # ========== KONEKTOR dari card ke timeline ==========
        ax.plot(
            [x, x], [card_bottom_y, y_timeline],
            color=item["warna"], linewidth=2,
            zorder=2,
        )

        # ========== BULATAN DI TIMELINE ==========
        ax.scatter(
            x, y_timeline, s=380,
            color=item["warna"], zorder=10,
            edgecolor="white", linewidth=3,
        )

        # ========== PANAH ANTAR TAHUN ==========
        if i < len(roadmap) - 1:
            next_x = x_positions[i + 1]
            arrow = FancyArrowPatch(
                (x + 0.25, y_timeline), (next_x - 0.25, y_timeline),
                arrowstyle="->,head_length=12,head_width=7",
                color="#94A3B8", linewidth=2,
                zorder=10,
            )
            ax.add_patch(arrow)

    # ========== LABEL "MULAI" DAN "AKHIR" DI BAWAH TIMELINE ==========
    # Kiri (mulai)
    ax.text(
        x_positions[0], y_timeline - 0.4,
        "▲ Mulai",
        ha="center", va="top",
        fontsize=10, style="italic",
        family="Times New Roman", color="#64748B",
    )

    # Kanan (akhir)
    ax.text(
        x_positions[-1], y_timeline - 0.4,
        "Akhir 2028 ▲",
        ha="center", va="top",
        fontsize=10, style="italic",
        family="Times New Roman", color="#64748B",
    )

    # ========== SIMPAN ==========
    plt.tight_layout()
    out = OUTPUT_DIR / "Gambar_2.9_Roadmap_Timeline.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def main():
    print("=" * 60)
    print("GENERATE GAMBAR 2.9 — ROADMAP TIMELINE (REDESIGN)")
    print("=" * 60 + "\n")

    draw_roadmap_timeline()

    print("\n" + "=" * 60)
    print("✅ SELESAI! File tersimpan di:")
    print("   reports/figures/Gambar_2.9_Roadmap_Timeline.png")
    print("=" * 60)


if __name__ == "__main__":
    main()