"""Render metrics.txt sebagai PNG cantik untuk Lampiran 5.2."""
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def render_metrics():
    metrics_path = Path("reports/metrics.txt")

    if not metrics_path.exists():
        print(f"[ERROR] File tidak ditemukan: {metrics_path}")
        print("Jalankan dulu: python -m src.models.evaluate")
        return

    with open(metrics_path, "r", encoding="cp1252") as f:
        content = f.read()

    # Figure
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=200)
    ax.axis("off")

    # Background card
    rect = plt.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        facecolor="#F8FAFC", edgecolor="#4F46E5",
        linewidth=2.5, zorder=0,
    )
    ax.add_patch(rect)

    # Isi metrics.txt
    ax.text(
        0.02, 0.98, content,
        fontfamily="Consolas",
        fontsize=9,
        va="top", ha="left",
        color="#1E293B",
        zorder=1,
    )

    # Judul di atas
    ax.text(
        0.5, 1.02,
        "Lampiran 5.2  Hasil Evaluasi Lengkap Model Machine Learning",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        family="Times New Roman", color="#1E293B",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "Lampiran_5.2_Metrics.png"
    plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    render_metrics()