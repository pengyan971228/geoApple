"""Generate F5 (main + ablation bar chart) and F6 (occlusion-stratified bar chart).

Publication-quality figures for EAAI submission:
- Okabe-Ito colorblind-safe palette
- Arial font, clean spines, 300 DPI
- No background highlights or chart junk
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "docs" / "paper_data" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication style ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe-Ito colorblind-safe palette
C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_CYAN   = "#56B4E9"
C_GREY   = "#999999"


# ── F5: Main Comparison + Ablation ─────────────────────────────
def make_f5():
    methods = [
        "E1\n(RGB baseline)",
        "E7\n(GeoApple-Seg)",
        "A1\nw/o Depth",
        "A2\nw/o Geom. Loss",
        "A3\nw/o Amodal",
        "A4\nw/o DepthDrop",
    ]
    iou  = [0.7602, 0.7719, 0.7546, 0.7718, 0.7684, 0.7182]
    prec = [0.8341, 0.8794, 0.8643, 0.8862, 0.8635, 0.8500]
    rec  = [0.8707, 0.8221, 0.8173, 0.8116, 0.8255, 0.7827]

    x = np.arange(len(methods))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)

    # Bars with hatching for redundant encoding
    b1 = ax.bar(x - w, iou,  w, label="IoU",       color=C_BLUE,   edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x,     prec, w, label="Precision",  color=C_ORANGE, edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + w, rec,  w, label="Recall",     color=C_GREEN,  edgecolor="white", linewidth=0.5)

    # Value labels (only on top)
    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5,
                    color="#333333")

    # Mark proposed method with a subtle bracket/annotation instead of highlight
    ax.annotate("Proposed", xy=(1, 0.89), fontsize=8, fontweight="bold",
                ha="center", color=C_BLUE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_BLUE,
                          lw=0.8, alpha=0.9))

    # Separator between baseline and ablations
    ax.axvline(x=1.5, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(0.5, 0.985, "Comparison", ha="center", fontsize=7.5, color="#666666",
            transform=ax.get_xaxis_transform())
    ax.text(3.5, 0.985, "Ablation Study", ha="center", fontsize=7.5, color="#666666",
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0.68, 0.92)
    ax.legend(loc="upper right", ncol=3, frameon=True, edgecolor="#cccccc",
              fancybox=False)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))
    ax.grid(axis="y", linestyle="-", alpha=0.15, color="#888888")
    ax.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"F5_main_ablation.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved F5_main_ablation.{pdf,png}")


# ── F6: Occlusion-Stratified Performance ───────────────────────
def make_f6():
    bins = [
        "Low\n[0, 0.30)\nn = 99",
        "Medium\n[0.30, 0.60)\nn = 182",
        "High\n[0.60, 1.00]\nn = 334",
        "Overall\nn = 615",
    ]
    iou_mean = [0.8291, 0.8168, 0.6728, 0.7406]
    iou_std  = [0.0829, 0.0960, 0.3029, 0.2432]
    prec     = [0.9513, 0.9426, 0.7874, 0.8597]
    rec      = [0.8661, 0.8598, 0.7203, 0.7850]

    x = np.arange(len(bins))
    w = 0.24
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)

    # All groups use same color per metric (consistent with F5)
    bars_iou = ax.bar(x - w, iou_mean, w, yerr=iou_std, label="IoU",
                      color=C_BLUE, edgecolor="white", linewidth=0.5,
                      capsize=3, error_kw={"elinewidth": 0.8, "capthick": 0.8,
                                           "ecolor": "#555555"})
    bars_p = ax.bar(x, prec, w, label="Precision",
                    color=C_ORANGE, edgecolor="white", linewidth=0.5)
    bars_r = ax.bar(x + w, rec, w, label="Recall",
                    color=C_GREEN, edgecolor="white", linewidth=0.5)

    # Value annotations
    for i, (v, s) in enumerate(zip(iou_mean, iou_std)):
        ax.text(i - w, v + s + 0.015, f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color="#333333")
    for bars, vals in [(bars_p, prec), (bars_r, rec)]:
        for i, v in enumerate(vals):
            ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                    v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0.35, 1.08)
    ax.legend(loc="upper center", ncol=3, frameon=True, edgecolor="#cccccc",
              fancybox=False, bbox_to_anchor=(0.5, 1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
    ax.grid(axis="y", linestyle="-", alpha=0.15, color="#888888")
    ax.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"F6_occlusion_stratified.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved F6_occlusion_stratified.{pdf,png}")


if __name__ == "__main__":
    make_f5()
    make_f6()
