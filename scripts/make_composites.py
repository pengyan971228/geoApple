"""Build composite/schematic figures for EAAI paper.

Publication-quality diagrams with professional color palette.

F1 Pipeline overview (schematic)
F2 Dataset overview (samples + stats)
F3 Architecture diagram (schematic)
F7 Qualitative 3-row composite (low/mid/high)
F8 Amodal ablation standalone
F9 Failure cases composite
"""
from pathlib import Path
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).parent.parent
FIG = ROOT / "docs" / "paper_data" / "figures"
QUAL = FIG / "qualitative"
FAIL = FIG / "failures"

# ── Publication style ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Professional muted palette (inspired by academic papers)
_PAL = {
    "input_rgb":  "#B8D4E3",   # soft steel blue
    "input_depth":"#A8C6A0",   # soft sage green
    "backbone":   "#D6D6D6",   # neutral grey
    "head":       "#F5D5A0",   # warm sand
    "mask":       "#C4B0D5",   # soft lavender
    "loss":       "#F2A7A0",   # soft coral
    "output":     "#E8E8E8",   # light grey
    "concat":     "#CCCCCC",   # mid grey
    "dropout":    "#FFE0B2",   # light peach (dashed)
    "fpn_inner":  "#C0C0C0",   # silver
}
_EC = "#555555"  # edge color for all boxes


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.{{pdf,png}}")


def _box(ax, x, y, w, h, text, color, fontsize=9, bold=True, ls="-", lw=1.0):
    """Draw a rounded box with centered text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06",
        facecolor=color, edgecolor=_EC, linewidth=lw, linestyle=ls))
    fw = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fw, color="#333333")


def _arrow(ax, p1, p2, **kw):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=12,
        color=kw.get("color", "#555555"),
        linewidth=kw.get("lw", 1.2),
        connectionstyle=kw.get("cs", "arc3,rad=0")))


# ── F1: Pipeline overview ─────────────────────────────────────
def make_f1():
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")

    boxes = [
        (0.2, 2.2, 1.8, 1.2, "RGB Image\n(640 × 640)",     _PAL["input_rgb"]),
        (0.2, 0.6, 1.8, 1.2, "Depth Map\n(DepthAnythingV2)", _PAL["input_depth"]),
        (2.6, 1.0, 2.0, 2.0, "RGB-D\nBackbone\n(YOLO-seg)", _PAL["backbone"]),
        (5.2, 2.4, 2.0, 1.0, "Modal Head\n(YOLACT)",        _PAL["head"]),
        (5.2, 0.8, 2.0, 1.0, "Amodal Head\n(YOLACT)",       _PAL["head"]),
        (7.8, 2.4, 2.0, 1.0, "Geometric\nLosses",           _PAL["loss"]),
        (7.8, 0.8, 2.0, 1.0, "Depth Dropout\n(p = 0.5)",    _PAL["dropout"]),
        (10.4, 1.2, 2.4, 1.6, "Predicted\nMasks +\nDiameter", _PAL["output"]),
    ]
    for x, y, w, h, t, c in boxes:
        _box(ax, x, y, w, h, t, c, fontsize=8.5)

    arrows = [
        ((2.0, 2.8), (2.6, 2.5)), ((2.0, 1.2), (2.6, 1.5)),
        ((4.6, 2.5), (5.2, 2.9)), ((4.6, 1.5), (5.2, 1.3)),
        ((7.2, 2.9), (7.8, 2.9)), ((7.2, 1.3), (7.8, 1.3)),
        ((9.8, 2.9), (10.4, 2.3)), ((9.8, 1.3), (10.4, 1.7)),
    ]
    for p1, p2 in arrows:
        _arrow(ax, p1, p2)

    ax.text(6.5, 3.95, "GeoApple-Seg: RGB-D Instance Segmentation with Geometric Supervision",
            ha="center", fontsize=10, fontweight="bold", color="#222222")
    save(fig, "F1_pipeline")


# ── F3: Architecture (detail) ─────────────────────────────────
def make_f3():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    # Inputs
    _box(ax, 0.1, 4.4, 1.5, 0.9, "RGB\n3 × 640²", _PAL["input_rgb"], fontsize=8.5)
    _box(ax, 0.1, 3.1, 1.5, 0.9, "Depth\n1 × 640²", _PAL["input_depth"], fontsize=8.5)

    # Concat
    _box(ax, 2.0, 3.7, 1.2, 0.9, "Concat\n4 × 640²", _PAL["concat"], fontsize=8.5)

    # Backbone block
    ax.add_patch(FancyBboxPatch(
        (3.6, 2.6), 2.4, 3.0, boxstyle="round,pad=0.06",
        facecolor=_PAL["backbone"], edgecolor=_EC, linewidth=1.3))
    ax.text(4.8, 5.3, "YOLO-seg\nBackbone + FPN",
            ha="center", fontsize=9.5, fontweight="bold", color="#333333")
    for lbl, y in [("P3  80²", 4.6), ("P4  40²", 3.9), ("P5  20²", 3.2)]:
        ax.add_patch(plt.Rectangle((3.8, y), 2.0, 0.5,
                     fc=_PAL["fpn_inner"], ec=_EC, lw=0.8))
        ax.text(4.8, y + 0.25, lbl, ha="center", va="center",
                fontsize=8, color="#333333")

    # Proto net
    _box(ax, 6.4, 4.3, 1.5, 0.9, "Proto Net\n32 × 160²", _PAL["head"], fontsize=8.5)

    # Dual coefficient heads
    _box(ax, 6.4, 2.8, 1.5, 0.9, "Modal Coef\n32-d", _PAL["head"], fontsize=8.5)
    _box(ax, 6.4, 1.4, 1.5, 0.9, "Amodal Coef\n32-d", _PAL["head"], fontsize=8.5)

    # Mask assembly
    _box(ax, 8.5, 3.6, 1.5, 0.9, "Modal Mask\nσ(coef · proto)", _PAL["mask"], fontsize=8)
    _box(ax, 8.5, 2.1, 1.5, 0.9, "Amodal Mask\nσ(coef · proto)", _PAL["mask"], fontsize=8)

    # Losses
    _box(ax, 10.3, 3.9, 1.5, 1.2, "L_mask\n+ L_geom\n+ L_size", _PAL["loss"], fontsize=8.5)
    _box(ax, 10.3, 1.6, 1.5, 1.0, "L_amodal", _PAL["loss"], fontsize=8.5)

    # Depth dropout annotation
    _box(ax, 2.0, 0.5, 4.0, 0.7,
         "Depth Dropout (p = 0.5): zero depth channel during training",
         _PAL["dropout"], fontsize=8, bold=False, ls="--", lw=0.8)

    # Arrows
    for p1, p2 in [
        ((1.6, 4.85), (2.0, 4.35)), ((1.6, 3.55), (2.0, 3.95)),
        ((3.2, 4.15), (3.6, 4.1)),
        ((6.0, 4.4), (6.4, 4.75)), ((6.0, 3.5), (6.4, 3.25)),
        ((6.0, 3.0), (6.4, 1.85)),
        ((7.9, 4.75), (8.5, 4.15)), ((7.9, 4.75), (8.5, 2.85)),
        ((7.9, 3.25), (8.5, 4.0)),  ((7.9, 1.85), (8.5, 2.5)),
        ((10.0, 4.05), (10.3, 4.5)), ((10.0, 2.55), (10.3, 2.1)),
    ]:
        _arrow(ax, p1, p2)

    ax.text(6.0, 5.95, "GeoApple-Seg Architecture",
            ha="center", fontsize=11, fontweight="bold", color="#222222")
    save(fig, "F3_architecture")


# ── F2: Dataset overview ──────────────────────────────────────
def make_f2():
    test_dir = ROOT / "data" / "yolo_format" / "images" / "test"
    imgs = sorted(
        [p for p in test_dir.glob("*.jpg")] + [p for p in test_dir.glob("*.png")]
    )
    random.seed(7)
    samples = random.sample(imgs, min(6, len(imgs)))

    fig = plt.figure(figsize=(7.5, 5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.75], hspace=0.3, wspace=0.25)

    for i, p in enumerate(samples):
        img = cv2.imread(str(p))[..., ::-1]
        ax = fig.add_subplot(gs[i // 3, i % 3 * 2:(i % 3 * 2 + 2)])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(p.stem[:20], fontsize=7, color="#555555")

    # Stats panel
    ax_stats = fig.add_subplot(gs[2, :3])
    ax_stats.axis("off")
    stats = (
        "Dataset: AmodalAppleSize-RGBD (Fuji)\n"
        "Total images: 3,890\n"
        "  Train: 2,290  |  Val: 801  |  Test: 799\n"
        "Modality: RGB + depth (DepthAnythingV2)\n"
        "Annotation: modal + amodal polygons,\n"
        "  per-apple diameter (mm)"
    )
    ax_stats.text(0.02, 0.95, stats, va="top", ha="left",
                  fontsize=8, family="monospace", color="#333333")

    # Split bar chart
    ax_bar = fig.add_subplot(gs[2, 3:])
    splits = ["Test", "Val", "Train"]
    counts = [799, 801, 2290]
    bar_colors = ["#0072B2", "#E69F00", "#009E73"]
    bars = ax_bar.barh(splits, counts, color=bar_colors, edgecolor="white",
                       height=0.6)
    for b, c in zip(bars, counts):
        ax_bar.text(b.get_width() + 30, b.get_y() + b.get_height() / 2,
                    f"{c}", va="center", fontsize=8, color="#333333")
    ax_bar.set_xlim(0, 2600)
    ax_bar.set_xlabel("Number of images", fontsize=8)
    ax_bar.set_title("Data split", fontsize=9, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(labelsize=8)

    fig.suptitle("Dataset Overview", fontsize=11, fontweight="bold", y=0.98)
    save(fig, "F2_dataset")


# ── Composites ────────────────────────────────────────────────
def grid_composite(img_paths, rows, cols, title, stem, figsize=None):
    figsize = figsize or (cols * 4.2, rows * 3.0)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for ax, p in zip(axes.flat, img_paths):
        if p and Path(p).exists():
            img = cv2.imread(str(p))[..., ::-1]
            ax.imshow(img)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold", color="#222222")
    plt.tight_layout()
    save(fig, stem)


def make_f7():
    grid_composite(
        [QUAL / "qual_low_00.png", QUAL / "qual_mid_02.png",
         QUAL / "qual_high_04.png"],
        3, 1, "Qualitative Results Across Occlusion Strata (Low / Mid / High)",
        "F7_qualitative_composite", figsize=(7.5, 18),
    )


def make_f8():
    grid_composite(
        [QUAL / "qual_low_01.png", QUAL / "qual_mid_03.png",
         QUAL / "qual_high_05.png"],
        3, 1, "Amodal Prediction: E7 (w/ amodal loss) vs. A3 (ablation)",
        "F8_amodal_ablation", figsize=(7.5, 18),
    )


def make_f9():
    paths = [FAIL / f"failure_{i:02d}.png" for i in range(6)]
    grid_composite(paths, 3, 2, "Failure Cases: Heavy Occlusion and Tiny Fruits",
                   "F9_failure_composite", figsize=(12, 18))


if __name__ == "__main__":
    make_f1()
    make_f3()
    make_f2()
    make_f7()
    make_f8()
    make_f9()
