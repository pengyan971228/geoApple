"""Generate publication-quality architecture diagram for GeoApple-Seg.

Target: EAAI / SCI top-journal level vector figure.
Style: CVPR/TPAMI-inspired, clean 3D feature blocks, muted palette.

Layout: RGB backbone top row, Depth encoder bottom row,
shared fusion/neck/heads in middle — all arrows flow cleanly
without crossing.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

OUT_DIR = Path(__file__).parent.parent / "docs" / "paper_data" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication style ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Professional muted palette ────────────────────────────────
PAL = {
    "rgb":       "#5B9BD5",
    "depth":     "#70AD47",
    "backbone":  "#BDD7EE",
    "depth_enc": "#C5E0B4",
    "fusion":    "#D9D9D9",
    "neck":      "#DBC4C4",
    "proto":     "#F4B183",
    "modal":     "#9DC3E6",
    "amodal":    "#A9D18E",
    "loss":      "#F0908A",
    "size":      "#C9B1D5",
    "dropout":   "#FFD966",
    "output":    "#E8E8E8",
    "det":       "#DCDCDC",
    "arrow":     "#404040",
    "text":      "#222222",
    "dim":       "#777777",
    "group_bg":  "#F7F7F7",
    "group_ec":  "#AAAAAA",
}


# ── Drawing helpers ───────────────────────────────────────────

def _lighten(c, a=0.25):
    rgb = [int(c[i:i+2], 16) for i in (1, 3, 5)]
    return "#{:02x}{:02x}{:02x}".format(
        *[min(255, int(v + (255 - v) * a)) for v in rgb])


def _darken(c, a=0.12):
    rgb = [int(c[i:i+2], 16) for i in (1, 3, 5)]
    return "#{:02x}{:02x}{:02x}".format(
        *[max(0, int(v * (1 - a))) for v in rgb])


def block3d(ax, cx, cy, w, h, d, color, label="", dim="",
            fs=7.5, dfs=5.5):
    """3D cuboid centred at (cx, cy). w, h = front face size, d = depth."""
    x, y = cx - w / 2, cy - h / 2
    dx, dy = d * 0.25, d * 0.25
    # back face
    ax.add_patch(plt.Polygon(
        [(x+dx, y+dy), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+dx, y+h+dy)],
        fc=color, ec="#666", lw=0.5, alpha=0.4, zorder=1))
    # top face
    ax.add_patch(plt.Polygon(
        [(x, y+h), (x+dx, y+h+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
        fc=_lighten(color), ec="#666", lw=0.5, alpha=0.65, zorder=2))
    # right face
    ax.add_patch(plt.Polygon(
        [(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
        fc=_darken(color), ec="#666", lw=0.5, alpha=0.65, zorder=2))
    # front face
    ax.add_patch(plt.Polygon(
        [(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
        fc=color, ec="#555", lw=0.7, alpha=0.85, zorder=3))
    if label:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=PAL["text"], zorder=5)
    if dim:
        ax.text(cx, y - 0.15, dim, ha="center", va="top",
                fontsize=dfs, color=PAL["dim"], zorder=5)


def box(ax, cx, cy, w, h, color, label="", fs=7, bold=True,
        ls="-", lw=0.7, alpha=0.85):
    """Rounded rectangle centred at (cx, cy)."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        fc=color, ec="#555", lw=lw, ls=ls, alpha=alpha, zorder=3))
    if label:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=PAL["text"], zorder=5)


def grp(ax, cx, cy, w, h, label=""):
    """Dashed grouping box centred at (cx, cy)."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        fc=PAL["group_bg"], ec=PAL["group_ec"],
        lw=0.7, ls="--", alpha=0.35, zorder=0))
    if label:
        ax.text(cx, y + h + 0.12, label, ha="center", va="bottom",
                fontsize=6.5, fontstyle="italic", color="#666", zorder=5)


def arr(ax, x1, y1, x2, y2, lw=0.9, color=None, cs="arc3,rad=0"):
    """Arrow from (x1,y1) to (x2,y2)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color or PAL["arrow"],
                                lw=lw, connectionstyle=cs,
                                shrinkA=2, shrinkB=2), zorder=4)


def note(ax, x, y, text, fs=5.5, color=None, style="normal"):
    """Small annotation text."""
    ax.text(x, y, text, ha="center", va="top",
            fontsize=fs, color=color or PAL["dim"],
            fontstyle=style, zorder=5)


# ══════════════════════════════════════════════════════════════
def make_architecture():
    fig, ax = plt.subplots(figsize=(17, 7))
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-1.0, 7.5)
    ax.axis("off")

    # ── Y coordinates ──
    YT = 5.8     # RGB backbone top row
    YM = 3.5     # middle (fusion, neck, shared)
    YB = 1.2     # Depth encoder bottom row
    # Dual head split
    YH_M = 5.0   # modal branch in heads section
    YH_A = 2.8   # amodal branch in heads section

    # ════════════════════════════════════════════════════════════
    # SECTION 1: INPUTS  (x ~ 0.6)
    # ════════════════════════════════════════════════════════════
    block3d(ax, 0.6, YT, 0.6, 1.3, 0.5, PAL["rgb"],
            "RGB", "3 x 640 x 640", fs=8, dfs=5.5)
    block3d(ax, 0.6, YB, 0.6, 1.3, 0.4, PAL["depth"],
            "Depth", "1 x 640 x 640", fs=8, dfs=5.5)
    # Depth dropout
    box(ax, 0.6, YB - 1.15, 1.3, 0.4, PAL["dropout"],
        "Depth Dropout (p = 0.5)", fs=5.5, bold=False, ls="--", lw=0.6)

    # ════════════════════════════════════════════════════════════
    # SECTION 2: DUAL BACKBONE  (x ~ 2.0 - 5.0)
    # ════════════════════════════════════════════════════════════
    # RGB Backbone group
    grp(ax, 3.5, YT, 4.2, 2.0, "RGB Backbone (YOLO-seg)")

    # Backbone blocks: C3, C4, C5
    bk_x = [2.2, 3.5, 4.8]
    bk_w = [0.85, 0.75, 0.65]
    bk_h = [1.0, 0.9, 0.8]
    bk_d = [0.45, 0.38, 0.30]
    bk_lbl = ["C3", "C4", "C5"]
    bk_dim = ["256 x 80²", "256 x 40²", "512 x 20²"]
    for i in range(3):
        block3d(ax, bk_x[i], YT, bk_w[i], bk_h[i], bk_d[i],
                PAL["backbone"], bk_lbl[i], bk_dim[i], fs=7.5, dfs=5)

    # Depth Encoder group
    grp(ax, 3.5, YB, 4.2, 2.0, "Depth Encoder")

    # Depth encoder blocks: D3, D4, D5
    de_lbl = ["D3", "D4", "D5"]
    de_dim = ["256 x 80²", "256 x 40²", "512 x 20²"]
    for i in range(3):
        block3d(ax, bk_x[i], YB, bk_w[i], bk_h[i], bk_d[i],
                PAL["depth_enc"], de_lbl[i], de_dim[i], fs=7.5, dfs=5)

    # Backbone internal arrows (C3→C4→C5, D3→D4→D5)
    for i in range(2):
        rx = bk_x[i] + bk_w[i] / 2 + bk_d[i] * 0.25
        lx = bk_x[i+1] - bk_w[i+1] / 2
        arr(ax, rx, YT, lx, YT, lw=0.6)
        arr(ax, rx, YB, lx, YB, lw=0.6)

    # Input → Backbone/Encoder
    arr(ax, 0.95, YT, bk_x[0] - bk_w[0] / 2, YT)
    arr(ax, 0.95, YB, bk_x[0] - bk_w[0] / 2, YB)

    # ════════════════════════════════════════════════════════════
    # SECTION 3: MULTI-SCALE FUSION  (x ~ 6.2)
    # ════════════════════════════════════════════════════════════
    grp(ax, 6.2, YM, 1.2, 5.2, "Fusion")

    fus_x = 6.2
    fus_y = [YT - 0.3, YM, YB + 0.3]   # 3 fusion boxes spread vertically
    fus_lbl = ["Cat\n+1x1", "Cat\n+1x1", "Cat\n+1x1"]
    fus_dim = ["F3: 256x80²", "F4: 256x40²", "F5: 512x20²"]
    for i in range(3):
        box(ax, fus_x, fus_y[i], 0.85, 0.65, PAL["fusion"],
            fus_lbl[i], fs=6)
        note(ax, fus_x, fus_y[i] - 0.42, fus_dim[i], fs=5)

    # Backbone → Fusion arrows (C_i → F_i from top, D_i → F_i from bottom)
    for i in range(3):
        rx = bk_x[i] + bk_w[i] / 2 + bk_d[i] * 0.25
        # C_i → F_i
        arr(ax, rx, YT - bk_h[i] * 0.1, fus_x - 0.42, fus_y[i] + 0.15,
            lw=0.7)
        # D_i → F_i
        arr(ax, rx, YB + bk_h[i] * 0.1, fus_x - 0.42, fus_y[i] - 0.15,
            lw=0.7)

    # ════════════════════════════════════════════════════════════
    # SECTION 4: FPN + PAN NECK  (x ~ 7.8)
    # ════════════════════════════════════════════════════════════
    grp(ax, 7.8, YM, 1.2, 5.2, "FPN + PAN Neck")

    nk_x = 7.8
    nk_y = fus_y  # same Y positions as fusion
    nk_w = [0.75, 0.65, 0.55]
    nk_h = [0.7, 0.65, 0.55]
    nk_d = [0.3, 0.25, 0.2]
    nk_lbl = ["P3", "P4", "P5"]
    nk_dim = ["128 x 80²", "256 x 40²", "512 x 20²"]
    for i in range(3):
        block3d(ax, nk_x, nk_y[i], nk_w[i], nk_h[i], nk_d[i],
                PAL["neck"], nk_lbl[i], nk_dim[i], fs=7, dfs=5)

    # Fusion → Neck arrows
    for i in range(3):
        arr(ax, fus_x + 0.42, fus_y[i], nk_x - nk_w[i] / 2 - 0.05,
            nk_y[i])

    # FPN bidirectional vertical arrows (top-down + bottom-up)
    for i in range(2):
        arr(ax, nk_x + 0.08, nk_y[i] - nk_h[i] / 2,
            nk_x + 0.08, nk_y[i+1] + nk_h[i+1] / 2, lw=0.5, color="#999")
        arr(ax, nk_x - 0.08, nk_y[i+1] + nk_h[i+1] / 2,
            nk_x - 0.08, nk_y[i] - nk_h[i] / 2, lw=0.5, color="#999")

    # ════════════════════════════════════════════════════════════
    # SECTION 5: DETECTION HEAD  (x ~ 9.2, bottom)
    # ════════════════════════════════════════════════════════════
    box(ax, 9.3, YB - 0.1, 1.1, 0.55, PAL["det"],
        "Detection Head", fs=6)
    note(ax, 9.3, YB - 0.48, "bbox + cls", fs=5)
    # P4, P5 → Det Head
    arr(ax, nk_x + nk_w[1] / 2 + nk_d[1] * 0.25, nk_y[1] - 0.1,
        9.3 - 0.55, YB + 0.1, lw=0.6)
    arr(ax, nk_x + nk_w[2] / 2 + nk_d[2] * 0.25, nk_y[2],
        9.3 - 0.55, YB - 0.1, lw=0.6)

    # ════════════════════════════════════════════════════════════
    # SECTION 6: DUAL MASK HEAD  (x ~ 9.5 - 13.5)
    # ════════════════════════════════════════════════════════════
    grp(ax, 11.3, YM + 0.2, 5.2, 4.0, "Dual Mask Head")

    # --- Proto Net ---
    block3d(ax, 9.5, YH_M, 1.0, 0.9, 0.5, PAL["proto"],
            "Proto\nNet", "32 x 160²", fs=7, dfs=5)

    # P3 → Proto
    arr(ax, nk_x + nk_w[0] / 2 + nk_d[0] * 0.25, nk_y[0],
        9.5 - 0.5 - 0.05, YH_M + 0.1)

    # --- Modal Coefficient Head ---
    box(ax, 11.0, YH_M, 1.0, 0.6, PAL["modal"],
        "Modal\nCoef", fs=6.5)
    note(ax, 11.0, YH_M - 0.4, "32 x N", fs=5)

    # --- Amodal Coefficient Head ---
    box(ax, 11.0, YH_A, 1.0, 0.6, PAL["amodal"],
        "Amodal\nCoef", fs=6.5)
    note(ax, 11.0, YH_A - 0.4, "32 x N", fs=5)
    # +depth ordering annotation
    ax.text(11.6, YH_A - 0.05, "+depth\nordering", ha="left", va="center",
            fontsize=5, color=PAL["depth"], fontstyle="italic", zorder=5)

    # P3 → Modal Coef, Neck → Amodal Coef
    arr(ax, nk_x + nk_w[0] / 2 + nk_d[0] * 0.25, nk_y[0] - 0.15,
        11.0 - 0.5, YH_M)
    arr(ax, nk_x + nk_w[1] / 2 + nk_d[1] * 0.25, nk_y[1] + 0.1,
        11.0 - 0.5, YH_A + 0.1)

    # --- Mask Assembly ---
    # Modal Mask: sigma(Proto * Coef)
    box(ax, 12.5, YH_M, 1.0, 0.6, PAL["modal"],
        "Modal\nMask", fs=6.5)
    note(ax, 12.5, YH_M - 0.4, "sigma(P * c)", fs=5)

    # Amodal Mask
    box(ax, 12.5, YH_A, 1.0, 0.6, PAL["amodal"],
        "Amodal\nMask", fs=6.5)
    note(ax, 12.5, YH_A - 0.4, "sigma(P * c)", fs=5)

    # Proto → both masks
    arr(ax, 10.0 + 0.5 * 0.25, YH_M, 12.0, YH_M)
    arr(ax, 10.0 + 0.5 * 0.25, YH_M - 0.3,
        12.0, YH_A + 0.15, lw=0.7)

    # Coef → masks
    arr(ax, 11.5, YH_M, 12.0, YH_M)
    arr(ax, 11.5, YH_A, 12.0, YH_A)

    # ════════════════════════════════════════════════════════════
    # SECTION 7: LOSSES + SIZE HEAD  (x ~ 14.2 - 15.5)
    # ════════════════════════════════════════════════════════════
    # Modal losses
    box(ax, 14.3, YH_M, 1.5, 0.65, PAL["loss"],
        "Lmask + Lgeom", fs=6.5)
    ax.text(14.3, YH_M - 0.42, "Dice+BCE, MCL,\nBoundary Align",
            ha="center", va="top", fontsize=4.5, color="#CC4444",
            fontstyle="italic", zorder=5)

    # Amodal loss
    box(ax, 14.3, YH_A, 1.5, 0.65, PAL["loss"],
        "Lamodal", fs=6.5)
    note(ax, 14.3, YH_A - 0.42, "Dice + BCE", fs=4.5, color="#CC4444",
         style="italic")

    # Size Head
    box(ax, 14.3, YB + 0.5, 1.5, 0.65, PAL["size"],
        "Size Head (MLP)", fs=6.5)
    note(ax, 14.3, YB + 0.5 - 0.42, "3 -> 64 -> 64 -> 1\ndiameter (mm)",
         fs=4.5)

    # Masks → Losses
    arr(ax, 13.0, YH_M, 13.55, YH_M)
    arr(ax, 13.0, YH_A, 13.55, YH_A)

    # Modal Mask → Size Head
    arr(ax, 13.0, YH_M - 0.25, 13.55, YB + 0.7)

    # ════════════════════════════════════════════════════════════
    # SECTION 8: OUTPUT  (x ~ 16.5)
    # ════════════════════════════════════════════════════════════
    box(ax, 16.5, YM + 0.5, 1.6, 2.4, PAL["output"],
        "Predicted\nModal Mask\nAmodal Mask\nDiameter (mm)", fs=7, lw=1.0)

    # → Output arrows
    arr(ax, 15.05, YH_M, 15.7, YM + 1.2, lw=0.7)
    arr(ax, 15.05, YH_A, 15.7, YM + 0.6, lw=0.7)
    arr(ax, 15.05, YB + 0.5, 15.7, YM + 0.0, lw=0.7)

    # ════════════════════════════════════════════════════════════
    # TITLE
    # ════════════════════════════════════════════════════════════
    ax.text(9.0, 7.2, "GeoApple-Seg Architecture",
            ha="center", fontsize=14, fontweight="bold", color=PAL["text"])

    # ════════════════════════════════════════════════════════════
    # LEGEND (2 rows x 6 cols)
    # ════════════════════════════════════════════════════════════
    legend_items = [
        (PAL["rgb"],       "RGB input"),
        (PAL["depth"],     "Depth input"),
        (PAL["backbone"],  "RGB backbone"),
        (PAL["depth_enc"], "Depth encoder"),
        (PAL["fusion"],    "Fusion (Cat + 1x1)"),
        (PAL["neck"],      "FPN / PAN neck"),
        (PAL["proto"],     "Prototypes"),
        (PAL["modal"],     "Modal branch"),
        (PAL["amodal"],    "Amodal branch"),
        (PAL["loss"],      "Losses"),
        (PAL["size"],      "Size head"),
        (PAL["dropout"],   "Depth dropout (train)"),
    ]
    n_cols = 6
    lx0, ly0 = 1.0, -0.25
    col_w = 2.85
    for i, (color, label) in enumerate(legend_items):
        xi = lx0 + (i % n_cols) * col_w
        yi = ly0 - (i // n_cols) * 0.4
        ax.add_patch(Rectangle((xi, yi - 0.1), 0.3, 0.22,
                     fc=color, ec="#555", lw=0.5, zorder=5))
        ax.text(xi + 0.4, yi + 0.01, label, va="center",
                fontsize=6, color="#444", zorder=5)

    # ── Save ──
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"F3_architecture.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved F3_architecture.{pdf,png}")


if __name__ == "__main__":
    make_architecture()
