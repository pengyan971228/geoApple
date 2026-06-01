#!/usr/bin/env python3
"""Replot F10_pr_curve from cached JSON with tightened axis ranges.

The original eval_pr_curve.py used overly wide axis limits, leaving most
of the canvas empty.  This script reads the cached precision-recall data
and redraws the figure with axis limits derived from the actual data range,
so the figure fills the available canvas.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = PROJECT_ROOT / "docs" / "paper_data" / "pr_curve_data.json"
OUT_DIR = PROJECT_ROOT / "docs" / "paper_data" / "figures"

# Okabe-Ito colour-blind safe palette
COLOURS = {
    "GeoApple-Seg (Ours)": "#D55E00",   # vermillion
    "YOLOv8s-seg":         "#0072B2",   # blue
    "YOLOv11s-seg":        "#009E73",   # bluish green
}
MARKERS = {
    "GeoApple-Seg (Ours)": "o",
    "YOLOv8s-seg":         "s",
    "YOLOv11s-seg":        "^",
}
DEFAULT_THR = {
    "GeoApple-Seg (Ours)": 0.50,
    "YOLOv8s-seg":         0.25,
    "YOLOv11s-seg":        0.25,
}


def autoscale(values, pad_lo=0.04, pad_hi=0.04):
    """Return (lo, hi) limits with given fractional padding."""
    vmin, vmax = min(values), max(values)
    span = max(vmax - vmin, 1e-3)
    lo = vmin - pad_lo * span
    hi = vmax + pad_hi * span
    return lo, hi


def replot() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Cached PR data not found: {JSON_PATH}")
    data = json.loads(JSON_PATH.read_text())

    # Collect all recall/precision/iou values across models for axis scaling.
    all_recall, all_prec, all_iou = [], [], []
    for pts in data.values():
        all_recall.extend(p["recall"] for p in pts)
        all_prec.extend(p["precision"] for p in pts)
        all_iou.extend(p["iou"] for p in pts)

    rec_lim   = autoscale(all_recall, pad_lo=0.06, pad_hi=0.04)
    prec_lim  = autoscale(all_prec,   pad_lo=0.05, pad_hi=0.05)
    iou_lim   = autoscale(all_iou,    pad_lo=0.05, pad_hi=0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- (a) Precision-Recall curve ---
    for name, pts in data.items():
        rec  = [p["recall"]    for p in pts]
        prec = [p["precision"] for p in pts]
        ax1.plot(rec, prec,
                 color=COLOURS.get(name, "gray"),
                 marker=MARKERS.get(name, "."),
                 markersize=6, linewidth=2.0, label=name)
        # Mark the default operating point
        thr = DEFAULT_THR.get(name, 0.25)
        closest = min(pts, key=lambda p: abs(p["threshold"] - thr))
        ax1.plot(closest["recall"], closest["precision"],
                 marker="*", markersize=18,
                 color=COLOURS.get(name, "gray"),
                 markeredgecolor="black", markeredgewidth=0.9, zorder=5)

    ax1.set_xlabel("Recall",    fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_title("(a) Precision–recall curve", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=10, framealpha=0.92)
    ax1.set_xlim(*rec_lim)
    ax1.set_ylim(*prec_lim)
    ax1.xaxis.set_major_locator(MultipleLocator(0.05))
    ax1.yaxis.set_major_locator(MultipleLocator(0.02))
    ax1.grid(True, linestyle=":", alpha=0.5)

    # --- (b) IoU at matched recall ---
    for name, pts in data.items():
        rec = [p["recall"] for p in pts]
        iou = [p["iou"]    for p in pts]
        ax2.plot(rec, iou,
                 color=COLOURS.get(name, "gray"),
                 marker=MARKERS.get(name, "."),
                 markersize=6, linewidth=2.0, label=name)
        thr = DEFAULT_THR.get(name, 0.25)
        closest = min(pts, key=lambda p: abs(p["threshold"] - thr))
        ax2.plot(closest["recall"], closest["iou"],
                 marker="*", markersize=18,
                 color=COLOURS.get(name, "gray"),
                 markeredgecolor="black", markeredgewidth=0.9, zorder=5)

    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("IoU",    fontsize=12)
    ax2.set_title("(b) IoU at matched recall", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.92)
    ax2.set_xlim(*rec_lim)
    ax2.set_ylim(*iou_lim)
    ax2.xaxis.set_major_locator(MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(MultipleLocator(0.02))
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Operating-point analysis: pixel-level metrics vs. threshold",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"F10_pr_curve.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()

    print(f"\nAxis ranges used:")
    print(f"  Recall:    [{rec_lim[0]:.3f}, {rec_lim[1]:.3f}]")
    print(f"  Precision: [{prec_lim[0]:.3f}, {prec_lim[1]:.3f}]")
    print(f"  IoU:       [{iou_lim[0]:.3f}, {iou_lim[1]:.3f}]")


if __name__ == "__main__":
    replot()
