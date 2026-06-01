"""P-R curve comparison: GeoApple-Seg vs YOLOv8s/v11s baselines.

Sweeps the operating-point threshold for each model and computes
pixel-level Precision, Recall, IoU at each point.

- GeoApple-Seg: sweep sigmoid binarization threshold (0.1 → 0.9)
- YOLO baselines: sweep detection conf threshold (0.05 → 0.90)

Outputs:
    docs/paper_data/figures/F10_pr_curve.{pdf,png}
    docs/paper_data/pr_curve_data.json

Usage:
    python scripts/eval_pr_curve.py
    python scripts/eval_pr_curve.py --device mps --models ours v11
    python scripts/eval_pr_curve.py --device cuda --batch 32
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import compute_pixel_metrics, predict_masks_avg
from src.data_module.rgbd_dataset import build_rgbd_dataloader
from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel

DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"
OUT_DIR = PROJECT_ROOT / "docs" / "paper_data" / "figures"
OUT_JSON = PROJECT_ROOT / "docs" / "paper_data" / "pr_curve_data.json"

# Thresholds to sweep
OURS_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                   0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                   0.80, 0.85, 0.90]
YOLO_CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                        0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                        0.75, 0.80, 0.85, 0.90]

YOLO_MODELS = {
    "v8":  ("YOLOv8s-seg",
            PROJECT_ROOT / "runs" / "baseline_yolov8s" / "weights" / "best.pt"),
    "v11": ("YOLOv11s-seg",
            PROJECT_ROOT / "runs" / "baseline_yolov11s" / "weights" / "best.pt"),
}

plt.rcParams.update({
    "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# YOLO baseline helpers (copied from run_yolo_baselines.py)
# ---------------------------------------------------------------------------

def _read_gt_mask(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
    return mask


def _aggregate_pred_masks(result, img_h: int, img_w: int) -> np.ndarray:
    out = np.zeros((img_h, img_w), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return out
    masks = result.masks.data.cpu().numpy()
    for m in masks:
        if m.shape[:2] != (img_h, img_w):
            m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        out |= (m > 0.5).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# GeoApple-Seg evaluation at variable threshold
# ---------------------------------------------------------------------------

def load_geoapple(device: str) -> GeoAppleSegModel:
    ckpt_path = PROJECT_ROOT / "runs" / "E7_v7" / "weights" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", GeoAppleConfig())
    try:
        cfg.yolo_weights = str(PROJECT_ROOT / "yolo26s-seg.pt")
    except Exception:
        cfg = replace(cfg, yolo_weights=str(PROJECT_ROOT / "yolo26s-seg.pt"))
    model = GeoAppleSegModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def extract_gt_masks(batch: dict, device: str) -> np.ndarray:
    """Extract GT modal masks from batch, matching evaluate.py."""
    from scripts.evaluate import extract_gt_masks as _extract
    return _extract(batch, device)


def eval_ours_at_thresholds(
    thresholds: List[float], device: str, batch_size: int,
) -> List[Dict[str, float]]:
    """Evaluate GeoApple-Seg at multiple sigmoid thresholds."""
    print(f"\n{'='*60}\n[P-R] GeoApple-Seg (E7_v7) — {len(thresholds)} thresholds\n{'='*60}")
    model = load_geoapple(device)
    loader = build_rgbd_dataloader(
        dataset_yaml=str(DATASET_YAML), split="test",
        batch_size=batch_size, img_size=640, workers=2, shuffle=False,
    )

    # Collect raw sigmoid probs and GT for all images
    all_probs: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []

    for batch in loader:
        rgb_in = batch["img"].to(device).float() / 255.0
        depth_in = batch["depth"].to(device)

        # Get raw probs instead of binary
        with torch.no_grad():
            outputs = model(rgb_in, depth_in)
            proto = outputs["proto"]
            coeffs = outputs["modal_coeffs"]
            B, nm, H, W = proto.shape
            avg = coeffs.mean(dim=2, keepdim=True)
            masks = torch.einsum("bmhw,bmk->bkhw", proto, avg).squeeze(1)
            probs = torch.sigmoid(masks).cpu().numpy()  # (B, H, W)

        gt_masks = extract_gt_masks(batch, device)
        if gt_masks.shape[-2:] != (H, W):
            gt_resized = np.zeros((gt_masks.shape[0], H, W), dtype=np.uint8)
            for i in range(gt_masks.shape[0]):
                gt_resized[i] = cv2.resize(
                    gt_masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
            gt_masks = gt_resized

        for i in range(B):
            if i < probs.shape[0]:
                all_probs.append(probs[i])
                all_gt.append(gt_masks[i])

    print(f"  Collected {len(all_probs)} images, sweeping thresholds...")
    results = []
    for thr in thresholds:
        metrics_list = []
        for prob, gt in zip(all_probs, all_gt):
            pred = (prob > thr).astype(np.uint8)
            metrics_list.append(compute_pixel_metrics(pred, gt))
        agg = {
            "threshold": thr,
            "iou": float(np.mean([m["iou"] for m in metrics_list])),
            "precision": float(np.mean([m["precision"] for m in metrics_list])),
            "recall": float(np.mean([m["recall"] for m in metrics_list])),
            "n": len(metrics_list),
        }
        print(f"    thr={thr:.2f}  IoU={agg['iou']:.4f}  "
              f"P={agg['precision']:.4f}  R={agg['recall']:.4f}")
        results.append(agg)
    return results


# ---------------------------------------------------------------------------
# YOLO baseline evaluation at variable conf
# ---------------------------------------------------------------------------

def eval_yolo_at_confs(
    tag: str, label: str, weights: Path,
    confs: List[float], img_size: int, batch_size: int, device: str,
) -> List[Dict[str, float]]:
    """Evaluate a YOLO model at multiple confidence thresholds."""
    print(f"\n{'='*60}\n[P-R] {label} — {len(confs)} thresholds\n{'='*60}")
    if not weights.exists():
        print(f"[WARN] missing {weights}, skipping")
        return []
    model = YOLO(str(weights))

    cfg = yaml.safe_load(open(DATASET_YAML))
    root = Path(cfg["path"])
    img_dir = root / cfg["test"]
    label_dir = img_dir.parent.parent / "labels" / img_dir.name
    img_paths = sorted(
        [p for p in img_dir.glob("*.jpg")] + [p for p in img_dir.glob("*.png")]
    )
    print(f"  {len(img_paths)} test images")

    # Run prediction once at lowest conf to get all possible detections,
    # then filter by conf. But YOLO conf filtering happens inside predict(),
    # so we must run predict at each conf level. To speed up, cache images.
    results_all = []
    for conf in confs:
        metrics_list: List[Dict[str, float]] = []
        for start in range(0, len(img_paths), batch_size):
            chunk = img_paths[start:start + batch_size]
            preds = model.predict(
                [str(p) for p in chunk],
                imgsz=img_size, device=device, verbose=False, conf=conf,
            )
            for p, res in zip(chunk, preds):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                H, W = img.shape[:2]
                gt = _read_gt_mask(label_dir / (p.stem + ".txt"), H, W)
                pred = _aggregate_pred_masks(res, H, W)
                metrics_list.append(compute_pixel_metrics(pred, gt))

        agg = {
            "threshold": conf,
            "iou": float(np.mean([m["iou"] for m in metrics_list])),
            "precision": float(np.mean([m["precision"] for m in metrics_list])),
            "recall": float(np.mean([m["recall"] for m in metrics_list])),
            "n": len(metrics_list),
        }
        print(f"    conf={conf:.2f}  IoU={agg['iou']:.4f}  "
              f"P={agg['precision']:.4f}  R={agg['recall']:.4f}")
        results_all.append(agg)
    return results_all


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pr_curves(all_data: Dict[str, List[Dict[str, float]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "GeoApple-Seg (Ours)": "#d62728",
        "YOLOv8s-seg": "#1f77b4",
        "YOLOv11s-seg": "#2ca02c",
    }
    markers = {
        "GeoApple-Seg (Ours)": "o",
        "YOLOv8s-seg": "s",
        "YOLOv11s-seg": "^",
    }

    # --- Figure A: P-R Curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for name, pts in all_data.items():
        rec = [p["recall"] for p in pts]
        prec = [p["precision"] for p in pts]
        ax1.plot(rec, prec, color=colors.get(name, "gray"),
                 marker=markers.get(name, "."), markersize=5,
                 linewidth=2, label=name)
        # Mark the default operating point (threshold closest to 0.5/0.25)
        default_thr = 0.50 if "Ours" in name else 0.25
        closest = min(pts, key=lambda p: abs(p["threshold"] - default_thr))
        ax1.plot(closest["recall"], closest["precision"], marker="*",
                 markersize=14, color=colors.get(name, "gray"),
                 markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_title("(a) Precision–Recall Curve", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # --- Figure B: IoU-vs-Recall Curve ---
    for name, pts in all_data.items():
        rec = [p["recall"] for p in pts]
        iou = [p["iou"] for p in pts]
        ax2.plot(rec, iou, color=colors.get(name, "gray"),
                 marker=markers.get(name, "."), markersize=5,
                 linewidth=2, label=name)
        default_thr = 0.50 if "Ours" in name else 0.25
        closest = min(pts, key=lambda p: abs(p["threshold"] - default_thr))
        ax2.plot(closest["recall"], closest["iou"], marker="*",
                 markersize=14, color=colors.get(name, "gray"),
                 markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("IoU", fontsize=12)
    ax2.set_title("(b) IoU at Matched Recall", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.3, 0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Operating-Point Analysis: Pixel-Level Metrics vs. Threshold",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"F10_pr_curve.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved F10_pr_curve.{{pdf,png}} to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ours", "v8", "v11"],
                    choices=["ours", "v8", "v11"])
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    all_data: Dict[str, List[Dict[str, float]]] = {}

    if "ours" in args.models:
        all_data["GeoApple-Seg (Ours)"] = eval_ours_at_thresholds(
            OURS_THRESHOLDS, args.device, args.batch)

    for tag in [t for t in args.models if t != "ours"]:
        label, weights = YOLO_MODELS[tag]
        pts = eval_yolo_at_confs(
            tag, label, weights,
            YOLO_CONF_THRESHOLDS, args.img_size, args.batch, args.device)
        if pts:
            all_data[label] = pts

    # Save raw data
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_data, indent=2))
    print(f"Saved raw data to {OUT_JSON}")

    # Plot
    if all_data:
        plot_pr_curves(all_data)

    # Print key finding: at matched recall, compare IoU
    if "GeoApple-Seg (Ours)" in all_data and len(all_data) > 1:
        print("\n" + "="*60)
        print("KEY FINDING: IoU at matched recall levels")
        print("="*60)
        ours = all_data["GeoApple-Seg (Ours)"]
        for name, pts in all_data.items():
            if "Ours" in name:
                continue
            # Find recall levels where both have data
            for target_rec in [0.80, 0.82, 0.85, 0.87, 0.90]:
                ours_pt = min(ours, key=lambda p: abs(p["recall"] - target_rec))
                other_pt = min(pts, key=lambda p: abs(p["recall"] - target_rec))
                if abs(ours_pt["recall"] - target_rec) < 0.05 and \
                   abs(other_pt["recall"] - target_rec) < 0.05:
                    diff = ours_pt["iou"] - other_pt["iou"]
                    sign = "+" if diff > 0 else ""
                    print(f"  Recall≈{target_rec:.2f}: "
                          f"Ours IoU={ours_pt['iou']:.4f} vs "
                          f"{name} IoU={other_pt['iou']:.4f}  "
                          f"({sign}{diff:.4f})")


if __name__ == "__main__":
    main()
