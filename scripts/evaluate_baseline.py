"""Evaluate E1 baseline (YOLO26s-seg RGB-only) with same pixel-level metrics as E7.

Computes IoU, Dice, Precision, Recall, F1 on the test set using YOLO's
native inference, then aggregates per-image binary masks for fair comparison
with GeoApple-Seg E7 results.

Usage:
    python scripts/evaluate_baseline.py --weights runs/E1_baseline_rgb/weights/best.pt --device cuda
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"


# ---------------------------------------------------------------------------
# Metrics (same as evaluate.py for consistency)
# ---------------------------------------------------------------------------

def compute_pixel_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    """Compute pixel-level segmentation metrics."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    intersection = tp
    union = tp + fp + fn
    iou = intersection / max(union, 1)
    dice = 2 * intersection / max(2 * intersection + fp + fn, 1)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate per-image metrics to mean and std."""
    keys = all_metrics[0].keys()
    result = {}
    for k in keys:
        values = [m[k] for m in all_metrics]
        result[f"{k}_mean"] = float(np.mean(values))
        result[f"{k}_std"] = float(np.std(values))
    return result


# ---------------------------------------------------------------------------
# GT mask extraction (reuse YOLO dataloader)
# ---------------------------------------------------------------------------

def load_gt_masks(dataset_yaml: str, split: str = "test") -> Dict[str, np.ndarray]:
    """Load ground truth masks from YOLO dataset.

    Returns:
        Dict mapping image stem -> binary mask (H, W).
    """
    import yaml
    from ultralytics.data.utils import check_det_dataset

    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    base_path = Path(cfg["path"])
    label_dir = base_path / "labels" / split
    img_dir = base_path / cfg[split]

    gt_masks = {}
    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem
        # Find corresponding image to get dimensions
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Parse YOLO-seg label: class x1 y1 x2 y2 ... (normalized polygon)
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # Skip class id, parse polygon coords
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 != 0:
                    continue
                points = np.array(coords).reshape(-1, 2)
                points[:, 0] *= w
                points[:, 1] *= h
                points = points.astype(np.int32)
                cv2.fillPoly(mask, [points], 1)

        gt_masks[stem] = mask

    logger.info("Loaded %d GT masks from %s", len(gt_masks), label_dir)
    return gt_masks


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_comparison(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Path,
    metrics: Dict[str, float],
) -> None:
    """Save side-by-side comparison image."""
    h, w = rgb.shape[:2]

    gt_resized = cv2.resize(
        gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
    )
    pred_resized = cv2.resize(
        pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
    )

    gt_overlay = rgb.copy()
    gt_overlay[gt_resized > 0] = (
        gt_overlay[gt_resized > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)

    pred_overlay = rgb.copy()
    pred_overlay[pred_resized > 0] = (
        pred_overlay[pred_resized > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)

    canvas = np.concatenate([rgb, gt_overlay, pred_overlay], axis=1)

    text = (
        f"IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f} "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}"
    )
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "GT (green)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "Pred (red)", (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(str(save_path), canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E1 Baseline (YOLO26s-seg)")
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to YOLO best.pt weights",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--num-vis", type=int, default=20, help="Number of visualization samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("E1 Baseline Evaluation (YOLO26s-seg RGB-only)")
    logger.info("=" * 60)

    # --- Load YOLO model ---
    from ultralytics import YOLO

    logger.info("Loading YOLO model: %s", args.weights)
    model = YOLO(args.weights)

    # --- Load GT masks ---
    logger.info("Loading ground truth masks...")
    gt_masks = load_gt_masks(str(DATASET_YAML), split="test")

    # --- Get test image paths ---
    import yaml

    with open(DATASET_YAML) as f:
        cfg = yaml.safe_load(f)

    base_path = Path(cfg["path"])
    img_dir = base_path / cfg["test"]
    img_paths = sorted(img_dir.glob("*.*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    logger.info("Test images: %d", len(img_paths))

    # --- Run inference and evaluate ---
    logger.info("Running inference...")
    all_metrics = []
    vis_count = 0

    output_dir = Path(args.weights).parent.parent
    vis_dir = output_dir / "visualizations_pixel"
    vis_dir.mkdir(exist_ok=True)

    times = []
    for img_path in img_paths:
        stem = img_path.stem

        # Skip if no GT
        if stem not in gt_masks:
            continue

        gt_mask = gt_masks[stem]
        h, w = gt_mask.shape[:2]

        # YOLO inference
        start = time.perf_counter()
        results = model.predict(
            str(img_path),
            conf=args.conf,
            iou=args.iou_thresh,
            device=args.device,
            verbose=False,
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        result = results[0]

        # Extract predicted masks -> aggregate to per-image binary mask
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if result.masks is not None and len(result.masks) > 0:
            for mask_data in result.masks.data:
                # mask_data is (mask_h, mask_w) tensor
                m = mask_data.cpu().numpy()
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask = np.maximum(pred_mask, (m_resized > 0.5).astype(np.uint8))

        # Compute metrics
        m = compute_pixel_metrics(pred_mask, gt_mask)
        all_metrics.append(m)

        # Save visualizations
        if vis_count < args.num_vis:
            rgb = cv2.imread(str(img_path))
            if rgb is not None:
                save_comparison(
                    rgb, gt_mask, pred_mask,
                    vis_dir / f"test_{vis_count:04d}.jpg", m,
                )
                vis_count += 1

        if len(all_metrics) % 100 == 0:
            logger.info("  Processed %d/%d images", len(all_metrics), len(img_paths))

    # --- Aggregate results ---
    agg = aggregate_metrics(all_metrics)

    avg_ms = float(np.mean(times))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS (E1 Baseline YOLO26s-seg)")
    logger.info("=" * 60)
    logger.info("  Images:     %d", len(all_metrics))
    logger.info("  IoU:        %.4f +/- %.4f", agg["iou_mean"], agg["iou_std"])
    logger.info("  Dice:       %.4f +/- %.4f", agg["dice_mean"], agg["dice_std"])
    logger.info("  Precision:  %.4f +/- %.4f", agg["precision_mean"], agg["precision_std"])
    logger.info("  Recall:     %.4f +/- %.4f", agg["recall_mean"], agg["recall_std"])
    logger.info("  F1:         %.4f +/- %.4f", agg["f1_mean"], agg["f1_std"])
    logger.info("  Speed:      %.1f ms/img (%.1f FPS)", avg_ms, fps)

    # --- Also run YOLO's native evaluation for mAP ---
    logger.info("Running YOLO native validation for mAP...")
    val_results = model.val(
        data=str(DATASET_YAML),
        split="test",
        device=args.device,
        verbose=False,
    )
    logger.info("  mAP@0.5:      %.4f", val_results.seg.map50)
    logger.info("  mAP@0.5:0.95: %.4f", val_results.seg.map)

    # --- Save results ---
    results_path = output_dir / "eval_results_pixel.txt"
    with open(results_path, "w") as f:
        f.write("E1 Baseline Evaluation Results (YOLO26s-seg RGB-only)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Images: {len(all_metrics)}\n\n")
        f.write("Pixel-Level Segmentation Metrics:\n")
        f.write(f"  IoU:       {agg['iou_mean']:.4f} +/- {agg['iou_std']:.4f}\n")
        f.write(f"  Dice:      {agg['dice_mean']:.4f} +/- {agg['dice_std']:.4f}\n")
        f.write(f"  Precision: {agg['precision_mean']:.4f} +/- {agg['precision_std']:.4f}\n")
        f.write(f"  Recall:    {agg['recall_mean']:.4f} +/- {agg['recall_std']:.4f}\n")
        f.write(f"  F1:        {agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f}\n\n")
        f.write("YOLO Native Metrics:\n")
        f.write(f"  mAP@0.5:      {val_results.seg.map50:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {val_results.seg.map:.4f}\n\n")
        f.write("Inference Speed:\n")
        f.write(f"  {avg_ms:.1f} ms/img ({fps:.1f} FPS)\n")
        f.write(f"Visualizations: {vis_dir}\n")

    logger.info("Results saved to: %s", results_path)
    logger.info("Visualizations saved to: %s", vis_dir)


if __name__ == "__main__":
    main()
