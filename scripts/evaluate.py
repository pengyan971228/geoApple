"""Evaluate GeoApple-Seg model on test set.

Computes pixel-level segmentation metrics and compares with E1 baseline.
Metrics: IoU, Dice, Precision, Recall, F1 (per-image and overall).
Also profiles inference speed and saves visual comparisons.

Usage:
    python scripts/evaluate.py --checkpoint runs/E7_v3/weights/best.pt --device cuda
    python scripts/evaluate.py --checkpoint runs/E7_v3/weights/best.pt --baseline-dir runs/E1_baseline_rgb
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_module.rgbd_dataset import build_rgbd_dataloader
from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pixel_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    """Compute pixel-level segmentation metrics.

    Args:
        pred: Predicted binary mask (H, W), values in {0, 1}.
        gt: Ground truth binary mask (H, W), values in {0, 1}.

    Returns:
        Dict with IoU, Dice, Precision, Recall, F1.
    """
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
# Model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_masks_avg(
    model: GeoAppleSegModel,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    threshold: float = 0.5,
    use_amodal: bool = False,
) -> np.ndarray:
    """Simple average-coefficient inference (original method).

    Args:
        model: GeoApple-Seg model in eval mode.
        rgb: RGB input (B, 3, H, W).
        depth: Depth input (B, 1, H, W).
        threshold: Sigmoid threshold for binarization.
        use_amodal: If True, use amodal_coeffs instead of modal_coeffs.

    Returns:
        Binary masks (B, H, W) as numpy array.
    """
    outputs = model(rgb, depth)
    proto = outputs["proto"]  # (B, nm, H, W)
    coeff_key = "amodal_coeffs" if use_amodal else "modal_coeffs"
    coeffs = outputs[coeff_key]  # (B, nm, N)

    B, nm, H, W = proto.shape
    avg = coeffs.mean(dim=2, keepdim=True)
    masks = torch.einsum("bmhw,bmk->bkhw", proto, avg).squeeze(1)
    probs = torch.sigmoid(masks)
    binary = (probs > threshold).cpu().numpy().astype(np.uint8)
    return binary


@torch.no_grad()
def predict_masks(
    model: GeoAppleSegModel,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    threshold: float = 0.5,
    top_k: int = 100,
    nms_threshold: float = 0.5,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Run model inference with per-anchor mask generation and NMS.

    Instead of averaging all anchor coefficients (which dilutes signal),
    we select top-K anchors by coefficient magnitude, generate per-instance
    masks, apply NMS to remove duplicates, then aggregate to per-image masks.

    Args:
        model: GeoApple-Seg model in eval mode.
        rgb: RGB input (B, 3, H, W), float [0, 1].
        depth: Depth input (B, 1, H, W), float [0, 1].
        threshold: Sigmoid threshold for binarization.
        top_k: Number of top anchors to consider per image.
        nms_threshold: IoU threshold for mask NMS.
        score_threshold: Minimum mask score to keep.

    Returns:
        Binary masks (B, H_mask, W_mask) as numpy array.
    """
    outputs = model(rgb, depth)
    proto = outputs["proto"]  # (B, nm, H, W)
    modal_coeffs = outputs["modal_coeffs"]  # (B, nm, N)

    B, nm, H, W = proto.shape
    result = np.zeros((B, H, W), dtype=np.uint8)

    for b in range(B):
        # proto_b: (nm, H, W), coeffs_b: (nm, N)
        proto_b = proto[b]
        coeffs_b = modal_coeffs[b]  # (nm, N)

        # Score each anchor by L2 norm of its coefficients
        # coeffs_b.T: (N, nm), each row is one anchor's coefficients
        anchor_scores = coeffs_b.norm(dim=0)  # (N,)

        # Select top-K anchors
        k = min(top_k, anchor_scores.shape[0])
        _, top_indices = anchor_scores.topk(k)

        # Generate masks for top-K anchors
        # top_coeffs: (nm, K)
        top_coeffs = coeffs_b[:, top_indices]

        # Per-anchor masks: proto_b (nm, H, W) @ top_coeffs (nm, K) -> (K, H, W)
        # einsum: for each anchor k, mask_k = sum_m(proto_b[m] * top_coeffs[m, k])
        anchor_masks = torch.einsum("mhw,mk->khw", proto_b, top_coeffs)
        anchor_probs = torch.sigmoid(anchor_masks)  # (K, H, W)

        # Score each mask by its mean probability in the positive region
        mask_scores = anchor_probs.amax(dim=(-2, -1))  # (K,) peak value

        # Filter by score threshold
        valid = mask_scores > score_threshold
        if valid.sum() == 0:
            continue

        anchor_probs = anchor_probs[valid]
        mask_scores = mask_scores[valid]

        # Apply mask NMS to remove duplicates
        keep = _mask_nms(anchor_probs, mask_scores, nms_threshold)
        anchor_probs = anchor_probs[keep]

        # Aggregate: max across all kept instance masks
        if len(anchor_probs) > 0:
            aggregated = anchor_probs.max(dim=0).values  # (H, W)
            result[b] = (aggregated > threshold).cpu().numpy().astype(np.uint8)

    return result


def _mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Simple mask-based NMS.

    Args:
        masks: Probability masks (N, H, W).
        scores: Confidence scores (N,).
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of kept masks.
    """
    # Sort by score descending
    order = scores.argsort(descending=True)
    masks = masks[order]
    binary = (masks > 0.5).float()

    keep = []
    suppressed = torch.zeros(len(masks), dtype=torch.bool, device=masks.device)

    for i in range(len(masks)):
        if suppressed[i]:
            continue
        keep.append(i)

        # Compute IoU with remaining masks
        mask_i = binary[i]
        for j in range(i + 1, len(masks)):
            if suppressed[j]:
                continue
            mask_j = binary[j]
            intersection = (mask_i * mask_j).sum()
            union = mask_i.sum() + mask_j.sum() - intersection
            iou = intersection / max(union.item(), 1)
            if iou > iou_threshold:
                suppressed[j] = True

    # Map back to original indices
    return order[torch.tensor(keep, device=masks.device)]


def segments_to_mask(
    segments: list,
    h: int,
    w: int,
) -> np.ndarray:
    """Convert YOLO polygon segments to a binary mask.

    Args:
        segments: List of polygon coordinate lists (normalized [0,1]).
        h: Target mask height.
        w: Target mask width.

    Returns:
        Binary mask (h, w) with all instances aggregated.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg in segments:
        coords = np.array(seg, dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        pts = coords.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def extract_amodal_gt_masks(
    batch: Dict[str, Any],
    mask_h: int,
    mask_w: int,
) -> np.ndarray:
    """Extract per-image amodal GT masks from batch polygon segments.

    Args:
        batch: Batch dict from dataloader (contains 'amodal_segments').
        mask_h: Target mask height (to match prediction size).
        mask_w: Target mask width.

    Returns:
        Per-image binary masks (B, mask_h, mask_w) as numpy array.
    """
    amodal_segments = batch.get("amodal_segments", [])
    B = batch["img"].shape[0]
    masks = np.zeros((B, mask_h, mask_w), dtype=np.uint8)
    for i in range(B):
        segs = amodal_segments[i] if i < len(amodal_segments) else []
        if segs:
            masks[i] = segments_to_mask(segs, mask_h, mask_w)
    return masks


def extract_gt_masks(
    batch: Dict[str, Any],
    device: str,
) -> np.ndarray:
    """Extract per-image GT masks from batch.

    Args:
        batch: Batch dict from dataloader.
        device: Device string.

    Returns:
        Per-image binary masks (B, H, W) as numpy array.
    """
    if "masks" not in batch:
        B = batch["img"].shape[0]
        return np.zeros((B, 160, 160), dtype=np.uint8)

    masks = batch["masks"].float()
    batch_idx = batch["batch_idx"].long()
    B = batch["img"].shape[0]

    if masks.dim() == 3 and masks.shape[0] != B:
        H, W = masks.shape[1], masks.shape[2]
        per_image = np.zeros((B, H, W), dtype=np.uint8)
        for i in range(B):
            inst = masks[batch_idx == i]
            if inst.numel() > 0:
                aggregated = inst.max(dim=0).values.clamp(0, 1)
                per_image[i] = (aggregated > 0.5).numpy().astype(np.uint8)
        return per_image
    else:
        return (masks.clamp(0, 1) > 0.5).numpy().astype(np.uint8)


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
    """Save side-by-side comparison image.

    Args:
        rgb: RGB image (H, W, 3), uint8.
        gt_mask: Ground truth mask (H_m, W_m), binary.
        pred_mask: Predicted mask (H_m, W_m), binary.
        save_path: Output file path.
        metrics: Per-image metrics dict.
    """
    h, w = rgb.shape[:2]

    # Resize masks to image size
    gt_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    pred_resized = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # Create overlay
    gt_overlay = rgb.copy()
    gt_overlay[gt_resized > 0] = (gt_overlay[gt_resized > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    pred_overlay = rgb.copy()
    pred_overlay[pred_resized > 0] = (pred_overlay[pred_resized > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

    # Concatenate: RGB | GT overlay | Pred overlay
    canvas = np.concatenate([rgb, gt_overlay, pred_overlay], axis=1)

    # Add metrics text
    text = f"IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}"
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "GT (green)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "Pred (red)", (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(str(save_path), canvas)


# ---------------------------------------------------------------------------
# Speed profiling
# ---------------------------------------------------------------------------

def profile_speed(
    model: GeoAppleSegModel,
    device: str,
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Profile inference speed."""
    model.eval()
    rgb = torch.randn(1, 3, 640, 640, device=device)
    depth = torch.randn(1, 1, 640, 640, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(rgb, depth)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(rgb, depth)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": float(np.mean(times)),
        "fps": float(1000.0 / np.mean(times)),
        "std_ms": float(np.std(times)),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GeoApple-Seg")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (best.pt)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default="avg", choices=["avg", "topk"],
                        help="Inference mode: avg (average coeffs) or topk (per-anchor + NMS)")
    parser.add_argument("--zero-depth", action="store_true",
                        help="Zero out depth input (for A1 ablation evaluation)")
    parser.add_argument("--eval-amodal", action="store_true",
                        help="Evaluate amodal mask quality instead of modal masks")
    parser.add_argument("--num-vis", type=int, default=20, help="Number of visualization samples")
    parser.add_argument("--speed-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("GeoApple-Seg Evaluation")
    logger.info("=" * 60)

    # --- Load model ---
    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False,
    )
    cfg = checkpoint.get("config", GeoAppleConfig())
    model = GeoAppleSegModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    logger.info(
        "Model loaded (epoch %d, best_metric=%.4f)",
        checkpoint.get("epoch", -1),
        checkpoint.get("best_metric", 0.0),
    )

    # --- Build test dataloader ---
    logger.info("Building test dataloader...")
    test_loader = build_rgbd_dataloader(
        dataset_yaml=str(DATASET_YAML),
        split="test",
        batch_size=args.batch,
        img_size=640,
        workers=4,
        shuffle=False,
    )
    logger.info("Test set: %d batches", len(test_loader))

    # --- Evaluate on test set ---
    logger.info("Running evaluation...")
    all_metrics = []
    vis_count = 0
    output_dir = Path(args.checkpoint).parent.parent
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    for batch_idx, batch in enumerate(test_loader):
        rgb = batch["img"].to(args.device).float() / 255.0
        depth = batch["depth"].to(args.device)
        B = rgb.shape[0]

        # Zero out depth for A1 ablation
        if args.zero_depth:
            depth = depth * 0.0

        # Model predictions
        if args.mode == "avg":
            pred_masks = predict_masks_avg(
                model, rgb, depth,
                threshold=args.threshold,
                use_amodal=args.eval_amodal,
            )
        else:
            pred_masks = predict_masks(model, rgb, depth, threshold=args.threshold)

        # Ground truth masks
        if args.eval_amodal:
            ph, pw = pred_masks.shape[-2:]
            gt_masks = extract_amodal_gt_masks(batch, ph, pw)
        else:
            gt_masks = extract_gt_masks(batch, args.device)

        # Ensure same spatial size
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            ph, pw = pred_masks.shape[-2:]
            gt_resized = np.zeros((B, ph, pw), dtype=np.uint8)
            for i in range(B):
                gt_resized[i] = cv2.resize(
                    gt_masks[i], (pw, ph), interpolation=cv2.INTER_NEAREST,
                )
            gt_masks = gt_resized

        # Compute per-image metrics
        for i in range(B):
            m = compute_pixel_metrics(pred_masks[i], gt_masks[i])
            all_metrics.append(m)

            # Save visualizations
            if vis_count < args.num_vis:
                rgb_np = batch["img"][i].permute(1, 2, 0).numpy().astype(np.uint8)
                save_comparison(
                    rgb_np, gt_masks[i], pred_masks[i],
                    vis_dir / f"test_{vis_count:04d}.jpg", m,
                )
                vis_count += 1

        if batch_idx % 10 == 0:
            logger.info("  Batch %d/%d", batch_idx, len(test_loader))

    # --- Aggregate results ---
    agg = aggregate_metrics(all_metrics)

    eval_type = "AMODAL" if args.eval_amodal else "MODAL"
    logger.info("=" * 60)
    logger.info("TEST SET RESULTS (GeoApple-Seg E7 — %s)", eval_type)
    logger.info("=" * 60)
    logger.info("  Images:     %d", len(all_metrics))
    logger.info("  IoU:        %.4f ± %.4f", agg["iou_mean"], agg["iou_std"])
    logger.info("  Dice:       %.4f ± %.4f", agg["dice_mean"], agg["dice_std"])
    logger.info("  Precision:  %.4f ± %.4f", agg["precision_mean"], agg["precision_std"])
    logger.info("  Recall:     %.4f ± %.4f", agg["recall_mean"], agg["recall_std"])
    logger.info("  F1:         %.4f ± %.4f", agg["f1_mean"], agg["f1_std"])

    # --- Speed profiling ---
    logger.info("Profiling inference speed...")
    speed = profile_speed(model, args.device, num_runs=args.speed_runs)
    logger.info(
        "Speed: %.1f ms/img (%.1f FPS, std=%.1f ms)",
        speed["avg_ms"], speed["fps"], speed["std_ms"],
    )

    # --- Save results to file ---
    results_path = output_dir / "eval_results.txt"
    with open(results_path, "w") as f:
        f.write("GeoApple-Seg Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', -1)}\n")
        f.write(f"Images: {len(all_metrics)}\n\n")
        f.write("Pixel-Level Segmentation Metrics:\n")
        f.write(f"  IoU:       {agg['iou_mean']:.4f} ± {agg['iou_std']:.4f}\n")
        f.write(f"  Dice:      {agg['dice_mean']:.4f} ± {agg['dice_std']:.4f}\n")
        f.write(f"  Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}\n")
        f.write(f"  Recall:    {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}\n")
        f.write(f"  F1:        {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}\n\n")
        f.write("Inference Speed:\n")
        f.write(f"  {speed['avg_ms']:.1f} ms/img ({speed['fps']:.1f} FPS)\n")
        f.write(f"Visualizations: {vis_dir}\n")

    logger.info("Results saved to: %s", results_path)
    logger.info("Visualizations saved to: %s", vis_dir)


if __name__ == "__main__":
    main()
