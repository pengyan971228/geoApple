"""Evaluate apple diameter estimation (MAE, RMSE, R²).

Strategy: Use GT instance polygon masks + raw depth map → size_head → predicted diameter.
This evaluates the size estimation head's calibration accuracy given correct masks.

Two baselines:
1. Learned size head: size_head(mask_area, mean_depth) → diameter_mm
2. Geometric formula: sqrt(area) * depth / focal_length * 1000

Usage:
    python scripts/eval_diameter.py --checkpoint runs/E7_v7/weights/best.pt --device cuda
    python scripts/eval_diameter.py --replot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data" / "data"
GT_DIAMETER_FILE = DATA_ROOT / "GT_diameter.txt"
GT_JSON_DIR = DATA_ROOT / "gt_json" / "test"
DEPTH_DIR = DATA_ROOT / "depth_maps"
IMG_DIR = DATA_ROOT / "images" / "test"
OUT_DIR = PROJECT_ROOT / "docs" / "paper_data"
OUT_JSON = OUT_DIR / "diameter_eval_results.json"
OUT_FIG = OUT_DIR / "figures" / "F_diameter_scatter"
FOCAL_LENGTH = 5805.34


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_gt_diameters(batch_prefix: str = "2018_01") -> Dict[int, float]:
    """Load apple_ID -> diameter_mm from GT_diameter.txt.

    The file contains entries from multiple batches (2018_01, 2020_01,
    2020_02) with overlapping apple_IDs but different diameters.
    Only load entries matching the specified batch prefix.
    """
    diameters: Dict[int, float] = {}
    with open(GT_DIAMETER_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            label_file = parts[0]
            diameter = float(parts[1])
            stem = label_file.replace(".txt", "")
            # Extract batch prefix (e.g. "2018_01") and apple_id
            prefix = "_".join(stem.split("_")[:2])
            if prefix != batch_prefix:
                continue
            apple_id = int(stem.rsplit("_", 1)[1])
            diameters[apple_id] = diameter
    logger.info("Loaded %d GT diameters from batch '%s' (range: %.1f-%.1f mm)",
                len(diameters), batch_prefix,
                min(diameters.values()),
                max(diameters.values()))
    return diameters


def _parse_via_json(json_path: Path) -> Dict[str, Dict[int, np.ndarray]]:
    """Parse VIA JSON into {image_stem: {apple_id: polygon_points}}."""
    with open(json_path) as f:
        data = json.load(f)

    result: Dict[str, Dict[int, np.ndarray]] = {}
    for key, entry in data.items():
        stem = Path(entry["filename"]).stem
        apples: Dict[int, np.ndarray] = {}
        for rid, region in entry["regions"].items():
            apple_id_str = region.get("region_attributes", {}).get("apple_ID", "")
            if not apple_id_str or not apple_id_str.isdigit():
                continue
            apple_id = int(apple_id_str)
            shape = region["shape_attributes"]
            if shape["name"] == "polygon":
                xs = shape["all_points_x"]
                ys = shape["all_points_y"]
                apples[apple_id] = np.array(
                    list(zip(xs, ys)), dtype=np.float32,
                )
        if apples:
            result[stem] = apples
    return result


def load_test_gt_annotations() -> Tuple[
    Dict[str, Dict[int, np.ndarray]],
    Dict[str, Dict[int, np.ndarray]],
]:
    """Load both amodal and modal GT annotations.

    Returns:
        (amodal_annotations, modal_annotations) where each is
        {image_stem: {apple_id: polygon_points}}.

    Strategy: use amodal polygon for AREA (full apple size matches GT
    diameter) and modal polygon for DEPTH (depth is only valid where
    the apple is actually visible, not behind occluding objects).
    """
    amodal = _parse_via_json(GT_JSON_DIR / "via_region_data_amodal.json")
    modal = _parse_via_json(GT_JSON_DIR / "via_region_data_instance.json")
    logger.info("Loaded GT: amodal=%d images, modal=%d images",
                len(amodal), len(modal))
    return amodal, modal


def polygon_to_mask(points: np.ndarray, H: int, W: int) -> np.ndarray:
    """Rasterize polygon points to binary mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = points.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


# ---------------------------------------------------------------------------
# Diameter computation
# ---------------------------------------------------------------------------

def compute_geometric_diameter(
    mask_area: float,
    mean_depth: float,
    focal_length: float,
) -> float:
    """Compute diameter using pure geometric formula."""
    if mask_area <= 0 or mean_depth <= 0:
        return 0.0
    return float(np.sqrt(mask_area) * mean_depth / focal_length * 1000.0)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint: Path,
    device: str = "cuda",
) -> Dict:
    """Evaluate diameter estimation using GT masks + depth → size_head.

    Uses amodal polygon for AREA and modal polygon for DEPTH extraction.
    """
    gt_diameters = load_gt_diameters()
    amodal_ann, modal_ann = load_test_gt_annotations()

    # Load model (only need size_head)
    logger.info("Loading model from %s", checkpoint)
    cfg = GeoAppleConfig()
    model = GeoAppleSegModel(cfg)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()

    # Get test image stems
    test_stems = sorted([p.stem for p in IMG_DIR.glob("*.png")])
    logger.info("Found %d test images", len(test_stems))

    results_learned = []
    results_geometric = []
    n_evaluated = 0
    n_skipped_no_depth = 0
    n_skipped_no_valid_depth = 0
    n_skipped_no_modal = 0

    with torch.no_grad():
        for stem in test_stems:
            if stem not in amodal_ann:
                continue

            amodal_apples = amodal_ann[stem]
            modal_apples = modal_ann.get(stem, {})

            # Load depth map
            depth_path = DEPTH_DIR / f"{stem}.npy"
            if not depth_path.exists():
                n_skipped_no_depth += 1
                continue
            depth_map = np.load(depth_path)
            H, W = depth_map.shape[:2]

            for apple_id, amodal_pts in amodal_apples.items():
                if apple_id not in gt_diameters:
                    continue
                gt_diam = gt_diameters[apple_id]

                # AREA from amodal polygon (full apple shape)
                amodal_mask = polygon_to_mask(amodal_pts, H, W)
                mask_area = float(amodal_mask.sum())
                if mask_area < 1:
                    continue

                # DEPTH from modal polygon (visible region only)
                if apple_id not in modal_apples:
                    n_skipped_no_modal += 1
                    continue
                modal_mask = polygon_to_mask(modal_apples[apple_id], H, W)
                masked_depth = depth_map[modal_mask > 0]
                valid_depth = masked_depth[masked_depth > 0]
                if len(valid_depth) == 0:
                    n_skipped_no_valid_depth += 1
                    continue
                mean_depth = float(valid_depth.mean())

                # Geometric-only diameter
                geo_diam = compute_geometric_diameter(
                    mask_area, mean_depth, FOCAL_LENGTH,
                )

                # Learned diameter via size_head
                area_t = torch.tensor([mask_area], device=device)
                depth_t = torch.tensor([mean_depth], device=device)
                learned_diam = model.size_head(area_t, depth_t).item()

                results_learned.append((learned_diam, gt_diam, apple_id, stem))
                results_geometric.append((geo_diam, gt_diam, apple_id, stem))
                n_evaluated += 1

        if n_evaluated > 0 and n_evaluated % 50 == 0:
            logger.info("Evaluated %d instances so far", n_evaluated)

    logger.info(
        "Done: %d instances evaluated, %d skipped (no depth), "
        "%d skipped (no valid depth), %d skipped (no modal polygon)",
        n_evaluated, n_skipped_no_depth, n_skipped_no_valid_depth,
        n_skipped_no_modal,
    )

    # Compute metrics
    def compute_metrics(
        pairs: List[Tuple[float, float, int, str]],
    ) -> Dict:
        if not pairs:
            return {"mae": 0, "rmse": 0, "r2": 0, "n": 0}
        preds = np.array([p[0] for p in pairs])
        gts = np.array([p[1] for p in pairs])
        errors = preds - gts
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((gts - gts.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "n": len(pairs),
            "mean_pred": float(preds.mean()),
            "mean_gt": float(gts.mean()),
            "mean_error": float(errors.mean()),
            "predictions": preds.tolist(),
            "ground_truths": gts.tolist(),
        }

    metrics_learned = compute_metrics(results_learned)
    metrics_geometric = compute_metrics(results_geometric)

    output = {
        "checkpoint": str(checkpoint),
        "n_evaluated": n_evaluated,
        "learned_head": metrics_learned,
        "geometric_only": metrics_geometric,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", OUT_JSON)

    print("\n" + "=" * 60)
    print("DIAMETER ESTIMATION RESULTS")
    print(f"(GT masks + depth → size_head, n={n_evaluated})")
    print("=" * 60)
    print(f"\nLearned Size Head:")
    print(f"  MAE:  {metrics_learned['mae']:.2f} mm")
    print(f"  RMSE: {metrics_learned['rmse']:.2f} mm")
    print(f"  R²:   {metrics_learned['r2']:.4f}")
    print(f"  Mean pred: {metrics_learned.get('mean_pred', 0):.1f} mm")
    print(f"  Mean GT:   {metrics_learned.get('mean_gt', 0):.1f} mm")
    print(f"  Mean error: {metrics_learned.get('mean_error', 0):.2f} mm")
    print(f"\nGeometric Formula Only:")
    print(f"  MAE:  {metrics_geometric['mae']:.2f} mm")
    print(f"  RMSE: {metrics_geometric['rmse']:.2f} mm")
    print(f"  R²:   {metrics_geometric['r2']:.4f}")
    print(f"  Mean pred: {metrics_geometric.get('mean_pred', 0):.1f} mm")
    print(f"  Mean GT:   {metrics_geometric.get('mean_gt', 0):.1f} mm")
    print(f"  Mean error: {metrics_geometric.get('mean_error', 0):.2f} mm")
    print("=" * 60)

    return output


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diameter_scatter(data: Dict) -> None:
    """Generate scatter plot of predicted vs GT diameters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    for ax, (method_key, label, color) in zip(axes, [
        ("geometric_only", "Geometric Formula", "#D55E00"),
        ("learned_head", "GeoApple-Seg Size Head", "#0072B2"),
    ]):
        metrics = data[method_key]
        if not metrics.get("predictions"):
            continue

        preds = np.array(metrics["predictions"])
        gts = np.array(metrics["ground_truths"])

        ax.scatter(gts, preds, s=10, alpha=0.4, color=color, edgecolors="none")

        # Identity line
        lo = min(gts.min(), preds.min()) - 5
        hi = max(gts.max(), preds.max()) + 5
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1,
                label="y = x", zorder=0)

        # Regression line
        z = np.polyfit(gts, preds, 1)
        p = np.poly1d(z)
        x_line = np.linspace(lo, hi, 100)
        ax.plot(x_line, p(x_line), "-", color=color, linewidth=1.5,
                label=f"fit: y={z[0]:.2f}x+{z[1]:.1f}")

        ax.set_xlabel("Ground Truth Diameter (mm)", fontsize=9)
        ax.set_ylabel("Predicted Diameter (mm)", fontsize=9)
        ax.set_title(f"{label}\nMAE={metrics['mae']:.2f} mm, "
                     f"R²={metrics['r2']:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out_base = str(OUT_FIG)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s.{pdf,png}", out_base)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate diameter estimation")
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT_ROOT / "runs" / "E7_v7" / "weights" / "best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replot", action="store_true",
                    help="Re-plot from existing JSON without re-evaluating")
    args = ap.parse_args()

    if args.replot:
        if not OUT_JSON.exists():
            print(f"[ERROR] {OUT_JSON} not found, run without --replot first")
            return
        data = json.loads(OUT_JSON.read_text())
        plot_diameter_scatter(data)
        return

    data = run_evaluation(
        checkpoint=args.checkpoint,
        device=args.device,
    )
    plot_diameter_scatter(data)


if __name__ == "__main__":
    main()
