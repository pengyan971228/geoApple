"""Generate paper-quality 6-panel qualitative figures for GeoApple-Seg.

Improvements vs v1:
    - Crop letterbox black borders before plotting.
    - Contour overlays (instead of fill) for GT and predictions, in distinct
      colors so they remain visible against green foliage.
    - Stratified sampling by occlusion ratio (low / mid / high) over the test
      set instead of taking the first N batches.
    - Larger panel titles and per-figure IoU + occlusion annotation.
    - Layout: 2 rows x 3 cols, RGB / Depth / GT-Modal on top,
              E7-Modal / E7-Amodal / A3-Amodal on bottom.
"""

import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import (
    extract_amodal_gt_masks,
    extract_gt_masks,
    predict_masks_avg,
)
from src.data_module.rgbd_dataset import build_rgbd_dataloader
from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel

DEVICE = "mps"
DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"
OUT_DIR = PROJECT_ROOT / "docs" / "paper_data" / "figures" / "qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Stratified targets: (low <0.3, mid 0.3-0.6, high >0.6) — 2 each = 6 figures
STRATA = [("low", 0.0, 0.30), ("mid", 0.30, 0.60), ("high", 0.60, 1.01)]
PER_STRATUM = 2
SCAN_BATCHES = 60  # batch_size 4 -> scan 240 images


def load_model(ckpt_path: Path) -> GeoAppleSegModel:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", GeoAppleConfig())
    try:
        cfg.yolo_weights = str(PROJECT_ROOT / "yolo26s-seg.pt")
    except Exception:
        cfg = replace(cfg, yolo_weights=str(PROJECT_ROOT / "yolo26s-seg.pt"))
    model = GeoAppleSegModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()


def crop_letterbox(rgb: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) bounds of real image content.

    Detects YOLO letterbox padding (114 gray) AND pure black padding.
    """
    is_pad = ((rgb == 114).all(axis=2)) | ((rgb == 0).all(axis=2))
    is_content = ~is_pad
    rows = np.where(is_content.any(axis=1))[0]
    cols = np.where(is_content.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, rgb.shape[0], 0, rgb.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def to_full_res(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def clean_mask(mask: np.ndarray, min_area_frac: float = 0.0003) -> np.ndarray:
    """Light per-instance cleanup: drop tiny noise blobs only.

    Avoids morphological opening to preserve apple body extent.
    """
    if mask.max() == 0:
        return mask
    m = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    min_area = max(20, int(min_area_frac * m.size))
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 1
    return out


def render_mask(rgb: np.ndarray, mask: np.ndarray, color,
                alpha: float = 0.45, thickness: int = 3,
                clean: bool = True) -> np.ndarray:
    """Semi-transparent fill + bold contour. Best-of-both visualization."""
    out = rgb.copy()
    if mask.max() == 0:
        return out
    m = clean_mask(mask) if clean else (mask > 0).astype(np.uint8)
    bm = m > 0
    if bm.any():
        out[bm] = (out[bm] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def draw_contours(rgb, mask, color, thickness=3, clean=True):
    return render_mask(rgb, mask, color, alpha=0.0, thickness=thickness, clean=clean)


def fill_overlay(rgb, mask, color, alpha=0.45, clean=True):
    return render_mask(rgb, mask, color, alpha=alpha, thickness=3, clean=clean)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def occlusion_ratio(modal: np.ndarray, amodal: np.ndarray) -> float:
    a = (amodal > 0).sum()
    if a == 0:
        return 0.0
    m = (modal > 0).sum()
    return float(1.0 - m / a)


def render_figure(sample: dict, save_stem: Path) -> None:
    rgb = sample["rgb"]
    depth_vis = sample["depth_vis"]
    gt_modal = sample["gt_modal"]
    e7_modal = sample["e7_modal"]
    e7_amodal = sample["e7_amodal"]
    a3_amodal = sample["a3_amodal"]
    iou_modal = sample["iou_modal"]
    iou_amodal = sample["iou_amodal"]
    occ = sample["occ"]

    panels = [
        (rgb, "RGB"),
        (depth_vis, "Depth"),
        (render_mask(rgb, gt_modal, (255, 0, 255), alpha=0.45), "GT Modal"),
        (render_mask(rgb, e7_modal, (255, 60, 60), alpha=0.45),
         f"E7 Pred Modal (IoU={iou_modal:.2f})"),
        (render_mask(rgb, e7_amodal, (60, 120, 255), alpha=0.40),
         f"E7 Pred Amodal (IoU={iou_amodal:.2f})"),
        (render_mask(rgb, a3_amodal, (240, 200, 60), alpha=0.40, clean=False),
         "A3 Pred Amodal (no amodal loss)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (img, title) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
    fig.suptitle(
        f"Stratum: {sample['stratum']}  |  Occlusion: {occ * 100:.0f}%",
        fontsize=15, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(f"{save_stem}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"{save_stem}.png", bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  saved {save_stem.name}.{{pdf,png}}")


def main() -> None:
    print("Loading models...")
    e7 = load_model(PROJECT_ROOT / "runs" / "E7_v7" / "weights" / "best.pt")
    a3 = load_model(PROJECT_ROOT / "runs" / "ablation_A3" / "weights" / "best.pt")

    print("Building test loader...")
    loader = build_rgbd_dataloader(
        dataset_yaml=str(DATASET_YAML), split="test",
        batch_size=4, img_size=640, workers=2, shuffle=False,
    )

    buckets: dict = {name: [] for name, _, _ in STRATA}
    seen_batches = 0

    for batch in loader:
        if seen_batches >= SCAN_BATCHES:
            break
        seen_batches += 1
        rgb_in = batch["img"].to(DEVICE).float() / 255.0
        depth_in = batch["depth"].to(DEVICE)
        B = rgb_in.shape[0]

        with torch.no_grad():
            e7_modal = predict_masks_avg(e7, rgb_in, depth_in, use_amodal=False)
            e7_amodal = predict_masks_avg(e7, rgb_in, depth_in, use_amodal=True)
            a3_amodal = predict_masks_avg(a3, rgb_in, depth_in, use_amodal=True)
        ph, pw = e7_modal.shape[-2:]
        gt_modal_b = extract_gt_masks(batch, DEVICE)
        gt_amodal_b = extract_amodal_gt_masks(batch, ph, pw)

        for i in range(B):
            rgb_full = batch["img"][i].permute(1, 2, 0).numpy().astype(np.uint8)
            y0, y1, x0, x1 = crop_letterbox(rgb_full)
            if (y1 - y0) < 100 or (x1 - x0) < 100:
                continue
            H, W = y1 - y0, x1 - x0

            depth_np = batch["depth"][i, 0].cpu().numpy()
            d = depth_np[y0:y1, x0:x1]
            d_norm = ((d - d.min()) / (np.ptp(d) + 1e-8) * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_VIRIDIS)[..., ::-1]

            rgb = rgb_full[y0:y1, x0:x1]

            gt_m = to_full_res(gt_modal_b[i], H, W)
            gt_a_full = to_full_res(gt_amodal_b[i], rgb_full.shape[0], rgb_full.shape[1])
            gt_a = gt_a_full[y0:y1, x0:x1]
            e7_m = to_full_res(e7_modal[i], H, W)
            e7_a = to_full_res(e7_amodal[i], H, W)
            a3_a = to_full_res(a3_amodal[i], H, W)

            occ = occlusion_ratio(gt_m, gt_a)
            # Skip degenerate samples: need real apples (modal GT) AND amodal GT.
            # Require at least ~1% of cropped area to be GT apples to filter
            # non-orchard / single-pot scenes.
            min_pix = max(500, int(0.01 * H * W))
            if (gt_m > 0).sum() < min_pix or (gt_a > 0).sum() < min_pix:
                continue

            stratum = None
            for name, lo, hi in STRATA:
                if lo <= occ < hi and len(buckets[name]) < PER_STRATUM:
                    stratum = name
                    break
            if stratum is None:
                continue

            sample = {
                "rgb": rgb,
                "depth_vis": depth_vis,
                "gt_modal": gt_m,
                "e7_modal": e7_m,
                "e7_amodal": e7_a,
                "a3_amodal": a3_a,
                "iou_modal": iou(e7_m, gt_m),
                "iou_amodal": iou(e7_a, gt_a),
                "occ": occ,
                "stratum": stratum,
            }
            buckets[stratum].append(sample)

        if all(len(buckets[s]) >= PER_STRATUM for s, _, _ in STRATA):
            break

    # Render
    idx = 0
    for name, _, _ in STRATA:
        for s in buckets[name]:
            render_figure(s, OUT_DIR / f"qual_{name}_{idx:02d}")
            idx += 1
    print(f"Done. {idx} figures in {OUT_DIR}")
    for name, _, _ in STRATA:
        print(f"  {name}: {len(buckets[name])} samples")


if __name__ == "__main__":
    main()
