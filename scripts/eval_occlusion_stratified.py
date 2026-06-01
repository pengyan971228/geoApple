"""Occlusion-stratified evaluation for GeoApple-Seg E7_v7.

For each test image, compute occlusion ratio = 1 - (modal_area / amodal_area)
based on GT polygons. Bin into [0, 0.3), [0.3, 0.6), [0.6, 1.0]. Report
modal IoU/Precision/Recall per bin to show robustness under heavy occlusion.

Outputs Markdown table to docs/paper_data/occlusion_stratified.md.
"""

import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import (
    compute_pixel_metrics,
    extract_amodal_gt_masks,
    extract_gt_masks,
    predict_masks_avg,
)
from src.data_module.rgbd_dataset import build_rgbd_dataloader
from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel

DEVICE = "mps"
DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"
CKPT = PROJECT_ROOT / "runs" / "E7_v7" / "weights" / "best.pt"
OUT_MD = PROJECT_ROOT / "docs" / "paper_data" / "occlusion_stratified.md"

BINS = [
    ("Low",    0.0,  0.30),
    ("Medium", 0.30, 0.60),
    ("High",   0.60, 1.01),
]


def load_model() -> GeoAppleSegModel:
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", GeoAppleConfig())
    try:
        cfg.yolo_weights = str(PROJECT_ROOT / "yolo26s-seg.pt")
    except Exception:
        cfg = replace(cfg, yolo_weights=str(PROJECT_ROOT / "yolo26s-seg.pt"))
    model = GeoAppleSegModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()


def occlusion_ratio(modal: np.ndarray, amodal: np.ndarray) -> float:
    a = (amodal > 0).sum()
    if a == 0:
        return -1.0  # invalid
    return float(1.0 - (modal > 0).sum() / a)


def bin_name(occ: float) -> str | None:
    for name, lo, hi in BINS:
        if lo <= occ < hi:
            return name
    return None


def main() -> None:
    print("Loading model...")
    model = load_model()
    print("Building test loader...")
    loader = build_rgbd_dataloader(
        dataset_yaml=str(DATASET_YAML), split="test",
        batch_size=8, img_size=640, workers=2, shuffle=False,
    )

    bucket_metrics: dict = defaultdict(list)
    n_skipped = 0
    n_total = 0

    for batch in loader:
        rgb = batch["img"].to(DEVICE).float() / 255.0
        depth = batch["depth"].to(DEVICE)
        with torch.no_grad():
            pred = predict_masks_avg(model, rgb, depth, use_amodal=False)

        ph, pw = pred.shape[-2:]
        gt_modal = extract_gt_masks(batch, DEVICE)
        gt_amodal = extract_amodal_gt_masks(batch, ph, pw)

        # Resize gt_modal to pred resolution if needed
        if gt_modal.shape[-2:] != (ph, pw):
            B = gt_modal.shape[0]
            tmp = np.zeros((B, ph, pw), dtype=np.uint8)
            for i in range(B):
                tmp[i] = cv2.resize(gt_modal[i], (pw, ph), interpolation=cv2.INTER_NEAREST)
            gt_modal = tmp

        for i in range(rgb.shape[0]):
            n_total += 1
            occ = occlusion_ratio(gt_modal[i], gt_amodal[i])
            if occ < 0:
                n_skipped += 1
                continue
            name = bin_name(occ)
            if name is None:
                continue
            m = compute_pixel_metrics(pred[i], gt_modal[i])
            m["occ"] = occ
            bucket_metrics[name].append(m)

    print(f"Total {n_total}, skipped {n_skipped} (no amodal GT)")
    for name in (b[0] for b in BINS):
        print(f"  {name}: {len(bucket_metrics[name])} images")

    # Build markdown table
    lines = [
        "# Occlusion-Stratified Evaluation (E7_v7, Test 799)",
        "",
        "Test images binned by occlusion ratio = 1 - modal_area / amodal_area.",
        "",
        "| Bin | Range | N | Mean Occ | Modal IoU | Dice | Precision | Recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, lo, hi in BINS:
        ms = bucket_metrics[name]
        if not ms:
            lines.append(f"| {name} | [{lo:.2f}, {hi:.2f}) | 0 | — | — | — | — | — |")
            continue
        n = len(ms)
        occ_mean = np.mean([m["occ"] for m in ms])
        iou = np.mean([m["iou"] for m in ms])
        iou_std = np.std([m["iou"] for m in ms])
        dice = np.mean([m["dice"] for m in ms])
        prec = np.mean([m["precision"] for m in ms])
        rec = np.mean([m["recall"] for m in ms])
        lines.append(
            f"| {name} | [{lo:.2f}, {hi:.2f}) | {n} | {occ_mean:.2f} | "
            f"{iou:.4f} ± {iou_std:.4f} | {dice:.4f} | {prec:.4f} | {rec:.4f} |"
        )

    # Overall
    all_ms = [m for ms in bucket_metrics.values() for m in ms]
    if all_ms:
        n = len(all_ms)
        iou = np.mean([m["iou"] for m in all_ms])
        iou_std = np.std([m["iou"] for m in all_ms])
        dice = np.mean([m["dice"] for m in all_ms])
        prec = np.mean([m["precision"] for m in all_ms])
        rec = np.mean([m["recall"] for m in all_ms])
        occ_mean = np.mean([m["occ"] for m in all_ms])
        lines.append(
            f"| **Overall** | [0,1] | **{n}** | {occ_mean:.2f} | "
            f"**{iou:.4f} ± {iou_std:.4f}** | {dice:.4f} | {prec:.4f} | {rec:.4f} |"
        )
    lines.append("")
    lines.append(f"Skipped {n_skipped} images without valid amodal GT.")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD}")
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
