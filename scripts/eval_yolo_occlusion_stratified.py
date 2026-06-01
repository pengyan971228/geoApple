"""Occlusion-stratified evaluation for YOLOv8/YOLOv11-seg baselines.

For each test image:
- read modal GT polygons   from labels/test/<stem>.txt
- read amodal GT polygons  from amodal_labels/test/<stem>.txt
- compute occlusion = 1 - modal_area / amodal_area
- bin into Low [0, 0.30) / Medium [0.30, 0.60) / High [0.60, 1.0]
- run YOLO predict, compute pixel IoU/Prec/Rec against modal GT
- aggregate per bin

Outputs Markdown table to docs/paper_data/occlusion_stratified_baselines.md
combining v8, v11 with E7_v7 reference numbers from prior runs.

Usage:
    python scripts/eval_yolo_occlusion_stratified.py
    python scripts/eval_yolo_occlusion_stratified.py --models v8
    python scripts/eval_yolo_occlusion_stratified.py --device cuda --batch 32
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import compute_pixel_metrics  # noqa: E402

DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
OUT_MD = PROJECT_ROOT / "docs" / "paper_data" / "occlusion_stratified_baselines.md"

MODELS = {
    "v8":  ("YOLOv8s-seg",  RUNS_DIR / "baseline_yolov8s"  / "weights" / "best.pt"),
    "v11": ("YOLOv11s-seg", RUNS_DIR / "baseline_yolov11s" / "weights" / "best.pt"),
}

BINS = [
    ("Low",    0.0,  0.30),
    ("Medium", 0.30, 0.60),
    ("High",   0.60, 1.01),
]


def read_poly_mask(label_path: Path, H: int, W: int) -> np.ndarray:
    """Rasterize a YOLO-format polygon label file into a binary mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if not label_path.exists():
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= W
            coords[:, 1] *= H
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
    return mask


def aggregate_pred_masks(result, H: int, W: int) -> np.ndarray:
    """Union all instance masks from one ultralytics result into a binary mask."""
    out = np.zeros((H, W), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return out
    masks = result.masks.data.cpu().numpy()
    for m in masks:
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        out |= (m > 0.5).astype(np.uint8)
    return out


def occlusion_ratio(modal: np.ndarray, amodal: np.ndarray) -> float:
    a = (amodal > 0).sum()
    if a == 0:
        return -1.0
    return float(1.0 - (modal > 0).sum() / a)


def bin_name(occ: float) -> Optional[str]:
    for name, lo, hi in BINS:
        if lo <= occ < hi:
            return name
    return None


def evaluate_one(tag: str, label: str, weights: Path,
                 img_paths: List[Path], modal_dir: Path, amodal_dir: Path,
                 img_size: int, batch: int, device: str) -> Dict[str, dict]:
    print(f"\n{'='*60}\n[EVAL] {label}  ({weights})\n{'='*60}")
    if not weights.exists():
        print(f"[WARN] missing {weights}, skipping")
        return {}
    model = YOLO(str(weights))

    bucket: Dict[str, List[dict]] = defaultdict(list)
    n_skipped = 0

    for start in range(0, len(img_paths), batch):
        chunk = img_paths[start:start + batch]
        results = model.predict(
            [str(p) for p in chunk],
            imgsz=img_size, device=device, verbose=False, conf=0.25,
        )
        for p, res in zip(chunk, results):
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            modal_gt  = read_poly_mask(modal_dir  / (p.stem + ".txt"), H, W)
            amodal_gt = read_poly_mask(amodal_dir / (p.stem + ".txt"), H, W)
            occ = occlusion_ratio(modal_gt, amodal_gt)
            if occ < 0:
                n_skipped += 1
                continue
            name = bin_name(occ)
            if name is None:
                continue
            pred = aggregate_pred_masks(res, H, W)
            m = compute_pixel_metrics(pred, modal_gt)
            m["occ"] = occ
            bucket[name].append(m)
        print(f"    {min(start + batch, len(img_paths))}/{len(img_paths)}")

    print(f"  skipped {n_skipped} (no amodal GT)")
    summary: Dict[str, dict] = {}
    for name, lo, hi in BINS:
        ms = bucket[name]
        if not ms:
            summary[name] = {"n": 0}
            continue
        summary[name] = {
            "n": len(ms),
            "occ_mean": float(np.mean([m["occ"] for m in ms])),
            "iou_mean": float(np.mean([m["iou"] for m in ms])),
            "iou_std":  float(np.std([m["iou"] for m in ms])),
            "dice_mean": float(np.mean([m["dice"] for m in ms])),
            "precision_mean": float(np.mean([m["precision"] for m in ms])),
            "recall_mean":    float(np.mean([m["recall"] for m in ms])),
        }
    all_ms = [m for ms in bucket.values() for m in ms]
    if all_ms:
        summary["Overall"] = {
            "n": len(all_ms),
            "occ_mean": float(np.mean([m["occ"] for m in all_ms])),
            "iou_mean": float(np.mean([m["iou"] for m in all_ms])),
            "iou_std":  float(np.std([m["iou"] for m in all_ms])),
            "dice_mean": float(np.mean([m["dice"] for m in all_ms])),
            "precision_mean": float(np.mean([m["precision"] for m in all_ms])),
            "recall_mean":    float(np.mean([m["recall"] for m in all_ms])),
        }
    for name in [b[0] for b in BINS] + ["Overall"]:
        s = summary.get(name, {})
        if s.get("n", 0):
            print(f"  {name:8s} n={s['n']:4d}  IoU={s['iou_mean']:.4f}±{s['iou_std']:.4f}"
                  f"  P={s['precision_mean']:.4f}  R={s['recall_mean']:.4f}")
    return summary


def write_report(all_results: Dict[str, Dict[str, dict]]) -> None:
    """all_results: model_label -> bin_name -> metric dict."""
    # E7_v7 reference numbers (from runs/E7_v7/eval_results.json via project memory)
    e7_ref = {
        "Low":     {"n": 99,  "iou_mean": 0.8291, "iou_std": 0.0829,
                    "precision_mean": 0.9513, "recall_mean": 0.8661},
        "Medium":  {"n": 182, "iou_mean": 0.8168, "iou_std": 0.0960,
                    "precision_mean": 0.9426, "recall_mean": 0.8598},
        "High":    {"n": 334, "iou_mean": 0.6728, "iou_std": 0.3029,
                    "precision_mean": 0.7874, "recall_mean": 0.7203},
        "Overall": {"n": 615, "iou_mean": 0.7406, "iou_std": 0.2432,
                    "precision_mean": 0.8597, "recall_mean": 0.7850},
    }
    all_results["E7_v7 — GeoApple-Seg (Ours)"] = e7_ref

    lines = [
        "# Occlusion-Stratified Baseline Comparison (COMPAG)",
        "",
        "Modal segmentation performance binned by GT occlusion ratio "
        "(1 − modal_area / amodal_area). Same test split (n=799 → 615 with valid amodal GT).",
        "All models evaluated with our pixel-level metric pipeline.",
        "",
        "## IoU per occlusion bin",
        "",
        "| Model | Low n=99 | Medium n=182 | High n=334 | Overall n=615 |",
        "|---|---|---|---|---|",
    ]
    for label, res in all_results.items():
        row = [label]
        for name in ["Low", "Medium", "High", "Overall"]:
            s = res.get(name, {})
            if s.get("n", 0):
                row.append(f"{s['iou_mean']:.4f} ± {s['iou_std']:.4f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Precision per occlusion bin",
        "",
        "| Model | Low | Medium | High | Overall |",
        "|---|---|---|---|---|",
    ]
    for label, res in all_results.items():
        row = [label]
        for name in ["Low", "Medium", "High", "Overall"]:
            s = res.get(name, {})
            row.append(f"{s['precision_mean']:.4f}" if s.get("n", 0) else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Recall per occlusion bin",
        "",
        "| Model | Low | Medium | High | Overall |",
        "|---|---|---|---|---|",
    ]
    for label, res in all_results.items():
        row = [label]
        for name in ["Low", "Medium", "High", "Overall"]:
            s = res.get(name, {})
            row.append(f"{s['recall_mean']:.4f}" if s.get("n", 0) else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "E7_v7 row reproduced from runs/E7_v7 (see docs/paper_data/occlusion_stratified.md).",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD}\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v8", "v11"], choices=list(MODELS.keys()))
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(DATASET_YAML))
    root = Path(cfg["path"])
    img_dir = root / cfg["test"]
    modal_dir  = img_dir.parent.parent / "labels" / img_dir.name
    amodal_dir = img_dir.parent.parent / "amodal_labels" / img_dir.name
    print(f"img_dir={img_dir}")
    print(f"modal_dir={modal_dir}")
    print(f"amodal_dir={amodal_dir}")
    img_paths = sorted(
        [p for p in img_dir.glob("*.jpg")] + [p for p in img_dir.glob("*.png")]
    )
    print(f"{len(img_paths)} test images")

    all_results: Dict[str, Dict[str, dict]] = {}
    for tag in args.models:
        label, weights = MODELS[tag]
        res = evaluate_one(tag, label, weights, img_paths, modal_dir, amodal_dir,
                           args.img_size, args.batch, args.device)
        if res:
            all_results[label] = res
            run_dir = weights.parent.parent
            (run_dir / "occlusion_stratified.json").write_text(json.dumps(res, indent=2))

    if all_results:
        write_report(all_results)


if __name__ == "__main__":
    main()
