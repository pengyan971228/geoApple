"""Multi-seed training and evaluation for GeoApple-Seg and YOLOv11s-seg.

Trains each model with 3 different seeds, evaluates on test set,
reports mean±std and paired t-test for statistical significance.

Usage:
    python scripts/run_multi_seed.py --models ours v11
    python scripts/run_multi_seed.py --models ours --skip-train
    python scripts/run_multi_seed.py --models ours v11 --device cuda --epochs 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import yaml
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import compute_pixel_metrics

DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
OUT_DIR = PROJECT_ROOT / "docs" / "paper_data"

SEEDS = [42, 123, 2024, 7, 314]


# ---------------------------------------------------------------------------
# GeoApple-Seg training
# ---------------------------------------------------------------------------

def train_ours(seed: int, device: str, epochs: int, batch: int) -> Path:
    """Train GeoApple-Seg with a given seed. Returns output dir."""
    run_name = f"E7_seed{seed}"
    output = RUNS_DIR / run_name
    best = output / "weights" / "best.pt"

    # Reuse existing E7_v7 for seed=42 (already trained)
    if seed == 42 and not best.exists():
        legacy = RUNS_DIR / "E7_v7" / "weights" / "best.pt"
        if legacy.exists():
            print(f"[LINK] Symlinking E7_v7 -> {run_name} (seed=42 already trained)")
            output.mkdir(parents=True, exist_ok=True)
            (output / "weights").mkdir(parents=True, exist_ok=True)
            best.symlink_to(legacy.resolve())
            return output

    if best.exists():
        print(f"[SKIP] {run_name} already trained ({best})")
        return output

    print(f"\n{'='*60}\n[TRAIN] GeoApple-Seg seed={seed} -> {run_name}\n{'='*60}")
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_geoapple.py"),
        "--device", device,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--seed", str(seed),
        "--output", f"runs/{run_name}",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] Training failed for seed={seed}")
    return output


# ---------------------------------------------------------------------------
# YOLOv11s-seg training
# ---------------------------------------------------------------------------

def train_v11(seed: int, device: str, epochs: int, batch: int) -> Path:
    """Train YOLOv11s-seg with a given seed."""
    from ultralytics import YOLO

    run_name = f"baseline_yolov11s_seed{seed}"
    output = RUNS_DIR / run_name
    best = output / "weights" / "best.pt"

    # Reuse existing baseline_yolov11s for seed=42
    if seed == 42 and not best.exists():
        legacy = RUNS_DIR / "baseline_yolov11s" / "weights" / "best.pt"
        if legacy.exists():
            print(f"[LINK] Symlinking baseline_yolov11s -> {run_name} (seed=42 already trained)")
            output.mkdir(parents=True, exist_ok=True)
            (output / "weights").mkdir(parents=True, exist_ok=True)
            best.symlink_to(legacy.resolve())
            return output

    if best.exists():
        print(f"[SKIP] {run_name} already trained ({best})")
        return output

    print(f"\n{'='*60}\n[TRAIN] YOLOv11s-seg seed={seed} -> {run_name}\n{'='*60}")
    model = YOLO("yolo11s-seg.pt")
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        verbose=True,
        patience=20,
        save=True,
        seed=seed,
    )
    print(f"[TRAIN] done -> {best}")
    return output


# ---------------------------------------------------------------------------
# Evaluation: GeoApple-Seg
# ---------------------------------------------------------------------------

def eval_ours(run_dir: Path, device: str, batch_size: int) -> Dict[str, float]:
    """Evaluate a GeoApple-Seg checkpoint on test set."""
    from dataclasses import replace
    from src.data_module.rgbd_dataset import build_rgbd_dataloader
    from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel
    from scripts.evaluate import extract_gt_masks, predict_masks_avg

    best = run_dir / "weights" / "best.pt"
    print(f"\n[EVAL] GeoApple-Seg: {best}")
    ckpt = torch.load(best, map_location=device, weights_only=False)
    cfg = ckpt.get("config", GeoAppleConfig())
    try:
        cfg.yolo_weights = str(PROJECT_ROOT / "yolo26s-seg.pt")
    except Exception:
        cfg = replace(cfg, yolo_weights=str(PROJECT_ROOT / "yolo26s-seg.pt"))
    model = GeoAppleSegModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    loader = build_rgbd_dataloader(
        dataset_yaml=str(DATASET_YAML), split="test",
        batch_size=batch_size, img_size=640, workers=2, shuffle=False,
    )

    all_metrics: List[Dict[str, float]] = []
    for batch in loader:
        rgb_in = batch["img"].to(device).float() / 255.0
        depth_in = batch["depth"].to(device)
        with torch.no_grad():
            pred = predict_masks_avg(model, rgb_in, depth_in, threshold=0.15)
        gt = extract_gt_masks(batch, device)
        if gt.shape[-2:] != pred.shape[-2:]:
            ph, pw = pred.shape[-2:]
            gt_r = np.zeros((gt.shape[0], ph, pw), dtype=np.uint8)
            for i in range(gt.shape[0]):
                gt_r[i] = cv2.resize(gt[i], (pw, ph), interpolation=cv2.INTER_NEAREST)
            gt = gt_r
        for i in range(pred.shape[0]):
            all_metrics.append(compute_pixel_metrics(pred[i], gt[i]))

    agg = {}
    for k in ["iou", "precision", "recall", "dice"]:
        vals = [m[k] for m in all_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    agg["n"] = len(all_metrics)
    print(f"  IoU={agg['iou_mean']:.4f}±{agg['iou_std']:.4f} "
          f"P={agg['precision_mean']:.4f} R={agg['recall_mean']:.4f} n={agg['n']}")
    return agg


# ---------------------------------------------------------------------------
# Evaluation: YOLOv11s-seg
# ---------------------------------------------------------------------------

def _read_gt_mask(label_path: Path, H: int, W: int) -> np.ndarray:
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


def _aggregate_pred_masks(result, H: int, W: int) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return out
    masks = result.masks.data.cpu().numpy()
    for m in masks:
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        out |= (m > 0.5).astype(np.uint8)
    return out


def eval_v11(run_dir: Path, device: str, batch_size: int) -> Dict[str, float]:
    """Evaluate a YOLOv11s-seg checkpoint on test set."""
    from ultralytics import YOLO

    best = run_dir / "weights" / "best.pt"
    print(f"\n[EVAL] YOLOv11s-seg: {best}")
    if not best.exists():
        print(f"  [WARN] missing {best}")
        return {}
    model = YOLO(str(best))

    cfg = yaml.safe_load(open(DATASET_YAML))
    root = Path(cfg["path"])
    img_dir = root / cfg["test"]
    label_dir = img_dir.parent.parent / "labels" / img_dir.name
    img_paths = sorted(
        [p for p in img_dir.glob("*.jpg")] + [p for p in img_dir.glob("*.png")]
    )

    all_metrics: List[Dict[str, float]] = []
    for start in range(0, len(img_paths), batch_size):
        chunk = img_paths[start:start + batch_size]
        results = model.predict(
            [str(p) for p in chunk],
            imgsz=640, device=device, verbose=False, conf=0.20,
        )
        for p, res in zip(chunk, results):
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            gt = _read_gt_mask(label_dir / (p.stem + ".txt"), H, W)
            pred = _aggregate_pred_masks(res, H, W)
            all_metrics.append(compute_pixel_metrics(pred, gt))

    agg = {}
    for k in ["iou", "precision", "recall", "dice"]:
        vals = [m[k] for m in all_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    agg["n"] = len(all_metrics)
    print(f"  IoU={agg['iou_mean']:.4f}±{agg['iou_std']:.4f} "
          f"P={agg['precision_mean']:.4f} R={agg['recall_mean']:.4f} n={agg['n']}")
    return agg


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_seeds(ours_results: List[Dict], v11_results: List[Dict],
                   seeds: List[int] = SEEDS) -> None:
    """Compare multi-seed results with paired t-test."""
    if len(ours_results) < 2 or len(v11_results) < 2:
        print("[WARN] Need at least 2 seeds for statistical comparison")
        return

    print(f"\n{'='*60}")
    print("MULTI-SEED STATISTICAL COMPARISON")
    print(f"{'='*60}")
    print(f"Seeds: {seeds[:len(ours_results)]}")
    print(f"N seeds: Ours={len(ours_results)}, v11={len(v11_results)}")
    print()

    for metric in ["iou", "precision", "recall"]:
        key = f"{metric}_mean"
        ours_vals = [r[key] for r in ours_results]
        v11_vals = [r[key] for r in v11_results]

        ours_mean = np.mean(ours_vals)
        ours_std = np.std(ours_vals, ddof=1)
        v11_mean = np.mean(v11_vals)
        v11_std = np.std(v11_vals, ddof=1)

        # Paired t-test (same seeds, same data split)
        if len(ours_vals) == len(v11_vals) and len(ours_vals) >= 2:
            t_stat, p_val = stats.ttest_rel(ours_vals, v11_vals)
        else:
            t_stat, p_val = stats.ttest_ind(ours_vals, v11_vals)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        diff = ours_mean - v11_mean
        sign = "+" if diff > 0 else ""

        print(f"  {metric.upper():10s}: Ours={ours_mean:.4f}±{ours_std:.4f}  "
              f"v11={v11_mean:.4f}±{v11_std:.4f}  "
              f"Δ={sign}{diff:.4f}  p={p_val:.4f} {sig}")

    print()
    print("Per-seed breakdown:")
    print(f"  {'Seed':>6s} | {'Ours IoU':>10s} | {'v11 IoU':>10s} | {'Δ':>8s}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*8}")
    for i, seed in enumerate(seeds[:min(len(ours_results), len(v11_results))]):
        o = ours_results[i]["iou_mean"]
        v = v11_results[i]["iou_mean"]
        d = o - v
        print(f"  {seed:>6d} | {o:>10.4f} | {v:>10.4f} | {d:>+8.4f}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(ours_all: List[Dict], v11_all: List[Dict],
                  seeds: List[int] = SEEDS) -> None:
    """Write multi-seed comparison markdown."""
    lines = [
        "# Multi-Seed Comparison (EAAI Supplementary)",
        "",
        f"Seeds: {seeds[:max(len(ours_all), len(v11_all))]}",
        f"Threshold: GeoApple-Seg thr=0.15, YOLOv11s-seg conf=0.20 (each model's best-IoU point)",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Model | IoU | Precision | Recall | Dice |",
        "|---|---|---|---|---|---|",
    ]
    for i, seed in enumerate(seeds):
        if i < len(ours_all):
            r = ours_all[i]
            lines.append(f"| {seed} | GeoApple-Seg | {r['iou_mean']:.4f} | "
                         f"{r['precision_mean']:.4f} | {r['recall_mean']:.4f} | "
                         f"{r['dice_mean']:.4f} |")
        if i < len(v11_all):
            r = v11_all[i]
            lines.append(f"| {seed} | YOLOv11s-seg | {r['iou_mean']:.4f} | "
                         f"{r['precision_mean']:.4f} | {r['recall_mean']:.4f} | "
                         f"{r['dice_mean']:.4f} |")

    lines += ["", "## Aggregated (mean ± std across seeds)", ""]
    lines.append("| Model | IoU | Precision | Recall |")
    lines.append("|---|---|---|---|")

    if ours_all:
        o_iou = [r["iou_mean"] for r in ours_all]
        o_p = [r["precision_mean"] for r in ours_all]
        o_r = [r["recall_mean"] for r in ours_all]
        lines.append(f"| GeoApple-Seg | {np.mean(o_iou):.4f} ± {np.std(o_iou, ddof=1):.4f} | "
                     f"{np.mean(o_p):.4f} ± {np.std(o_p, ddof=1):.4f} | "
                     f"{np.mean(o_r):.4f} ± {np.std(o_r, ddof=1):.4f} |")
    if v11_all:
        v_iou = [r["iou_mean"] for r in v11_all]
        v_p = [r["precision_mean"] for r in v11_all]
        v_r = [r["recall_mean"] for r in v11_all]
        lines.append(f"| YOLOv11s-seg | {np.mean(v_iou):.4f} ± {np.std(v_iou, ddof=1):.4f} | "
                     f"{np.mean(v_p):.4f} ± {np.std(v_p, ddof=1):.4f} | "
                     f"{np.mean(v_r):.4f} ± {np.std(v_r, ddof=1):.4f} |")

    out_path = OUT_DIR / "multi_seed_comparison.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ours", "v11"],
                    choices=["ours", "v11"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--skip-train", action="store_true",
                    help="Only evaluate existing checkpoints")
    args = ap.parse_args()

    seeds = args.seeds

    ours_results: List[Dict[str, float]] = []
    v11_results: List[Dict[str, float]] = []

    for seed in seeds:
        if "ours" in args.models:
            if not args.skip_train:
                run_dir = train_ours(seed, args.device, args.epochs, args.batch)
            else:
                run_dir = RUNS_DIR / f"E7_seed{seed}"
            if (run_dir / "weights" / "best.pt").exists():
                metrics = eval_ours(run_dir, args.device, args.eval_batch)
                ours_results.append(metrics)
                (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

        if "v11" in args.models:
            if not args.skip_train:
                run_dir = train_v11(seed, args.device, args.epochs, args.batch)
            else:
                run_dir = RUNS_DIR / f"baseline_yolov11s_seed{seed}"
            if (run_dir / "weights" / "best.pt").exists():
                metrics = eval_v11(run_dir, args.device, args.eval_batch)
                v11_results.append(metrics)
                (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save all results
    all_data = {"seeds": seeds, "ours": ours_results, "v11": v11_results}
    out_json = OUT_DIR / "multi_seed_results.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_data, indent=2))
    print(f"\nSaved raw data to {out_json}")

    # Statistical comparison
    if ours_results and v11_results:
        compare_seeds(ours_results, v11_results, seeds)

    # Write report
    write_report(ours_results, v11_results, seeds)


if __name__ == "__main__":
    main()
