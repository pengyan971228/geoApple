"""Find and visualize worst-case failures of GeoApple-Seg E7_v7.

Strategy: scan test set, score each image by modal IoU, keep the bottom-K
samples that still have meaningful GT (filter degenerate / non-orchard).
Render with the same 6-panel layout as visualize_paper.py.
"""

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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
from scripts.visualize_paper import (
    crop_letterbox,
    iou,
    occlusion_ratio,
    render_mask,
    to_full_res,
)
from src.data_module.rgbd_dataset import build_rgbd_dataloader
from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel

DEVICE = "mps"
DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"
OUT_DIR = PROJECT_ROOT / "docs" / "paper_data" / "figures" / "failures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_FAILURES = 6


def load_models():
    def _load(p):
        ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
        cfg = ckpt.get("config", GeoAppleConfig())
        try:
            cfg.yolo_weights = str(PROJECT_ROOT / "yolo26s-seg.pt")
        except Exception:
            cfg = replace(cfg, yolo_weights=str(PROJECT_ROOT / "yolo26s-seg.pt"))
        m = GeoAppleSegModel(cfg)
        m.load_state_dict(ckpt["model_state_dict"])
        return m.to(DEVICE).eval()

    return (
        _load(PROJECT_ROOT / "runs" / "E7_v7" / "weights" / "best.pt"),
        _load(PROJECT_ROOT / "runs" / "ablation_A3" / "weights" / "best.pt"),
    )


def render_figure(sample: dict, save_stem: Path) -> None:
    rgb = sample["rgb"]
    panels = [
        (rgb, "RGB"),
        (sample["depth_vis"], "Depth"),
        (render_mask(rgb, sample["gt_modal"], (255, 0, 255), alpha=0.45), "GT Modal"),
        (render_mask(rgb, sample["e7_modal"], (255, 60, 60), alpha=0.45),
         f"E7 Pred Modal (IoU={sample['iou_modal']:.2f})"),
        (render_mask(rgb, sample["e7_amodal"], (60, 120, 255), alpha=0.40),
         f"E7 Pred Amodal (IoU={sample['iou_amodal']:.2f})"),
        (render_mask(rgb, sample["a3_amodal"], (240, 200, 60), alpha=0.40, clean=False),
         "A3 Pred Amodal"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (img, t) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(t, fontsize=14, fontweight="bold")
        ax.axis("off")
    fig.suptitle(
        f"FAILURE  |  Occlusion: {sample['occ'] * 100:.0f}%  |  "
        f"Modal IoU: {sample['iou_modal']:.2f}",
        fontsize=15, y=1.00, color="darkred",
    )
    plt.tight_layout()
    plt.savefig(f"{save_stem}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"{save_stem}.png", bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  saved {save_stem.name}")


def main() -> None:
    print("Loading models...")
    e7, a3 = load_models()
    print("Building test loader...")
    loader = build_rgbd_dataloader(
        dataset_yaml=str(DATASET_YAML), split="test",
        batch_size=8, img_size=640, workers=2, shuffle=False,
    )

    candidates = []  # list of dicts
    for batch in loader:
        rgb_in = batch["img"].to(DEVICE).float() / 255.0
        depth_in = batch["depth"].to(DEVICE)
        with torch.no_grad():
            e7_modal = predict_masks_avg(e7, rgb_in, depth_in, use_amodal=False)
            e7_amodal = predict_masks_avg(e7, rgb_in, depth_in, use_amodal=True)
            a3_amodal = predict_masks_avg(a3, rgb_in, depth_in, use_amodal=True)
        ph, pw = e7_modal.shape[-2:]
        gt_modal_b = extract_gt_masks(batch, DEVICE)
        gt_amodal_b = extract_amodal_gt_masks(batch, ph, pw)
        if gt_modal_b.shape[-2:] != (ph, pw):
            B = gt_modal_b.shape[0]
            tmp = np.zeros((B, ph, pw), dtype=np.uint8)
            for i in range(B):
                tmp[i] = cv2.resize(gt_modal_b[i], (pw, ph), interpolation=cv2.INTER_NEAREST)
            gt_modal_b = tmp

        for i in range(rgb_in.shape[0]):
            iou_m = compute_pixel_metrics(e7_modal[i], gt_modal_b[i])["iou"]
            occ = occlusion_ratio(gt_modal_b[i], gt_amodal_b[i])
            # Need real GT (filter empties) and high occlusion
            if (gt_modal_b[i] > 0).sum() < 200 or (gt_amodal_b[i] > 0).sum() < 200:
                continue
            if occ < 0.4:
                continue
            candidates.append({
                "iou": iou_m,
                "occ": occ,
                "idx": (id(batch), i),
                "rgb_full": batch["img"][i].permute(1, 2, 0).numpy().astype(np.uint8),
                "depth_full": batch["depth"][i, 0].cpu().numpy(),
                "gt_modal": gt_modal_b[i],
                "gt_amodal": gt_amodal_b[i],
                "e7_modal": e7_modal[i],
                "e7_amodal": e7_amodal[i],
                "a3_amodal": a3_amodal[i],
            })

    print(f"Scanned {len(candidates)} candidate images (occ >= 0.4)")
    candidates.sort(key=lambda c: c["iou"])
    worst = candidates[:NUM_FAILURES]

    for k, c in enumerate(worst):
        rgb_full = c["rgb_full"]
        y0, y1, x0, x1 = crop_letterbox(rgb_full)
        if (y1 - y0) < 100 or (x1 - x0) < 100:
            y0, y1, x0, x1 = 0, rgb_full.shape[0], 0, rgb_full.shape[1]
        H, W = y1 - y0, x1 - x0
        rgb = rgb_full[y0:y1, x0:x1]

        d = c["depth_full"][y0:y1, x0:x1]
        d_norm = ((d - d.min()) / (np.ptp(d) + 1e-8) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_VIRIDIS)[..., ::-1]

        gt_a_full = to_full_res(c["gt_amodal"], rgb_full.shape[0], rgb_full.shape[1])
        sample = {
            "rgb": rgb,
            "depth_vis": depth_vis,
            "gt_modal": to_full_res(c["gt_modal"], H, W),
            "gt_amodal": gt_a_full[y0:y1, x0:x1],
            "e7_modal": to_full_res(c["e7_modal"], H, W),
            "e7_amodal": to_full_res(c["e7_amodal"], H, W),
            "a3_amodal": to_full_res(c["a3_amodal"], H, W),
            "occ": c["occ"],
            "iou_modal": c["iou"],
        }
        sample["iou_amodal"] = iou(sample["e7_amodal"], sample["gt_amodal"])
        render_figure(sample, OUT_DIR / f"failure_{k:02d}")

    print(f"Done. {len(worst)} failures in {OUT_DIR}")
    for k, c in enumerate(worst):
        print(f"  failure_{k:02d}: IoU={c['iou']:.3f} Occ={c['occ']:.2f}")


if __name__ == "__main__":
    main()
