"""M2: Train YOLO26s-seg RGB-only baseline on AmodalAppleSize_RGB-D.

This establishes the E1 baseline for ablation studies.
"""

import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs"


def train_baseline() -> None:
    """Train YOLO26s-seg baseline (E1: RGB-only)."""
    logger.info("=" * 60)
    logger.info("M2: Training YOLO26s-seg RGB-only baseline (E1)")
    logger.info("=" * 60)

    # Load pretrained YOLO26s-seg
    model = YOLO("yolo26s-seg.pt")

    # Train
    results = model.train(
        data=str(DATASET_YAML),
        epochs=300,
        imgsz=640,
        batch=16,
        device="cuda",  # Apple Silicon GPU
        project=str(OUTPUT_DIR),
        name="E1_baseline_rgb",
        # Optimizer
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,  # Final LR factor (cosine decay)
        momentum=0.937,
        weight_decay=0.0005,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Training
        patience=50,
        save_period=50,
        val=True,
        plots=True,
        seed=42,
        # Performance
        workers=4,
        cache=False,  # Dataset too large to cache in RAM
    )

    logger.info("Training complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR / 'E1_baseline_rgb'}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device="cuda",
        project=str(OUTPUT_DIR),
        name="E1_baseline_rgb_test",
    )

    logger.info("Test results:")
    logger.info(f"  Mask mAP@0.5:     {metrics.seg.map50:.4f}")
    logger.info(f"  Mask mAP@0.5:0.95: {metrics.seg.map:.4f}")
    logger.info(f"  Box mAP@0.5:      {metrics.box.map50:.4f}")
    logger.info(f"  Box mAP@0.5:0.95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    train_baseline()
