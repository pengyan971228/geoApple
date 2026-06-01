"""Train GeoApple-Seg: Full model with RGB-D fusion, amodal masks, and size estimation.

Usage:
    python scripts/train_geoapple.py [--device mps|cuda|cpu] [--epochs 300] [--batch 16]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_module.rgbd_dataset import build_rgbd_dataloader
from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel
from src.trainer_module.trainer import GeoAppleTrainer
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "runs" / "train_geoapple.log"),
    ],
)
logger = logging.getLogger(__name__)

DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoApple-Seg")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="runs/E7_full_model", help="Output dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("GeoApple-Seg Training")
    logger.info("=" * 60)
    logger.info("Args: %s", vars(args))

    # Build dataloaders
    logger.info("Building dataloaders...")
    train_loader = build_rgbd_dataloader(
        dataset_yaml=str(DATASET_YAML),
        split="train",
        batch_size=args.batch,
        img_size=640,
        workers=args.workers,
        shuffle=True,
    )
    val_loader = build_rgbd_dataloader(
        dataset_yaml=str(DATASET_YAML),
        split="val",
        batch_size=args.batch,
        img_size=640,
        workers=args.workers,
        shuffle=False,
    )
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    # Build model
    logger.info("Building GeoApple-Seg model...")
    cfg = GeoAppleConfig(
        yolo_weights=str(PROJECT_ROOT / "yolo26s-seg.pt"),
        num_classes=1,
        img_size=640,
        depth_dropout=0.5,
        focal_length=5805.34,
        freeze_backbone_epochs=50,
    )
    model = GeoAppleSegModel(cfg)

    # Train
    output_dir = PROJECT_ROOT / args.output
    trainer = GeoAppleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        device=args.device,
        epochs=args.epochs,
        base_lr=args.lr,
        patience=args.patience,
    )

    history = trainer.train()

    logger.info("Training complete!")
    logger.info("Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()