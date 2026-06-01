"""Run ablation experiments for GeoApple-Seg.

Ablation experiments:
    A0: Full model (E7_v4 config) — already trained
    A1: w/o Depth encoder (zero depth input)
    A2: w/o Geometric losses (MCL + Boundary = 0)
    A3: w/o Amodal branch (amodal_mask weight = 0)
    A4: w/o Depth dropout (depth_dropout = 0)

Usage:
    python scripts/train_ablation.py --ablation A1 --device cuda
    python scripts/train_ablation.py --ablation A2 --device cuda
    python scripts/train_ablation.py --ablation all --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_module.rgbd_dataset import build_rgbd_dataloader
from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel
from src.trainer_module.loss import LossWeights
from src.trainer_module.trainer import GeoAppleTrainer
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_YAML = PROJECT_ROOT / "data" / "yolo_format" / "dataset.yaml"

# E7_v7 config as baseline
BASE_CONFIG = {
    "lr": 0.001,
    "epochs": 300,
    "patience": 50,
    "warmup_epochs": 5,
    "depth_dropout": 0.5,
    "freeze_backbone_epochs": 50,
}

ABLATION_CONFIGS = {
    "A1": {
        "name": "w/o Depth (zero depth input)",
        "zero_depth": True,
    },
    "A2": {
        "name": "w/o Geometric Losses (MCL + Boundary)",
        "loss_weights": LossWeights(
            modal_mask=1.0,
            amodal_mask=0.8,
            mask_continuity=0.0,
            boundary=0.0,
            size=0.5,
        ),
    },
    "A3": {
        "name": "w/o Amodal Branch",
        "loss_weights": LossWeights(
            modal_mask=1.0,
            amodal_mask=0.0,
            mask_continuity=0.5,
            boundary=0.3,
            size=0.5,
        ),
    },
    "A4": {
        "name": "w/o Depth Dropout",
        "depth_dropout": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoApple-Seg Ablation Study")
    parser.add_argument(
        "--ablation", type=str, required=True,
        choices=["A1", "A2", "A3", "A4", "all"],
        help="Ablation experiment ID or 'all'",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_ablation(ablation_id: str, args: argparse.Namespace) -> None:
    """Run a single ablation experiment."""
    abl_cfg = ABLATION_CONFIGS[ablation_id]

    logger.info("=" * 60)
    logger.info("Ablation %s: %s", ablation_id, abl_cfg["name"])
    logger.info("=" * 60)

    set_seed(args.seed)

    # Determine depth dropout
    depth_dropout = abl_cfg.get("depth_dropout", BASE_CONFIG["depth_dropout"])
    freeze_epochs = abl_cfg.get(
        "freeze_backbone_epochs", BASE_CONFIG["freeze_backbone_epochs"],
    )

    # Build model config
    cfg = GeoAppleConfig(
        yolo_weights=str(PROJECT_ROOT / "yolo26s-seg.pt"),
        num_classes=1,
        img_size=640,
        depth_dropout=depth_dropout,
        focal_length=5805.34,
        freeze_backbone_epochs=freeze_epochs,
    )
    model = GeoAppleSegModel(cfg)

    # Build dataloaders
    zero_depth = abl_cfg.get("zero_depth", False)
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

    # Output directory
    output_dir = PROJECT_ROOT / "runs" / f"ablation_{ablation_id}"

    # Build trainer
    trainer = GeoAppleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        device=args.device,
        epochs=BASE_CONFIG["epochs"],
        base_lr=BASE_CONFIG["lr"],
        patience=BASE_CONFIG["patience"],
        warmup_epochs=BASE_CONFIG["warmup_epochs"],
    )

    # Override loss weights if specified
    if "loss_weights" in abl_cfg:
        trainer.criterion = __import__(
            "src.trainer_module.loss", fromlist=["GeoAppleSegLoss"],
        ).GeoAppleSegLoss(abl_cfg["loss_weights"])
        logger.info("Custom loss weights: %s", abl_cfg["loss_weights"])

    # Override depth to zeros for A1
    if zero_depth:
        original_train_epoch = trainer._train_epoch
        original_validate_epoch = trainer._validate_epoch

        def _train_epoch_zero_depth(epoch):
            """Wrap training to zero out depth."""
            for batch in trainer.train_loader:
                batch["depth"] = batch["depth"] * 0.0
                batch["depth_raw"] = batch["depth_raw"] * 0.0
            return original_train_epoch(epoch)

        # Simpler approach: monkey-patch the model forward to ignore depth
        original_forward = model.forward

        def forward_zero_depth(rgb, depth, depth_raw=None):
            zero_d = depth * 0.0
            zero_dr = depth_raw * 0.0 if depth_raw is not None else None
            return original_forward(rgb, zero_d, zero_dr)

        model.forward = forward_zero_depth
        logger.info("Depth input zeroed out for ablation A1")

    # Train
    history = trainer.train()
    logger.info("Ablation %s complete. Results: %s", ablation_id, output_dir)


def main() -> None:
    args = parse_args()

    if args.ablation == "all":
        for ablation_id in ["A1", "A2", "A3", "A4"]:
            run_ablation(ablation_id, args)
    else:
        run_ablation(args.ablation, args)


if __name__ == "__main__":
    main()
