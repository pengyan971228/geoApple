"""Custom trainer for GeoApple-Seg.

Handles two-phase training, custom loss scheduling, and RGB-D
data loading with synchronized augmentation.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model_module.geoapple_model import GeoAppleConfig, GeoAppleSegModel
from src.trainer_module.loss import GeoAppleSegLoss, LossWeights

logger = logging.getLogger(__name__)


class GeoAppleTrainer:
    """Trainer for GeoApple-Seg model.

    Implements two-phase training:
    - Phase 1 (epochs 1-100): Frozen backbone, train new modules
    - Phase 2 (epochs 101-300): Unfreeze all, joint fine-tuning

    Args:
        model: GeoApple-Seg model instance.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        output_dir: Directory for saving checkpoints and logs.
        device: Training device.
        epochs: Total training epochs.
        base_lr: Base learning rate.
        patience: Early stopping patience.
        save_period: Save checkpoint every N epochs.
    """

    def __init__(
        self,
        model: GeoAppleSegModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str = "runs/geoapple",
        device: str = "mps",
        epochs: int = 300,
        base_lr: float = 0.001,
        patience: int = 50,
        save_period: int = 50,
        warmup_epochs: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = epochs
        self.base_lr = base_lr
        self.patience = patience
        self.save_period = save_period
        self.warmup_epochs = warmup_epochs

        # Loss
        self.criterion = GeoAppleSegLoss(LossWeights())

        # Optimizer (will be set up per phase)
        self.optimizer = None
        self.scheduler = None

        # Tracking
        self.best_metric = float("-inf")
        self.patience_counter = 0
        self.current_epoch = 0

    def _setup_optimizer(self, phase: int) -> None:
        """Set up optimizer for the given training phase.

        Args:
            phase: 1 for frozen backbone, 2 for full fine-tuning.
        """
        if phase == 1:
            self.model.freeze_backbone()
            params = [
                {"params": self.model.depth_encoder.parameters()},
                {"params": self.model.fusion.parameters()},
                {"params": self.model.amodal_head.parameters()},
                {"params": self.model.size_head.parameters()},
            ]
            lr = self.base_lr  # Same LR as base for new modules
        else:
            self.model.unfreeze_backbone()
            params = self.model.get_param_groups(self.base_lr)
            lr = self.base_lr

        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=0.01,
        )

        # Cosine annealing with warmup
        remaining_epochs = self.epochs - self.current_epoch
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=remaining_epochs,
            eta_min=lr * 0.01,
        )

        # Linear warmup for the first N epochs of each phase
        warmup_steps = self.warmup_epochs
        if warmup_steps > 0 and phase == 1:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = cosine_scheduler

        logger.info(
            "Phase %d optimizer: lr=%.4f, %d param groups",
            phase, lr, len(params),
        )

    def train(self) -> Dict[str, Any]:
        """Run full training loop.

        Returns:
            Dict with training history and best metrics.
        """
        logger.info("=" * 60)
        logger.info("Starting GeoApple-Seg Training")
        logger.info("  Epochs: %d", self.epochs)
        logger.info("  Device: %s", self.device)
        logger.info("  Output: %s", self.output_dir)
        logger.info("=" * 60)

        history = {"train_loss": [], "val_loss": []}

        # Phase 1: Frozen backbone
        phase1_epochs = self.model.cfg.freeze_backbone_epochs
        self._setup_optimizer(phase=1)

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # Switch to Phase 2
            if epoch == phase1_epochs + 1:
                logger.info("=" * 40)
                logger.info("Switching to Phase 2: Full fine-tuning")
                logger.info("=" * 40)
                self._setup_optimizer(phase=2)

            # Train one epoch
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # Validate
            val_total, val_modal = self._validate_epoch(epoch)
            history["val_loss"].append(val_total)

            # Learning rate step
            self.scheduler.step()

            # Save checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping uses modal-only val loss, so amodal/geometric
            # loss components don't interfere with checkpoint selection.
            current_metric = -val_modal
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(
                    "  New best model saved (modal_val=%.4f, total_val=%.4f)",
                    val_modal, val_total,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch, self.patience,
                    )
                    break

        logger.info(
            "Training complete. Best modal_val_loss: %.4f", -self.best_metric,
        )
        return history

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            rgb = batch["img"].to(self.device).float() / 255.0
            depth = batch["depth"].to(self.device)
            depth_raw = batch.get("depth_raw", depth).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(rgb, depth, depth_raw)

            # Compute loss
            # Build target dict from batch
            targets = self._build_targets(batch)
            predictions = self._build_predictions(outputs, depth)

            # Ensure spatial dims match and targets are valid
            self._align_masks(predictions, targets)

            # Diagnostic logging (first batch only)
            if batch_idx == 0 and epoch == 1:
                self._log_mask_diagnostics(predictions, targets)

            losses = self.criterion(predictions, targets, current_epoch=epoch)
            loss = losses["total"]

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                logger.info(
                    "  Epoch %d [%d/%d] loss=%.4f",
                    epoch, batch_idx, len(self.train_loader), loss.item(),
                )

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d train: loss=%.4f, lr=%.6f, time=%.1fs",
            epoch, avg_loss, lr, elapsed,
        )
        return avg_loss

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> tuple:
        """Validate for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (total_val_loss, modal_val_loss).
            Modal loss is used for checkpoint selection and early stopping
            to prevent amodal/geometric loss components from interfering
            with model selection.
        """
        self.model.eval()
        total_loss = 0.0
        modal_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            rgb = batch["img"].to(self.device).float() / 255.0
            depth = batch["depth"].to(self.device)

            outputs = self.model(rgb, depth)

            targets = self._build_targets(batch)
            predictions = self._build_predictions(outputs, depth)
            self._align_masks(predictions, targets)

            losses = self.criterion(predictions, targets, current_epoch=epoch)
            total_loss += losses["total"].item()
            modal_loss += losses.get("modal_mask", torch.tensor(0.0)).item()
            num_batches += 1

        avg_total = total_loss / max(num_batches, 1)
        avg_modal = modal_loss / max(num_batches, 1)
        logger.info(
            "Epoch %d val: total_loss=%.4f, modal_loss=%.4f",
            epoch, avg_total, avg_modal,
        )
        return avg_total, avg_modal

    def _align_masks(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> None:
        """Ensure prediction and target masks have matching spatial dims and valid values."""
        for key in ("modal_masks", "amodal_masks"):
            if key in predictions and key in targets:
                pred = predictions[key]
                tgt = targets[key]
                # Resize target to match prediction spatial dims
                if pred.shape[-2:] != tgt.shape[-2:]:
                    tgt = F.interpolate(
                        tgt.unsqueeze(1),
                        size=pred.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                    targets[key] = tgt
                # Safety clamp (should already be [0,1] from _build_targets)
                targets[key] = tgt.clamp(0.0, 1.0)

    def _log_mask_diagnostics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> None:
        """Log shapes and value ranges for debugging (first batch only)."""
        for key in ("modal_masks", "amodal_masks"):
            if key in predictions:
                p = predictions[key]
                logger.info(
                    "DIAG pred[%s]: shape=%s, min=%.4f, max=%.4f",
                    key, list(p.shape), p.min().item(), p.max().item(),
                )
            if key in targets:
                t = targets[key]
                logger.info(
                    "DIAG target[%s]: shape=%s, min=%.4f, max=%.4f, unique_count=%d",
                    key, list(t.shape), t.min().item(), t.max().item(),
                    t.unique().numel(),
                )

    def _build_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract target tensors from batch dict.

        Args:
            batch: Batch dict from dataloader.

        Returns:
            Target dict for loss computation.
        """
        targets = {}

        # Modal masks from YOLO labels
        # YOLO collate produces masks of shape (N_instances, H, W) and
        # batch_idx in batch["batch_idx"] to identify which image each belongs to.
        # We aggregate to per-image masks (B, H, W) via max over instances.
        if "masks" in batch:
            masks = batch["masks"].to(self.device).float()
            batch_idx = batch["batch_idx"].to(self.device).long()
            B = batch["img"].shape[0]

            if masks.dim() == 3 and masks.shape[0] != B:
                # Aggregate instance masks to per-image masks
                H, W = masks.shape[1], masks.shape[2]
                per_image_masks = torch.zeros(B, H, W, device=self.device)
                for i in range(B):
                    instance_mask = masks[batch_idx == i]
                    if instance_mask.numel() > 0:
                        per_image_masks[i] = instance_mask.max(dim=0).values
                targets["modal_masks"] = per_image_masks.clamp(0.0, 1.0)
            else:
                targets["modal_masks"] = masks.clamp(0.0, 1.0)

        # Amodal masks from polygon segments
        if "amodal_segments" in batch:
            amodal_segments = batch["amodal_segments"]  # list of lists
            B = batch["img"].shape[0]
            # Use same spatial size as modal masks
            H = targets["modal_masks"].shape[1] if "modal_masks" in targets else 160
            W = targets["modal_masks"].shape[2] if "modal_masks" in targets else 160
            amodal_masks = torch.zeros(B, H, W, device=self.device)
            for i in range(B):
                segs = amodal_segments[i] if i < len(amodal_segments) else []
                if not segs:
                    continue
                mask_np = np.zeros((H, W), dtype=np.uint8)
                for seg in segs:
                    coords = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= W
                    coords[:, 1] *= H
                    pts = coords.astype(np.int32)
                    cv2.fillPoly(mask_np, [pts], 1)
                amodal_masks[i] = torch.from_numpy(mask_np).float()
            targets["amodal_masks"] = amodal_masks.clamp(0.0, 1.0)

        # Size ground truth (if available)
        if "size_gt" in batch:
            targets["size_gt"] = batch["size_gt"].to(self.device).float()

        return targets

    def _build_predictions(
        self,
        outputs: Dict[str, Any],
        depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build prediction dict for loss computation.

        Reconstructs instance masks from prototypes and coefficients.

        Args:
            outputs: Model forward outputs.
            depth: Depth tensor for geometric losses.

        Returns:
            Prediction dict for loss computation.
        """
        predictions = {"depth": depth}

        proto = outputs["proto"]  # (B, nm, H_p, W_p)
        modal_coeffs = outputs["modal_coeffs"]  # (B, nm, N)
        amodal_coeffs = outputs["amodal_coeffs"]  # (B, nm, N)

        # Reconstruct per-image masks from prototypes and coefficients.
        # proto: (B, nm, H, W), coeffs: (B, nm, N)
        # To avoid OOM (N can be ~8400), compute per-image aggregated masks
        # by averaging coefficients across anchors, then generating one mask per image.
        if proto is not None and modal_coeffs is not None:
            B, nm, H, W = proto.shape

            # Average coefficients across all anchor positions -> (B, nm, 1)
            modal_avg = modal_coeffs.mean(dim=2, keepdim=True)
            amodal_avg = amodal_coeffs.mean(dim=2, keepdim=True)

            # Generate per-image masks: (B, 1, H, W) -> (B, H, W)
            modal_masks = torch.einsum(
                "bmhw,bmk->bkhw", proto, modal_avg,
            ).squeeze(1)
            predictions["modal_masks"] = modal_masks

            amodal_masks = torch.einsum(
                "bmhw,bmk->bkhw", proto, amodal_avg,
            ).squeeze(1)
            predictions["amodal_masks"] = amodal_masks

            # Per-image aggregated masks for geometric losses
            predictions["modal_masks_per_image"] = torch.sigmoid(modal_masks)

        return predictions

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.model.cfg,
        }

        weights_dir = self.output_dir / "weights"
        weights_dir.mkdir(exist_ok=True)

        if is_best:
            path = weights_dir / "best.pt"
            torch.save(checkpoint, path)
            logger.info("Best checkpoint saved: %s", path)

        path = weights_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # Also save latest
        latest_path = weights_dir / "last.pt"
        torch.save(checkpoint, latest_path)
