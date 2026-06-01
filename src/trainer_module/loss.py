"""Custom loss functions for GeoApple-Seg.

Extends standard YOLO segmentation loss with:
1. Amodal mask loss (Dice + BCE for complete apple shapes)
2. Mask Continuity Loss (MCL) - penalizes mask discontinuities in
   continuous depth regions
3. Depth-Mask Boundary Loss - encourages mask edges to align with
   depth discontinuities
4. Size estimation loss (Smooth L1 for diameter regression)
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LossWeights:
    """Loss weight configuration."""

    modal_mask: float = 1.0
    amodal_mask: float = 0.05
    mask_continuity: float = 0.5
    boundary: float = 0.3
    size: float = 0.5
    # Warm-up epoch for geometric losses (MCL + boundary)
    geometric_warmup_epoch: int = 20
    # Warm-up epoch for amodal loss (delayed so modal stabilizes first)
    amodal_warmup_epoch: int = 30


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for mask prediction."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined Dice + BCE loss.

        Args:
            pred: Predicted mask logits, shape (N, H, W).
            target: Ground truth binary masks, shape (N, H, W).

        Returns:
            Combined loss scalar.
        """
        pred_sigmoid = torch.sigmoid(pred)

        # Dice loss
        intersection = (pred_sigmoid * target).sum(dim=(-2, -1))
        union = pred_sigmoid.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice.mean()

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="mean",
        )

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class MaskContinuityLoss(nn.Module):
    """Mask Continuity Loss (MCL).

    Penalizes mask value changes in image regions where depth is
    spatially continuous, encouraging geometric consistency between
    predicted masks and depth structure.
    """

    def forward(
        self,
        pred_mask: torch.Tensor,
        depth: torch.Tensor,
        threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute mask continuity loss.

        Args:
            pred_mask: Predicted mask probabilities, (N, H, W).
            depth: Depth map, (N, 1, H, W) or (N, H, W).
            threshold: Depth difference threshold for continuity.

        Returns:
            MCL loss scalar.
        """
        if depth.dim() == 4:
            depth = depth.squeeze(1)

        # Resize depth to match mask resolution
        if depth.shape[-2:] != pred_mask.shape[-2:]:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=pred_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # Input is already probabilities (post-sigmoid, post-max aggregation)
        pred_prob = pred_mask

        # Compute depth gradients
        depth_grad_x = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
        depth_grad_y = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])

        # Continuous regions: where depth gradient is small
        cont_x = (depth_grad_x < threshold).float()
        cont_y = (depth_grad_y < threshold).float()

        # Mask gradients
        mask_grad_x = torch.abs(pred_prob[:, :, 1:] - pred_prob[:, :, :-1])
        mask_grad_y = torch.abs(pred_prob[:, 1:, :] - pred_prob[:, :-1, :])

        # Penalize mask changes in continuous depth regions
        loss_x = (mask_grad_x * cont_x).mean()
        loss_y = (mask_grad_y * cont_y).mean()

        return loss_x + loss_y


class DepthBoundaryLoss(nn.Module):
    """Depth-Mask Boundary Alignment Loss.

    Encourages predicted mask boundaries to coincide with depth
    discontinuities (edges in depth map).
    """

    def forward(
        self,
        pred_mask: torch.Tensor,
        depth: torch.Tensor,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """Compute boundary alignment loss.

        Args:
            pred_mask: Predicted mask probabilities, (N, H, W).
            depth: Depth map, (N, 1, H, W) or (N, H, W).
            threshold: Depth gradient threshold for boundary detection.

        Returns:
            Boundary loss scalar.
        """
        if depth.dim() == 4:
            depth = depth.squeeze(1)

        if depth.shape[-2:] != pred_mask.shape[-2:]:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=pred_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # Input is already probabilities (post-sigmoid, post-max aggregation)
        pred_prob = pred_mask

        # Depth edges via Sobel-like gradient
        depth_grad_x = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
        depth_grad_y = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])

        # Depth boundaries
        depth_edge_x = (depth_grad_x > threshold).float()
        depth_edge_y = (depth_grad_y > threshold).float()

        # Mask edges
        mask_grad_x = torch.abs(pred_prob[:, :, 1:] - pred_prob[:, :, :-1])
        mask_grad_y = torch.abs(pred_prob[:, 1:, :] - pred_prob[:, :-1, :])

        # At depth boundaries, mask should also have edges (high gradient)
        # Loss: at depth edges, penalize low mask gradient
        loss_x = (depth_edge_x * (1.0 - mask_grad_x)).mean()
        loss_y = (depth_edge_y * (1.0 - mask_grad_y)).mean()

        return loss_x + loss_y


class GeoAppleSegLoss(nn.Module):
    """Combined loss function for GeoApple-Seg.

    Integrates all loss components with scheduled weighting.

    Args:
        weights: Loss weight configuration.
    """

    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights
        self.modal_loss = DiceBCELoss()
        self.amodal_loss = DiceBCELoss()
        self.mcl_loss = MaskContinuityLoss()
        self.boundary_loss = DepthBoundaryLoss()
        self.size_loss = nn.SmoothL1Loss()

        logger.info("GeoAppleSegLoss initialized: weights=%s", weights)

    def forward(
        self,
        predictions: dict,
        targets: dict,
        current_epoch: int = 0,
    ) -> dict:
        """Compute total loss.

        Args:
            predictions: Dict with model outputs:
                - modal_masks: (N, H, W) modal mask logits
                - amodal_masks: (N, H, W) amodal mask logits
                - size_pred: (N,) predicted diameters
                - depth: (B, 1, H, W) depth map
            targets: Dict with ground truth:
                - modal_masks: (N, H, W) binary masks
                - amodal_masks: (N, H, W) binary masks
                - size_gt: (N,) ground truth diameters

        Returns:
            Dict with individual and total losses.
        """
        losses = {}

        # Modal mask loss
        if "modal_masks" in predictions and "modal_masks" in targets:
            losses["modal_mask"] = self.modal_loss(
                predictions["modal_masks"], targets["modal_masks"],
            ) * self.weights.modal_mask

        # Amodal mask loss (with warmup)
        amodal_weight = self._amodal_warmup(current_epoch)
        if amodal_weight > 0 and "amodal_masks" in predictions and "amodal_masks" in targets:
            losses["amodal_mask"] = self.amodal_loss(
                predictions["amodal_masks"], targets["amodal_masks"],
            ) * self.weights.amodal_mask * amodal_weight

        # Geometric losses with warm-up
        geo_weight = self._geometric_warmup(current_epoch)

        if geo_weight > 0 and "modal_masks_per_image" in predictions and "depth" in predictions:
            # Geometric losses operate on per-image aggregated masks
            losses["mask_continuity"] = self.mcl_loss(
                predictions["modal_masks_per_image"], predictions["depth"],
            ) * self.weights.mask_continuity * geo_weight

            losses["boundary"] = self.boundary_loss(
                predictions["modal_masks_per_image"], predictions["depth"],
            ) * self.weights.boundary * geo_weight

        # Size estimation loss
        if "size_pred" in predictions and "size_gt" in targets:
            losses["size"] = self.size_loss(
                predictions["size_pred"], targets["size_gt"],
            ) * self.weights.size

        losses["total"] = sum(losses.values())
        return losses

    def _amodal_warmup(self, current_epoch: int) -> float:
        """Linear warm-up for amodal loss.

        Returns 0.0 before warmup epoch, then linearly increases to 1.0
        over 10 epochs. Lets modal branch stabilize first.
        """
        warmup_start = self.weights.amodal_warmup_epoch
        warmup_end = warmup_start + 10
        if current_epoch < warmup_start:
            return 0.0
        if current_epoch >= warmup_end:
            return 1.0
        return (current_epoch - warmup_start) / (warmup_end - warmup_start)

    def _geometric_warmup(self, current_epoch: int) -> float:
        """Linear warm-up for geometric losses.

        Returns 0.0 before warmup epoch, then linearly increases to 1.0
        over 10 epochs.
        """
        warmup_start = self.weights.geometric_warmup_epoch
        warmup_end = warmup_start + 10
        if current_epoch < warmup_start:
            return 0.0
        if current_epoch >= warmup_end:
            return 1.0
        return (current_epoch - warmup_start) / (warmup_end - warmup_start)
