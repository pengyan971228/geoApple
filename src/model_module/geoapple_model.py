"""GeoApple-Seg: Full model integrating all components.

Wraps YOLO26s-seg backbone+neck with:
- Depth encoder (C1)
- FPN-level fusion (C1)
- Amodal mask head (C2)
- Size estimation head (C3)

YOLO26s-seg layer structure (verified empirically):
  Backbone (layers 0-10):
    Layer 4  (C3k2): 256ch, 80x80  -> C3 (stride 8)
    Layer 6  (C3k2): 256ch, 40x40  -> C4 (stride 16)
    Layer 10 (C2PSA): 512ch, 20x20 -> C5 (stride 32)
  Neck (layers 11-22, FPN+PAN):
    Layer 16 (C3k2): 128ch, 80x80  -> P3
    Layer 19 (C3k2): 256ch, 40x40  -> P4
    Layer 22 (C3k2): 512ch, 20x20  -> P5
  Head (layer 23): Segment26 (detection + mask)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO

from .amodal_head import AmodalSegmentHead
from .depth_encoder import DepthEncoder
from .fusion import FPNFusion
from .size_head import SizeEstimationHead

logger = logging.getLogger(__name__)

# YOLO26s-seg backbone feature layer indices and channels
BACKBONE_INDICES = (4, 6, 10)  # C3, C4, C5
BACKBONE_CHANNELS = (256, 256, 512)  # Verified empirically

# Neck output layer indices and channels
NECK_INDICES = (16, 19, 22)
NECK_CHANNELS = (128, 256, 512)

# Head layer index (excluded from our forward pass)
HEAD_INDEX = 23


@dataclass(frozen=True)
class GeoAppleConfig:
    """Configuration for GeoApple-Seg model."""

    # Backbone
    yolo_weights: str = "yolo26s-seg.pt"
    num_classes: int = 1
    img_size: int = 640

    # Depth encoder (outputs match backbone channels)
    depth_channels: Tuple[int, int, int] = BACKBONE_CHANNELS
    depth_dropout: float = 0.5

    # Fusion (at backbone output level)
    backbone_channels: Tuple[int, int, int] = BACKBONE_CHANNELS

    # Neck output channels (for custom heads)
    neck_channels: Tuple[int, int, int] = NECK_CHANNELS

    # Amodal head
    num_masks: int = 32
    num_protos: int = 256

    # Size head
    size_hidden_dim: int = 64
    focal_length: float = 5805.34

    # Training phases
    freeze_backbone_epochs: int = 50


class GeoAppleSegModel(nn.Module):
    """GeoApple-Seg: Geometry-Aware Apple Instance Segmentation.

    Architecture:
    1. Run YOLO backbone (layers 0-10) to get C3, C4, C5
    2. Run depth encoder to get D3, D4, D5
    3. Fuse at backbone level: Fi = Fuse(Ci, Di)
    4. Run YOLO neck (layers 11-22) on fused features to get P3, P4, P5
    5. Feed P3/P4/P5 to amodal head for mask prediction
    6. Use size head for diameter estimation

    Args:
        cfg: Model configuration.
    """

    def __init__(self, cfg: GeoAppleConfig = GeoAppleConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        # Load YOLO26s-seg layers
        self._init_yolo(cfg.yolo_weights)

        # Depth encoder (C1) — outputs match backbone channels
        self.depth_encoder = DepthEncoder(
            in_channels=1,
            out_channels=cfg.depth_channels,
            depth_dropout=cfg.depth_dropout,
        )

        # FPN fusion (C1) — fuses at backbone output level
        self.fusion = FPNFusion(
            rgb_channels=cfg.backbone_channels,
            depth_channels=cfg.depth_channels,
            out_channels=cfg.backbone_channels,  # Same channels for neck compatibility
        )

        # Amodal mask head (C2) — operates on neck outputs
        self.amodal_head = AmodalSegmentHead(
            num_classes=cfg.num_classes,
            num_masks=cfg.num_masks,
            num_protos=cfg.num_protos,
            feat_channels=cfg.neck_channels,
            fused_channels_p3=cfg.neck_channels[0],
            depth_channels_p3=cfg.neck_channels[0],  # After projection to neck dims
        )

        # Size estimation head (C3)
        self.size_head = SizeEstimationHead(
            hidden_dim=cfg.size_hidden_dim,
            focal_length=cfg.focal_length,
        )

        # Depth feature projection layers (backbone channels -> neck channels)
        # Needed because depth encoder outputs at backbone dims but amodal
        # head expects neck dims
        self.depth_proj = nn.ModuleList([
            nn.Conv2d(dc, nc, kernel_size=1, bias=False)
            if dc != nc else nn.Identity()
            for dc, nc in zip(cfg.depth_channels, cfg.neck_channels)
        ])

        self._log_param_count()

    def _init_yolo(self, weights_path: str) -> None:
        """Load YOLO model and separate into backbone and neck layers."""
        yolo = YOLO(weights_path)
        self.yolo_layers = yolo.model.model  # nn.Sequential of all layers

        # Verify expected structure
        num_layers = len(self.yolo_layers)
        logger.info(
            "YOLO loaded from %s: %d layers (backbone: 0-%d, neck: %d-%d, head: %d)",
            weights_path, num_layers,
            BACKBONE_INDICES[-1], BACKBONE_INDICES[-1] + 1,
            HEAD_INDEX - 1, HEAD_INDEX,
        )

    def _run_yolo_layers(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
        layer_outputs: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Run a range of YOLO layers with proper skip connections.

        Args:
            x: Input tensor (output of previous layer).
            start: First layer index (inclusive).
            end: Last layer index (exclusive).
            layer_outputs: Dict storing outputs for skip connections.

        Returns:
            Output of the last layer.
        """
        for i in range(start, end):
            layer = self.yolo_layers[i]
            f = layer.f

            # Determine input based on 'f' attribute (skip connections)
            if isinstance(f, int):
                x_in = layer_outputs[f] if f != -1 else x
            elif isinstance(f, list):
                x_in = [layer_outputs[j] if j != -1 else x for j in f]
            else:
                x_in = x

            x = layer(x_in)
            layer_outputs[i] = x

        return x

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass of GeoApple-Seg.

        Args:
            rgb: RGB input, shape (B, 3, H, W).
            depth: Normalized depth map, shape (B, 1, H, W), values in [0, 1].
            depth_raw: Raw depth in meters, shape (B, 1, H, W).

        Returns:
            Dict with proto, modal_coeffs, amodal_coeffs, neck_features.
        """
        layer_outputs: Dict[int, torch.Tensor] = {}

        # --- Step 1: Run YOLO backbone (layers 0 to 10) ---
        self._run_yolo_layers(rgb, start=0, end=BACKBONE_INDICES[-1] + 1, layer_outputs=layer_outputs)

        # Extract backbone features C3, C4, C5
        backbone_feats = [layer_outputs[i] for i in BACKBONE_INDICES]

        # --- Step 2: Depth encoder ---
        depth_feats = self.depth_encoder(depth)

        # --- Step 3: Fuse at backbone level ---
        fused_feats = self.fusion(backbone_feats, depth_feats)

        # Replace backbone outputs with fused features for neck
        for idx, fused in zip(BACKBONE_INDICES, fused_feats):
            layer_outputs[idx] = fused

        # --- Step 4: Run YOLO neck (layers 11 to 22) ---
        self._run_yolo_layers(
            fused_feats[-1],  # Last fused feature as sequential input
            start=BACKBONE_INDICES[-1] + 1,
            end=HEAD_INDEX,
            layer_outputs=layer_outputs,
        )

        # Extract neck outputs P3, P4, P5
        neck_feats = [layer_outputs[i] for i in NECK_INDICES]

        # --- Step 5: Amodal mask head ---
        # Project depth features from backbone channels to neck channels
        depth_feats_for_head = self._project_depth_feats(depth_feats)

        mask_outputs = self.amodal_head(
            neck_feats, depth_feats_for_head, depth_map=depth,
        )

        return {
            "proto": mask_outputs["proto"],
            "modal_coeffs": mask_outputs["modal_coeffs"],
            "amodal_coeffs": mask_outputs["amodal_coeffs"],
            "neck_features": neck_feats,
        }

    def _project_depth_feats(
        self,
        depth_feats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Project depth features from backbone channels to neck channels.

        Args:
            depth_feats: [D3, D4, D5] at backbone channel dimensions.

        Returns:
            Projected depth features at neck channel dimensions.
        """
        return [proj(feat) for proj, feat in zip(self.depth_proj, depth_feats)]

    def predict_size(
        self,
        modal_masks: torch.Tensor,
        depth_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Predict physical apple diameter from masks and depth.

        Args:
            modal_masks: Binary masks, shape (N, H, W).
            depth_raw: Raw depth in meters, shape (B, 1, H, W).

        Returns:
            Predicted diameter in mm, shape (N,).
        """
        mask_areas = modal_masks.sum(dim=(-2, -1)).float()
        masks_expanded = modal_masks.unsqueeze(1).float()
        depth_expanded = depth_raw.expand_as(masks_expanded)
        masked_depth = depth_expanded * masks_expanded
        depth_sum = masked_depth.sum(dim=(-2, -1)).squeeze(1)
        pixel_count = masks_expanded.sum(dim=(-2, -1)).squeeze(1).clamp(min=1)
        mean_depths = depth_sum / pixel_count
        return self.size_head(mask_areas, mean_depths)

    def freeze_backbone(self) -> None:
        """Freeze YOLO backbone+neck weights (Phase 1 training)."""
        for param in self.yolo_layers.parameters():
            param.requires_grad = False
        logger.info("YOLO backbone+neck frozen")

    def unfreeze_backbone(self, lr_scale: float = 0.1) -> None:
        """Unfreeze YOLO backbone+neck for fine-tuning (Phase 2)."""
        for param in self.yolo_layers.parameters():
            param.requires_grad = True
        logger.info("YOLO backbone+neck unfrozen (lr_scale=%.2f)", lr_scale)

    def get_param_groups(self, base_lr: float = 0.01) -> List[Dict[str, Any]]:
        """Get parameter groups with different learning rates."""
        backbone_params = list(self.yolo_layers.parameters())
        new_params = (
            list(self.depth_encoder.parameters())
            + list(self.fusion.parameters())
            + list(self.depth_proj.parameters())
            + list(self.amodal_head.parameters())
            + list(self.size_head.parameters())
        )

        return [
            {"params": backbone_params, "lr": base_lr * 0.1},
            {"params": new_params, "lr": base_lr},
        ]

    def _log_param_count(self) -> None:
        """Log parameter counts for each module."""
        modules = {
            "YOLO backbone+neck": self.yolo_layers,
            "Depth encoder": self.depth_encoder,
            "FPN fusion": self.fusion,
            "Depth projection": self.depth_proj,
            "Amodal head": self.amodal_head,
            "Size head": self.size_head,
        }
        total = 0
        for name, module in modules.items():
            count = sum(p.numel() for p in module.parameters())
            total += count
            logger.info("  %s: %.2fM params", name, count / 1e6)
        logger.info("  Total: %.2fM params", total / 1e6)
