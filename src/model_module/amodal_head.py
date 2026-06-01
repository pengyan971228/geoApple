"""Amodal mask prediction head with depth ordering (Contribution C2).

Extends YOLACT-style prototype mask prediction to output both:
- Modal masks (visible portion of each apple)
- Amodal masks (complete shape including occluded parts)

Depth ordering signal is injected into the amodal mask coefficient
prediction to help reason about occlusion relationships.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AmodalProto(nn.Module):
    """Depth-guided prototype generation.

    Generates K prototype masks using both RGB-D fused features
    and depth features for geometric structure awareness.

    Args:
        fused_channels: Input channels from fused features.
        depth_channels: Input channels from depth features (for injection).
        hidden_channels: Intermediate channels.
        num_protos: Number of prototype masks to generate.
    """

    def __init__(
        self,
        fused_channels: int,
        depth_channels: int,
        hidden_channels: int = 256,
        num_protos: int = 32,
    ) -> None:
        super().__init__()
        # Depth feature projection to match fused channels
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )
        # Prototype generation from combined features
        self.cv1 = nn.Sequential(
            nn.Conv2d(fused_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.ConvTranspose2d(
            hidden_channels, hidden_channels,
            kernel_size=2, stride=2, padding=0, bias=False,
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.cv3 = nn.Conv2d(hidden_channels, num_protos, kernel_size=1)

    def forward(
        self,
        fused_feat: torch.Tensor,
        depth_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Generate prototype masks.

        Args:
            fused_feat: Fused feature from F3 level, (B, C_fused, H, W).
            depth_feat: Depth feature from D3 level, (B, C_depth, H, W).

        Returns:
            Prototype masks, shape (B, num_protos, H*2, W*2).
        """
        depth_proj = self.depth_proj(depth_feat)
        combined = torch.cat([fused_feat, depth_proj], dim=1)
        x = self.cv1(combined)
        x = self.upsample(x)
        x = self.cv2(x)
        return self.cv3(x)


class AmodalSegmentHead(nn.Module):
    """Dual mask prediction head for modal and amodal segmentation.

    For each detected instance, predicts:
    1. Modal mask coefficients -> visible mask via prototype combination
    2. Amodal mask coefficients -> complete mask via prototype combination

    Depth ordering (relative depth rank among overlapping instances)
    is injected as an auxiliary feature for amodal coefficient prediction.

    Args:
        num_classes: Number of object classes.
        num_masks: Number of mask coefficients per instance.
        num_protos: Number of prototype masks.
        feat_channels: Feature channels at each detection level.
        fused_channels_p3: Channels of F3 (for prototype generation).
        depth_channels_p3: Channels of D3 (for depth-guided protos).
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_masks: int = 32,
        num_protos: int = 256,
        feat_channels: tuple = (128, 256, 512),
        fused_channels_p3: int = 128,
        depth_channels_p3: int = 128,
    ) -> None:
        super().__init__()
        self.nm = num_masks
        self.num_protos = num_protos

        # Depth-guided prototype generation
        self.proto = AmodalProto(
            fused_channels=fused_channels_p3,
            depth_channels=depth_channels_p3,
            hidden_channels=num_protos,
            num_protos=num_masks,
        )

        # Modal mask coefficient predictors (one per detection level)
        self.modal_cv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, num_masks, kernel_size=1),
            )
            for ch in feat_channels
        ])

        # Amodal mask coefficient predictors (with depth ordering input)
        # Extra +1 channel for depth ordering signal
        self.amodal_cv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch + 1, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, num_masks, kernel_size=1),
            )
            for ch in feat_channels
        ])

        logger.info(
            "AmodalSegmentHead initialized: nm=%d, protos=%d, levels=%d",
            num_masks, num_protos, len(feat_channels),
        )

    def forward(
        self,
        fused_features: List[torch.Tensor],
        depth_features: List[torch.Tensor],
        depth_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict modal and amodal mask coefficients.

        Args:
            fused_features: [F3, F4, F5] fused multi-scale features.
            depth_features: [D3, D4, D5] depth features.
            depth_map: Original depth map (B, 1, H, W) for ordering.

        Returns:
            Dict with keys:
                - proto: Prototype masks (B, nm, H_proto, W_proto)
                - modal_coeffs: Modal mask coefficients (B, nm, N_anchors)
                - amodal_coeffs: Amodal mask coefficients (B, nm, N_anchors)
        """
        bs = fused_features[0].shape[0]

        # Generate prototypes from P3 level (highest resolution)
        proto = self.proto(fused_features[0], depth_features[0])

        # Predict modal coefficients at each level
        modal_coeffs = []
        for i, (feat, cv) in enumerate(zip(fused_features, self.modal_cv)):
            coeff = cv(feat)  # (B, nm, Hi, Wi)
            modal_coeffs.append(coeff.view(bs, self.nm, -1))
        modal_coeffs = torch.cat(modal_coeffs, dim=2)  # (B, nm, N_total)

        # Create depth ordering maps at each feature level
        amodal_coeffs = []
        for i, (feat, cv) in enumerate(zip(fused_features, self.amodal_cv)):
            h, w = feat.shape[2:]
            if depth_map is not None:
                depth_order = F.interpolate(
                    depth_map, size=(h, w), mode="bilinear", align_corners=False,
                )
            else:
                depth_order = torch.zeros(bs, 1, h, w, device=feat.device)
            feat_with_depth = torch.cat([feat, depth_order], dim=1)
            coeff = cv(feat_with_depth)  # (B, nm, Hi, Wi)
            amodal_coeffs.append(coeff.view(bs, self.nm, -1))
        amodal_coeffs = torch.cat(amodal_coeffs, dim=2)  # (B, nm, N_total)

        return {
            "proto": proto,
            "modal_coeffs": modal_coeffs,
            "amodal_coeffs": amodal_coeffs,
        }
