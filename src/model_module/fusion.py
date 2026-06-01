"""FPN-level RGB-D fusion module (Contribution C1).

Fuses RGB backbone features (C3, C4, C5) with depth encoder features
(D3, D4, D5) via channel concatenation + 1x1 convolution at each
feature pyramid level.
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FPNFusionLayer(nn.Module):
    """Single-level feature fusion: concat + 1x1 conv."""

    def __init__(self, rgb_channels: int, depth_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(rgb_channels + depth_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        """Fuse RGB and depth features at one pyramid level.

        Args:
            rgb_feat: RGB feature, shape (B, C_rgb, H, W).
            depth_feat: Depth feature, shape (B, C_depth, H, W).

        Returns:
            Fused feature, shape (B, C_out, H, W).
        """
        return self.conv(torch.cat([rgb_feat, depth_feat], dim=1))


class FPNFusion(nn.Module):
    """Multi-scale FPN-level fusion for RGB-D features.

    Args:
        rgb_channels: Channel dimensions of C3, C4, C5 from RGB backbone.
        depth_channels: Channel dimensions of D3, D4, D5 from depth encoder.
        out_channels: Output channel dimensions for F3, F4, F5.
            Typically same as rgb_channels to maintain compatibility with neck.
    """

    def __init__(
        self,
        rgb_channels: Tuple[int, int, int] = (128, 256, 512),
        depth_channels: Tuple[int, int, int] = (128, 256, 512),
        out_channels: Tuple[int, int, int] = (128, 256, 512),
    ) -> None:
        super().__init__()
        self.fusion_layers = nn.ModuleList([
            FPNFusionLayer(rc, dc, oc)
            for rc, dc, oc in zip(rgb_channels, depth_channels, out_channels)
        ])
        logger.info(
            "FPNFusion initialized: rgb=%s, depth=%s, out=%s",
            rgb_channels, depth_channels, out_channels,
        )

    def forward(
        self,
        rgb_features: List[torch.Tensor],
        depth_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Fuse multi-scale RGB and depth features.

        Args:
            rgb_features: [C3, C4, C5] from RGB backbone.
            depth_features: [D3, D4, D5] from depth encoder.

        Returns:
            [F3, F4, F5] fused features.
        """
        return [
            fusion(rgb, depth)
            for fusion, rgb, depth in zip(
                self.fusion_layers, rgb_features, depth_features,
            )
        ]
