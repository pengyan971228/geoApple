"""Lightweight depth encoder for RGB-D fusion (Contribution C1).

Extracts multi-scale depth features D3, D4, D5 aligned with the
YOLO backbone feature pyramid levels C3, C4, C5.

Architecture: 4 conv blocks with stride-2 downsampling.
Parameters: ~2.1M (lightweight relative to RGB backbone).
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthEncoder(nn.Module):
    """Lightweight depth feature encoder.

    Produces multi-scale features D3, D4, D5 matching the spatial
    resolution and channel dimensions of the RGB backbone outputs.

    Args:
        in_channels: Input depth channels (1 for single-channel depth).
        out_channels: Output channel dimensions for D3, D4, D5.
            Must match the RGB backbone C3, C4, C5 channels.
        depth_dropout: Probability of zeroing the entire depth input
            during training for robustness to missing depth.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Tuple[int, int, int] = (256, 256, 512),
        depth_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.depth_dropout = depth_dropout
        c3_ch, c4_ch, c5_ch = out_channels

        # Stem: 1-channel depth -> 32 channels, stride 2 (640->320)
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1),
        )

        # Stage 1: 32 -> 64, stride 2 (320->160)
        self.stage1 = nn.Sequential(
            ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # Stage 2: 64 -> c3_ch, stride 2 (160->80) -> D3
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, c3_ch, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c3_ch, c3_ch, kernel_size=3, stride=1, padding=1),
        )

        # Stage 3: c3_ch -> c4_ch, stride 2 (80->40) -> D4
        self.stage3 = nn.Sequential(
            ConvBNReLU(c3_ch, c4_ch, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c4_ch, c4_ch, kernel_size=3, stride=1, padding=1),
        )

        # Stage 4: c4_ch -> c5_ch, stride 2 (40->20) -> D5
        self.stage4 = nn.Sequential(
            ConvBNReLU(c4_ch, c5_ch, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(c5_ch, c5_ch, kernel_size=3, stride=1, padding=1),
        )

        self._init_weights()
        logger.info(
            "DepthEncoder initialized: out_channels=%s, params=%.2fM",
            out_channels, sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def _init_weights(self) -> None:
        """Kaiming initialization for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, depth: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale depth features.

        Args:
            depth: Depth map tensor, shape (B, 1, H, W), normalized to [0, 1].

        Returns:
            List of [D3, D4, D5] feature tensors at stride 8, 16, 32.
        """
        # Depth dropout: zero entire depth input with probability p during training
        if self.training and self.depth_dropout > 0:
            mask = torch.rand(depth.shape[0], 1, 1, 1, device=depth.device)
            depth = depth * (mask > self.depth_dropout).float()

        x = self.stem(depth)       # stride 2
        x = self.stage1(x)         # stride 4
        d3 = self.stage2(x)        # stride 8
        d4 = self.stage3(d3)       # stride 16
        d5 = self.stage4(d4)       # stride 32

        return [d3, d4, d5]
