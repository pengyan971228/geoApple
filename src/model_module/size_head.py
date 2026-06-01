"""Geometry-calibrated size estimation head (Contribution C3).

Lightweight MLP that predicts physical apple diameter in millimeters
from mask area (pixels), mean depth (meters), and camera focal length.

Geometric basis: diameter = sqrt(area) * depth / focal_length
The learned MLP applies corrections for depth noise, non-spherical
shapes, and SfM reconstruction errors.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SizeEstimationHead(nn.Module):
    """Apple physical size estimation MLP.

    Args:
        hidden_dim: Hidden layer dimension.
        focal_length: Camera focal length in pixels (default: 5805.34 for dataset).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        focal_length: float = 5805.34,
    ) -> None:
        super().__init__()
        self.focal_length = focal_length

        # Input: [mask_area_normalized, mean_depth, geometric_diameter_estimate]
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # Diameter is always positive
        )

        self._init_weights()
        logger.info(
            "SizeEstimationHead initialized: hidden=%d, focal=%.2f",
            hidden_dim, focal_length,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(
        self,
        mask_areas: torch.Tensor,
        mean_depths: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate physical apple diameter.

        Args:
            mask_areas: Mask area in pixels for each instance, shape (N,).
            mean_depths: Mean depth in meters for each instance, shape (N,).

        Returns:
            Predicted diameter in mm, shape (N,).
        """
        # Geometric estimate: diameter = sqrt(area) * depth / focal * 1000 (m->mm)
        sqrt_area = torch.sqrt(mask_areas.clamp(min=1.0))
        geo_diameter = sqrt_area * mean_depths / self.focal_length * 1000.0

        # Normalize inputs for MLP stability
        area_norm = sqrt_area / 100.0  # Rough normalization
        depth_norm = mean_depths  # Already in meters (typical 1-5m)
        geo_norm = geo_diameter / 100.0  # Rough normalization

        features = torch.stack([area_norm, depth_norm, geo_norm], dim=-1)
        correction = self.mlp(features).squeeze(-1)

        # Final prediction: geometric estimate + learned correction
        return geo_diameter + correction
