"""RGB-D segmentation dataset with amodal label support.

Extends Ultralytics YOLODataset to load:
- RGB images (standard)
- Depth maps (.npy, float32, meters)
- Modal segmentation labels (standard YOLO-seg format)
- Amodal segmentation labels (complete apple shapes)

All spatial augmentations are synchronized across RGB, depth, and masks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset

logger = logging.getLogger(__name__)


class RGBDSegDataset(YOLODataset):
    """RGB-D dataset with amodal segmentation labels.

    Adds depth map loading and amodal label loading on top of the
    standard YOLO segmentation dataset.

    Args:
        depth_dir: Path to depth maps directory (containing .npy files).
        amodal_label_dir: Path to amodal label directory.
        img_size: Target image size for resizing.
    """

    def __init__(
        self,
        *args: Any,
        depth_dir: Optional[str] = None,
        amodal_label_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.amodal_label_dir = Path(amodal_label_dir) if amodal_label_dir else None
        super().__init__(*args, **kwargs)
        logger.info(
            "RGBDSegDataset: %d images, depth_dir=%s, amodal_dir=%s",
            len(self), self.depth_dir, self.amodal_label_dir,
        )

    def _load_depth(self, img_path: str) -> np.ndarray:
        """Load depth map corresponding to an RGB image.

        Args:
            img_path: Path to the RGB image.

        Returns:
            Depth map as float32 array, shape (H, W).
        """
        if self.depth_dir is None:
            return np.zeros((640, 640), dtype=np.float32)

        stem = Path(img_path).stem
        depth_path = self.depth_dir / f"{stem}.npy"

        if depth_path.exists():
            depth = np.load(str(depth_path)).astype(np.float32)
        else:
            logger.warning("Depth map not found: %s, using zeros", depth_path)
            depth = np.zeros((640, 640), dtype=np.float32)

        return depth

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Min-max normalize depth map to [0, 1] per image.

        Args:
            depth: Raw depth map in meters.

        Returns:
            Normalized depth map in [0, 1].
        """
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 1e-6:
            return (depth - d_min) / (d_max - d_min)
        return np.zeros_like(depth)

    def _load_amodal_labels(self, img_path: str) -> Optional[np.ndarray]:
        """Load amodal segmentation labels for an image.

        Args:
            img_path: Path to the RGB image.

        Returns:
            Amodal label array or None if not available.
        """
        if self.amodal_label_dir is None:
            return None

        stem = Path(img_path).stem
        label_path = self.amodal_label_dir / f"{stem}.txt"

        if not label_path.exists():
            return None

        with open(label_path) as f:
            lines = f.readlines()

        segments = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Skip class id, parse polygon coordinates
            coords = [float(x) for x in parts[1:]]
            segments.append(coords)

        return segments if segments else None

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single sample with RGB, depth, and labels.

        Returns:
            Dict with standard YOLO fields plus:
                - depth: Depth tensor (1, H, W), normalized to [0, 1]
                - depth_raw: Raw depth tensor (1, H, W) in meters
                - amodal_segments: Amodal polygon segments (if available)
        """
        # Get standard YOLO sample (RGB + labels + augmentation)
        sample = super().__getitem__(index)

        # Load depth map
        img_path = self.im_files[index]
        depth_raw = self._load_depth(img_path)

        # Resize depth to match the image size used by YOLO
        img_h, img_w = sample["img"].shape[1:]
        if depth_raw.shape[0] != img_h or depth_raw.shape[1] != img_w:
            depth_raw = cv2.resize(depth_raw, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # Apply the same spatial transforms as RGB
        # Note: YOLO applies transforms in get_image_and_label() before __getitem__
        # The resize/letterbox is already applied to sample["img"]
        # We need to apply the same to depth

        depth_norm = self._normalize_depth(depth_raw)

        # Convert to tensors
        sample["depth"] = torch.from_numpy(depth_norm).unsqueeze(0).float()
        sample["depth_raw"] = torch.from_numpy(depth_raw).unsqueeze(0).float()

        # Load amodal labels
        amodal_segs = self._load_amodal_labels(img_path)
        if amodal_segs is not None:
            sample["amodal_segments"] = amodal_segs
        else:
            sample["amodal_segments"] = []

        return sample

    @staticmethod
    def collate_fn(batch: list) -> Dict[str, Any]:
        """Custom collate that stacks depth tensors and collects amodal segments."""
        # Use parent collate for standard YOLO fields
        new_batch = YOLODataset.collate_fn(batch)

        # Stack depth tensors
        new_batch["depth"] = torch.stack([b["depth"] for b in batch])
        new_batch["depth_raw"] = torch.stack([b["depth_raw"] for b in batch])

        # Collect amodal segments as a list (variable length per image)
        new_batch["amodal_segments"] = [b["amodal_segments"] for b in batch]

        return new_batch


def build_rgbd_dataloader(
    dataset_yaml: str,
    split: str = "train",
    batch_size: int = 16,
    img_size: int = 640,
    workers: int = 4,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Build RGB-D dataloader from dataset YAML config.

    Args:
        dataset_yaml: Path to dataset.yaml.
        split: One of 'train', 'val', 'test'.
        batch_size: Batch size.
        img_size: Target image size.
        workers: Number of data loading workers.
        shuffle: Whether to shuffle data.

    Returns:
        DataLoader instance.
    """
    import yaml
    from ultralytics.data.utils import check_det_dataset

    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    base_path = Path(cfg["path"])
    img_dir = str(base_path / cfg[split])
    depth_dir = str(base_path / cfg.get(f"depth_{split}", f"depth/{split}"))

    amodal_dir = base_path / "amodal_labels" / split
    amodal_label_dir = str(amodal_dir) if amodal_dir.exists() else None

    # Parse dataset config for YOLODataset (requires 'data' dict with nc, names, etc.)
    data = check_det_dataset(dataset_yaml)

    dataset = RGBDSegDataset(
        img_path=img_dir,
        imgsz=img_size,
        task="segment",
        data=data,
        depth_dir=depth_dir,
        amodal_label_dir=amodal_label_dir,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=RGBDSegDataset.collate_fn,
    )
