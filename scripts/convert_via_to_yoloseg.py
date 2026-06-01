"""Convert AmodalAppleSize_RGB-D VIA annotations to YOLO-seg format.

VIA format: JSON with polygon annotations (all_points_x, all_points_y)
YOLO-seg format: class_id x1 y1 x2 y2 ... (normalized polygon coordinates)

Also creates symlinks/copies for images and depth maps organized by split.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_ROOT = Path(__file__).parent.parent / "data" / "data"
OUTPUT_ROOT = Path(__file__).parent.parent / "data" / "yolo_format"

# Image resolution (from dataset paper)
IMG_WIDTH = 1300
IMG_HEIGHT = 1300
CLASS_ID = 0  # Single class: apple


def load_via_json(json_path: Path) -> Dict:
    """Load VIA JSON annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} entries from {json_path.name}")
    return data


def extract_polygons(via_data: Dict) -> Dict[str, List[List[Tuple[float, float]]]]:
    """Extract polygon annotations per image from VIA JSON.

    Returns:
        Dict mapping filename -> list of polygons,
        where each polygon is list of (x, y) tuples.
    """
    result = {}
    total_instances = 0

    for key, entry in via_data.items():
        filename = entry.get("filename", "")
        if not filename:
            continue

        regions = entry.get("regions", {})
        if isinstance(regions, dict):
            region_list = list(regions.values())
        else:
            region_list = regions

        polygons = []
        for region in region_list:
            sa = region.get("shape_attributes", {})
            if sa.get("name") != "polygon":
                continue

            xs = sa.get("all_points_x", [])
            ys = sa.get("all_points_y", [])
            if len(xs) < 3 or len(ys) < 3:
                continue

            polygon = list(zip(xs, ys))
            polygons.append(polygon)

        if polygons:
            result[filename] = polygons
            total_instances += len(polygons)

    logger.info(f"Extracted {total_instances} instances from {len(result)} images")
    return result


def polygon_to_yoloseg(polygon: List[Tuple[float, float]],
                       img_w: int, img_h: int) -> str:
    """Convert a polygon to YOLO-seg format line.

    Args:
        polygon: List of (x, y) pixel coordinates.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        YOLO-seg format string: "class_id x1 y1 x2 y2 ..."
    """
    normalized = []
    for x, y in polygon:
        nx = max(0.0, min(1.0, x / img_w))
        ny = max(0.0, min(1.0, y / img_h))
        normalized.extend([nx, ny])

    coords_str = " ".join(f"{v:.6f}" for v in normalized)
    return f"{CLASS_ID} {coords_str}"


def convert_split(split: str, annotation_type: str = "instance") -> int:
    """Convert one split (train/val/test) to YOLO-seg format.

    Args:
        split: One of "train", "val", "test".
        annotation_type: "instance" for modal, "amodal" for amodal.

    Returns:
        Number of images processed.
    """
    json_name = f"via_region_data_{annotation_type}.json"
    json_path = DATA_ROOT / "gt_json" / split / json_name
    img_dir = DATA_ROOT / "images" / split

    if annotation_type == "amodal":
        label_subdir = "amodal_labels"
    else:
        label_subdir = "labels"

    out_label_dir = OUTPUT_ROOT / label_subdir / split
    out_img_dir = OUTPUT_ROOT / "images" / split
    out_depth_dir = OUTPUT_ROOT / "depth" / split

    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_depth_dir.mkdir(parents=True, exist_ok=True)

    via_data = load_via_json(json_path)
    polygons_per_image = extract_polygons(via_data)

    processed = 0
    skipped_no_img = 0
    skipped_no_depth = 0

    for filename, polygons in polygons_per_image.items():
        img_path = img_dir / filename
        if not img_path.exists():
            skipped_no_img += 1
            continue

        # Get actual image size
        stem = img_path.stem

        # Write YOLO-seg label
        label_path = out_label_dir / f"{stem}.txt"
        lines = []
        for polygon in polygons:
            line = polygon_to_yoloseg(polygon, IMG_WIDTH, IMG_HEIGHT)
            lines.append(line)

        with open(label_path, 'w') as f:
            f.write("\n".join(lines))

        # Symlink image (avoid copying large files)
        out_img_path = out_img_dir / filename
        if not out_img_path.exists():
            out_img_path.symlink_to(img_path.resolve())

        # Symlink depth map
        depth_name = f"{stem}.npy"
        depth_path = DATA_ROOT / "depth_maps" / depth_name
        out_depth_path = out_depth_dir / depth_name
        if depth_path.exists() and not out_depth_path.exists():
            out_depth_path.symlink_to(depth_path.resolve())
        elif not depth_path.exists():
            skipped_no_depth += 1

        processed += 1

    logger.info(
        f"[{split}/{annotation_type}] Processed: {processed}, "
        f"Skipped (no img): {skipped_no_img}, "
        f"Skipped (no depth): {skipped_no_depth}"
    )
    return processed


def create_dataset_yaml() -> None:
    """Create YOLO dataset.yaml configuration file."""
    yaml_content = f"""# GeoApple-Seg Dataset Configuration
# AmodalAppleSize_RGB-D (Fuji subset)

path: {OUTPUT_ROOT.resolve()}
train: images/train
val: images/val
test: images/test

# Depth maps (custom field for our model)
depth_train: depth/train
depth_val: depth/val
depth_test: depth/test

# Classes
nc: 1
names:
  0: apple
"""
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    logger.info(f"Created dataset.yaml at {yaml_path}")


def print_statistics() -> None:
    """Print dataset statistics after conversion."""
    for split in ["train", "val", "test"]:
        img_dir = OUTPUT_ROOT / "images" / split
        label_dir = OUTPUT_ROOT / "labels" / split
        depth_dir = OUTPUT_ROOT / "depth" / split
        amodal_dir = OUTPUT_ROOT / "amodal_labels" / split

        n_img = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        n_label = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        n_depth = len(list(depth_dir.glob("*.npy"))) if depth_dir.exists() else 0
        n_amodal = len(list(amodal_dir.glob("*.txt"))) if amodal_dir.exists() else 0

        # Count total instances
        total_instances = 0
        if label_dir.exists():
            for lf in label_dir.glob("*.txt"):
                with open(lf) as f:
                    total_instances += len(f.readlines())

        logger.info(
            f"[{split}] Images: {n_img}, Labels: {n_label}, "
            f"Depth: {n_depth}, Amodal: {n_amodal}, "
            f"Instances: {total_instances}"
        )


def main() -> None:
    """Run full conversion pipeline."""
    logger.info("=" * 60)
    logger.info("Converting AmodalAppleSize_RGB-D to YOLO-seg format")
    logger.info("=" * 60)

    # Clean output directory
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    # Convert modal (instance) annotations
    for split in ["train", "val", "test"]:
        convert_split(split, "instance")

    # Convert amodal annotations
    for split in ["train", "val", "test"]:
        convert_split(split, "amodal")

    # Create dataset.yaml
    create_dataset_yaml()

    # Print statistics
    logger.info("=" * 60)
    logger.info("Conversion complete. Statistics:")
    print_statistics()


if __name__ == "__main__":
    main()
