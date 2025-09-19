"""
Canonical dataset format and loading utilities.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image


@dataclass
class Sample:
    """Canonical sample format."""
    image_path: str
    width: int
    height: int
    boxes: List[List[float]]  # [[x1, y1, x2, y2], ...]
    labels: List[int]


@dataclass
class DatasetSplits:
    """Dataset splits."""
    train: List[Sample]
    val: List[Sample]
    test: List[Sample] = None


def load_canonical_dataset(labels_json: str, splits_dir: str) -> DatasetSplits:
    """
    Load canonical dataset from labels JSON and split files.
    
    Args:
        labels_json: Path to labels_internal.json
        splits_dir: Path to directory containing train.txt, val.txt, test.txt
    
    Returns:
        DatasetSplits containing train/val/test samples
    """
    # Load labels data
    with open(labels_json, 'r') as f:
        labels_data = json.load(f)
    
    # Create image path to label mapping
    image_to_labels = {}
    for record in labels_data:
        image_path = record['image_path']
        image_to_labels[image_path] = record
    
    # Load splits
    splits_dir = Path(splits_dir)
    train_paths = _load_split_file(splits_dir / "train.txt")
    val_paths = _load_split_file(splits_dir / "val.txt")
    test_paths = _load_split_file(splits_dir / "test.txt") if (splits_dir / "test.txt").exists() else None
    
    # Convert to canonical samples
    train_samples = _paths_to_samples(train_paths, image_to_labels)
    val_samples = _paths_to_samples(val_paths, image_to_labels)
    test_samples = _paths_to_samples(test_paths, image_to_labels) if test_paths else None
    
    return DatasetSplits(
        train=train_samples,
        val=val_samples,
        test=test_samples
    )


def _load_split_file(split_path: Path) -> List[str]:
    """Load image paths from split file."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    with open(split_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    return paths


def _paths_to_samples(image_paths: List[str], image_to_labels: Dict[str, Any]) -> List[Sample]:
    """Convert image paths to canonical samples."""
    samples = []
    
    for image_path in image_paths:
        record = image_to_labels.get(image_path)

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            # Skip unreadable images
            continue

        boxes = []
        labels = []

        if record is not None:
            for bbox_info in record.get('bboxes', []):
                xyxy = bbox_info['xyxy']
                boxes.append(xyxy)
                labels.append(0)  # Single class: probe
        # else: negatives or unlabeled images → keep empty boxes/labels

        sample = Sample(
            image_path=image_path,
            width=width,
            height=height,
            boxes=boxes,
            labels=labels
        )

        samples.append(sample)
    
    return samples


def get_variant_from_path(image_path: str) -> int:
    """
    Extract variant from image path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Variant number (0 for bottom, 2 for top, None for unknown)
    """
    if image_path.endswith('_0.jpg'):
        return 0  # bottom
    elif image_path.endswith('_2.jpg'):
        return 2  # top
    else:
        return None  # unknown


def filter_samples_by_variant(samples: List[Sample], variant: int) -> List[Sample]:
    """Filter samples by variant."""
    return [s for s in samples if get_variant_from_path(s.image_path) == variant]


def get_dataset_stats(samples: List[Sample]) -> Dict[str, Any]:
    """Get dataset statistics."""
    if not samples:
        return {}
    
    total_images = len(samples)
    total_boxes = sum(len(s.boxes) for s in samples)
    
    # Box statistics
    box_areas = []
    box_heights = []
    box_widths = []
    aspect_ratios = []
    
    for sample in samples:
        for box in sample.boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            box_areas.append(area)
            box_heights.append(height)
            box_widths.append(width)
            aspect_ratios.append(width / height if height > 0 else 0)
    
    # Variant distribution
    variant_counts = {}
    for sample in samples:
        variant = get_variant_from_path(sample.image_path)
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
    
    return {
        'total_images': total_images,
        'total_boxes': total_boxes,
        'boxes_per_image': total_boxes / total_images if total_images > 0 else 0,
        'box_area_mean': sum(box_areas) / len(box_areas) if box_areas else 0,
        'box_area_std': (sum((a - sum(box_areas)/len(box_areas))**2 for a in box_areas) / len(box_areas))**0.5 if box_areas else 0,
        'box_height_mean': sum(box_heights) / len(box_heights) if box_heights else 0,
        'box_width_mean': sum(box_widths) / len(box_widths) if box_widths else 0,
        'aspect_ratio_mean': sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 0,
        'variant_distribution': variant_counts
    }
