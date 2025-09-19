"""
YOLO adapter for offline dataset export with augmentation.
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from src.datasets.canonical import Sample, DatasetSplits
from src.datasets.augment import build_augmentation_pipelines, apply_augmentation, build_albu_pipeline
from src.config import Config, YOLOConfig
from copy import deepcopy


def export_yolo_dataset(samples: DatasetSplits,
                       config: Config,
                       output_dir: str = "tmp",
                       n_augmentations: int = 1,
                       include_original: bool = True) -> str:
    """
    Export canonical dataset to YOLO format with offline augmentation.
    
    Args:
        samples: Canonical dataset splits
        config: Configuration object
        output_dir: Output directory for YOLO dataset
        n_augmentations: Number of augmented versions per image
        include_original: Whether to include the original image (saved as <base>.jpg)
        
    Returns:
        Path to dataset YAML file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create directory structure
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Build augmentation pipelines
    # For YOLO export we want photometric-only augmentation offline for the training split.
    # Do NOT modify the global config; make a copy and zero-out geometric params.
    aug_photo_only = deepcopy(config.aug)
    aug_photo_only.degrees = 0.0
    aug_photo_only.translate = 0.0
    aug_photo_only.scale_min = 1.0
    aug_photo_only.scale_max = 1.0
    aug_photo_only.shear = 0.0
    aug_photo_only.fliplr_p = 0.0
    aug_photo_only.flipud_p = 0.0

    # Build pipelines: photometric-only for train, standard val (no aug)
    train_pipeline = build_albu_pipeline(
        aug_photo_only,
        is_training=True,
        imgsz=config.yolo.imgsz,
        bbox_format='pascal_voc',
        bbox_label_fields=['labels'],
        bbox_min_visibility=0.1,
        bbox_min_area=1,
    )
    val_pipeline = build_albu_pipeline(
        config.aug,
        is_training=False,
        imgsz=config.yolo.imgsz,
        bbox_format='pascal_voc',
        bbox_label_fields=['labels'],
    )
    
    # Export splits
    for split_name in ['train', 'val']:
        split_samples = getattr(samples, split_name)
        if not split_samples:
            continue
        
        print(f"Exporting {split_name} split ({len(split_samples)} images)...")
        
        # Choose pipeline based on split
        pipeline = train_pipeline if split_name == 'train' else val_pipeline
        
        # Export samples
        for sample in split_samples:
            _export_sample_to_yolo(
                sample=sample,
                pipeline=pipeline,
                images_dir=images_dir / split_name,
                labels_dir=labels_dir / split_name,
                n_augmentations=n_augmentations if split_name == 'train' else 1,
                include_original=include_original if split_name == 'train' else True
            )
    
    # Create dataset YAML
    dataset_yaml_path = _create_dataset_yaml(
        output_path=output_path,
        config=config,
        samples=samples
    )
    
    # Create split files
    _create_split_files(output_path)
    
    print(f"✅ YOLO dataset exported to {output_path}")
    return str(dataset_yaml_path)


def _export_sample_to_yolo(sample: Sample,
                          pipeline: callable,
                          images_dir: Path,
                          labels_dir: Path,
                          n_augmentations: int = 1,
                          include_original: bool = True) -> None:
    """
    Export a single sample to YOLO format with augmentation.
    
    Args:
        sample: Canonical sample
        pipeline: Augmentation pipeline
        images_dir: Directory for images
        labels_dir: Directory for labels
        n_augmentations: Number of augmented versions to create
        include_original: Whether to include the original image (saved as <base>.jpg)
    """
    # Load original image
    try:
        image = Image.open(sample.image_path).convert('RGB')
        image_np = np.array(image)
    except Exception as e:
        print(f"⚠️  Warning: Could not load image {sample.image_path}: {e}")
        return
    
    base_name = Path(sample.image_path).stem
    
    # Save original image if requested
    if include_original:
        # Save original image as <base>.jpg
        original_image_path = images_dir / f"{base_name}.jpg"
        Image.fromarray(image_np).save(original_image_path, 'JPEG')
        
        # Save original labels
        original_label_path = labels_dir / f"{base_name}.txt"
        _save_yolo_labels(
            boxes=sample.boxes,
            labels=sample.labels,
            label_path=original_label_path,
            image_size=image_np.shape[:2]
        )
    
    # Create augmented versions
    num_augmentations = n_augmentations - (1 if include_original else 0)
    for aug_idx in range(num_augmentations):
        # Apply augmentation
        try:
            result = apply_augmentation(
                pipeline=pipeline,
                image=image_np,
                boxes=sample.boxes,
                labels=sample.labels,
                image_path=sample.image_path
            )
        except Exception as e:
            print(f"⚠️  Warning: Augmentation failed for {sample.image_path}: {e}")
            # Fallback to original
            result = type('Result', (), {
                'image': image_np,
                'boxes': sample.boxes,
                'labels': sample.labels
            })()
        
        # Convert image back to PIL
        if hasattr(result.image, 'numpy'):
            result.image = result.image.numpy()
        if hasattr(result.image, 'permute'):
            result.image = result.image.permute(1, 2, 0).numpy()
        if hasattr(result.image, 'cpu'):
            result.image = result.image.cpu().numpy()
        
        # Convert from CHW to HWC format
        if len(result.image.shape) == 3 and result.image.shape[0] == 3:
            result.image = np.transpose(result.image, (1, 2, 0))
        
        # Ensure uint8 format
        if result.image.dtype != np.uint8:
            if result.image.max() <= 1.0:
                result.image = (result.image * 255).astype(np.uint8)
            else:
                result.image = result.image.astype(np.uint8)
        
        # Create filename for augmented version
        image_name = f"{base_name}_aug{aug_idx}"
        
        # Save image
        image_path = images_dir / f"{image_name}.jpg"
        Image.fromarray(result.image).save(image_path, 'JPEG')
        
        # Save labels in YOLO format
        label_path = labels_dir / f"{image_name}.txt"
        _save_yolo_labels(
            boxes=result.boxes,
            labels=result.labels,
            image_size=result.image.shape[:2],  # (height, width)
            label_path=label_path
        )


def _save_yolo_labels(boxes: List[List[float]],
                     labels: List[int],
                     image_size: tuple,
                     label_path: Path) -> None:
    """
    Save bounding boxes in YOLO format.
    
    Args:
        boxes: Bounding boxes [[x1, y1, x2, y2], ...]
        labels: Class labels
        image_size: Image size (height, width)
        label_path: Path to save labels
    """
    height, width = image_size
    
    with open(label_path, 'w') as f:
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            
            # Write label line
            f.write(f"{label} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


def _create_dataset_yaml(output_path: Path,
                        config: Config,
                        samples: DatasetSplits) -> Path:
    """
    Create YOLO dataset YAML file.
    
    Args:
        output_path: Output directory
        config: Configuration object
        samples: Dataset splits
        
    Returns:
        Path to dataset YAML file
    """
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(config.data.class_names),
        'names': config.data.class_names
    }
    
    # Add test split if available
    if samples.test:
        dataset_config['test'] = 'images/test'
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        import yaml
        yaml.dump(dataset_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created dataset YAML: {yaml_path}")
    return yaml_path


def _create_split_files(output_path: Path) -> None:
    """
    Create split files for YOLO training by scanning exported images.
    
    Args:
        output_path: Output directory
    """
    for split_name in ['train', 'val']:
        images_dir = output_path / "images" / split_name
        
        if not images_dir.exists():
            continue
        
        # Scan for all *.jpg files in the split directory
        image_files = sorted(images_dir.glob("*.jpg"))
        
        if not image_files:
            continue
        
        # Create relative paths to images
        split_paths = []
        for image_file in image_files:
            relative_path = f"images/{split_name}/{image_file.name}"
            split_paths.append(relative_path)
        
        # Write split file
        split_file = output_path / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for path in split_paths:
                f.write(f"{path}\n")
        
        print(f"✅ Created split file: {split_file} ({len(split_paths)} images)")


def cleanup_yolo_dataset(output_dir: str) -> None:
    """
    Clean up temporary YOLO dataset.
    
    Args:
        output_dir: Directory to clean up
    """
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"✅ Cleaned up YOLO dataset: {output_path}")


def get_yolo_dataset_info(output_dir: str) -> Dict[str, Any]:
    """
    Get information about exported YOLO dataset.
    
    Args:
        output_dir: YOLO dataset directory
        
    Returns:
        Dictionary with dataset information
    """
    output_path = Path(output_dir)
    
    info = {
        'total_images': 0,
        'total_labels': 0,
        'splits': {}
    }
    
    for split_name in ['train', 'val', 'test']:
        images_dir = output_path / "images" / split_name
        labels_dir = output_path / "labels" / split_name
        
        if not images_dir.exists():
            continue
        
        # Count images
        image_files = list(images_dir.glob("*.jpg"))
        n_images = len(image_files)
        
        # Count labels
        n_labels = 0
        for image_file in image_files:
            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    n_labels += len([line for line in f if line.strip()])
        
        info['splits'][split_name] = {
            'images': n_images,
            'labels': n_labels,
            'labels_per_image': n_labels / n_images if n_images > 0 else 0
        }
        
        info['total_images'] += n_images
        info['total_labels'] += n_labels
    
    return info
