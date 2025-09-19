"""
Augmentation module with Albumentations pipeline and EdgeSnapHook.
"""
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️  Albumentations not available. Install with: pip install albumentations")



@dataclass
class AugmentationResult:
    """Result of augmentation pipeline."""
    image: Any  # Can be np.ndarray or torch.Tensor depending on pipeline
    boxes: List[List[float]]
    labels: List[int]




def build_albu_pipeline(
    config: 'AugmentationConfig',
    is_training: bool = True,
    imgsz: int = 640,
    bbox_format: str = 'pascal_voc',
    bbox_label_fields: Optional[List[str]] = None,
    bbox_min_visibility: Optional[float] = None,
    bbox_min_area: Optional[float] = None,
) -> Callable:
    """
    Build Albumentations pipeline from configuration.
    
    Args:
        config: AugmentationConfig object
        is_training: Whether this is for training (applies augmentations)
        
    Returns:
        Albumentations pipeline function
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Albumentations not available. Install with: pip install albumentations")
    
    transforms_list = []
    
    if is_training:
        # Geometry augmentations (physics-aware)
        if config.degrees > 0:
            transforms_list.append(A.Rotate(limit=config.degrees, p=0.5))
        
        if config.translate > 0:
            transforms_list.append(A.Affine(
                translate_percent={'x': (-config.translate, config.translate), 
                                 'y': (-config.translate, config.translate)},
                p=0.5
            ))
        
        if config.scale_min != 1.0 or config.scale_max != 1.0:
            transforms_list.append(A.Affine(
                scale={'x': (config.scale_min, config.scale_max), 
                      'y': (config.scale_min, config.scale_max)},
                p=1.0
            ))
        
        if config.shear > 0:
            transforms_list.append(A.Affine(
                shear={'x': (-config.shear, config.shear), 
                      'y': (-config.shear, config.shear)},
                p=0.3
            ))
        
        if config.fliplr_p > 0:
            transforms_list.append(A.HorizontalFlip(p=config.fliplr_p))
        
        if config.flipud_p > 0:
            transforms_list.append(A.VerticalFlip(p=config.flipud_p))
        
        # Photometric augmentations (increased coverage)
        if config.brightness > 0 or config.contrast > 0:
            transforms_list.append(A.RandomBrightnessContrast(
                brightness_limit=config.brightness,
                contrast_limit=config.contrast,
                p=config.brightness_contrast_p
            ))
        
        if config.gamma and len(config.gamma) == 2:
            gamma_min, gamma_max = config.gamma
            if gamma_min != 1.0 or gamma_max != 1.0:
                # Convert to integers >= 1 for Albumentations
                gamma_min = max(1, int(gamma_min * 100))  # Convert 0.8 -> 80
                gamma_max = max(gamma_min, int(gamma_max * 100))  # Convert 1.2 -> 120
                if gamma_max > gamma_min:  # Only add if valid range
                    transforms_list.append(A.RandomGamma(
                        gamma_limit=(gamma_min, gamma_max),
                        p=config.gamma_p
                    ))
        
        if config.motion_blur_p > 0:
            transforms_list.append(A.MotionBlur(
                blur_limit=tuple(config.motion_blur_limit),
                p=config.motion_blur_p
            ))
        
        if config.noise_p > 0:
            std_rng = config.noise_std_range
            transforms_list.append(A.GaussNoise(std_range=(float(std_rng[0]), float(std_rng[1])), p=config.noise_p))
        
        if config.clahe_p > 0:
            transforms_list.append(A.CLAHE(
                clip_limit=config.clahe_clip_limit,
                tile_grid_size=tuple(config.clahe_tile_grid_size),
                p=config.clahe_p
            ))
        
        if config.glare_p > 0:
            transforms_list.append(A.RandomSunFlare(
                flare_roi=tuple(config.glare_flare_roi),
                src_radius=config.glare_src_radius, 
                src_color=tuple(config.glare_src_color),
                angle_range=tuple(config.glare_angle_range),
                num_flare_circles_range=tuple(config.glare_num_circles_range),
                method=config.glare_method,
                p=config.glare_p
            ))
        
        if config.vignette_p > 0:
            transforms_list.append(A.RandomShadow(
                shadow_roi=tuple(config.vignette_shadow_roi),
                num_shadows_limit=tuple(config.vignette_num_shadows_limit),
                shadow_dimension=config.vignette_shadow_dimension,
                shadow_intensity_range=tuple(config.vignette_intensity_range),
                p=config.vignette_p
            ))
    
    # Padding
    if config.pixels > 0:
        border_mode = cv2.BORDER_REFLECT_101 if config.use_reflect else cv2.BORDER_CONSTANT
        fill_value = 0 if not config.use_reflect else None
        
        pad_args = {
            'min_height': imgsz + config.pixels,
            'min_width': imgsz + config.pixels,
            'border_mode': border_mode,
            'p': 1.0
        }
        if not config.use_reflect:
            pad_args['fill'] = fill_value
        
        transforms_list.append(A.PadIfNeeded(**pad_args))
    
    # Convert to tensor
    transforms_list.append(ToTensorV2())
    
    # Create bbox params
    bbox_params_kwargs = {
        'format': bbox_format or 'pascal_voc',
        'label_fields': bbox_label_fields if bbox_label_fields is not None else ['labels'],
    }
    if bbox_min_visibility is not None:
        bbox_params_kwargs['min_visibility'] = float(bbox_min_visibility)
    if bbox_min_area is not None:
        bbox_params_kwargs['min_area'] = float(bbox_min_area)

    # Create the augmentation pipeline
    aug_pipeline = A.Compose(transforms_list, bbox_params=A.BboxParams(**bbox_params_kwargs))
    
    
    return aug_pipeline


def build_augmentation_pipelines(config: 'AugmentationConfig', imgsz: int = 640) -> tuple:
    """
    Build training and validation augmentation pipelines.
    
    Args:
        config: AugmentationConfig object
        imgsz: Image size for padding
        
    Returns:
        Tuple of (train_pipeline, val_pipeline)
    """
    # Build and return base pipelines directly
    train_pipeline = build_albu_pipeline(config, is_training=True, imgsz=imgsz)
    val_pipeline = build_albu_pipeline(config, is_training=False, imgsz=imgsz)
    
    return train_pipeline, val_pipeline


def apply_augmentation(pipeline: Callable, 
                      image: np.ndarray, 
                      boxes: List[List[float]], 
                      labels: List[int],
                      image_path: Optional[str] = None,
                      deterministic: bool = False,
                      original_labels: List[int] = None) -> AugmentationResult:
    """
    Apply augmentation pipeline to image and boxes.
    
    Args:
        pipeline: Augmentation pipeline (Albumentations)
        image: Input image (HWC, uint8)
        boxes: Bounding boxes [[x1, y1, x2, y2], ...]
        labels: Class labels
        image_path: Path to image (unused, kept for compatibility)
        
    Returns:
        AugmentationResult with processed image, boxes, and labels
    """
    # Apply Albumentations pipeline directly
    augmented = pipeline(
        image=image,
        bboxes=boxes,
        labels=labels
    )
    
    # Convert labels to integers and validate
    labels = [int(label) for label in augmented['labels']]
    
    # Fix and validate labels
    valid_labels = []
    for i, label in enumerate(labels):
        if not isinstance(label, int) or label < 0 or label >= 80:
            valid_labels.append(0)  # Fix invalid label
        else:
            valid_labels.append(int(label))
    
    # Fix and validate bounding boxes
    valid_boxes = []
    valid_labels_final = []
    
    for i, (box, label) in enumerate(zip(augmented['bboxes'], valid_labels)):
        x1, y1, x2, y2 = box
        
        # Skip invalid boxes
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            continue
            
        # Skip boxes that are too small
        width = x2 - x1
        height = y2 - y1
        if width < 5 or height < 5:
            continue
            
        valid_boxes.append([x1, y1, x2, y2])
        valid_labels_final.append(label)
    
    # If no valid boxes remain, return the original data (fallback)
    if not valid_boxes:
        return AugmentationResult(
            image=image,
            boxes=boxes,
            labels=original_labels if original_labels is not None else labels
        )
    
    return AugmentationResult(
        image=augmented['image'],
        boxes=valid_boxes,
        labels=valid_labels_final
    )
