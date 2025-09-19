"""
RCNN adapter for PyTorch Dataset with on-the-fly augmentation.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import torchvision.transforms as transforms

from src.datasets.canonical import Sample
from src.datasets.augment import apply_augmentation


class RCNNDataset(Dataset):
    """
    PyTorch Dataset for Faster R-CNN with on-the-fly augmentation.
    """
    
    def __init__(self, 
                 samples: List[Sample],
                 augmentation_pipeline: Optional[Callable] = None,
                 image_size: int = 800):
        """
        Initialize RCNN Dataset.
        
        Args:
            samples: List of canonical samples
            augmentation_pipeline: Augmentation pipeline (None for no augmentation)
            image_size: Target image size for resizing
        """
        self.samples = samples
        self.augmentation_pipeline = augmentation_pipeline
        self.image_size = image_size
        
        # Basic transforms: torchvision detection expects [0,1] float tensors, no normalization
        self.transforms = transforms.ToTensor()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'image' and 'target' keys
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample.image_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply augmentation if available
        if self.augmentation_pipeline and len(sample.boxes) > 0:
            result = apply_augmentation(
                pipeline=self.augmentation_pipeline,
                image=image_np,
                boxes=sample.boxes,
                labels=sample.labels,
                image_path=sample.image_path
            )
            
            # Convert back to PIL Image
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
            
            # Update image and boxes
            image = Image.fromarray(result.image)
            boxes = result.boxes
            labels = result.labels
        else:
            boxes = sample.boxes
            labels = sample.labels
        
        # Resize image if needed and scale boxes accordingly
        orig_w, orig_h = image.size
        if image.size != (self.image_size, self.image_size):
            scale_x = float(self.image_size) / float(orig_w)
            scale_y = float(self.image_size) / float(orig_h)
            # Scale boxes to the resized image coordinate space
            if boxes and len(boxes) > 0:
                scaled_boxes = []
                for x1, y1, x2, y2 in boxes:
                    scaled_boxes.append([
                        float(x1) * scale_x,
                        float(y1) * scale_y,
                        float(x2) * scale_x,
                        float(y2) * scale_y,
                    ])
                boxes = scaled_boxes
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_tensor = self.transforms(image)
        
        # Prepare target for Faster R-CNN
        target = self._prepare_target(boxes, labels, image_tensor.shape[1:])
        
        return {
            'image': image_tensor,
            'target': target
        }
    
    def _prepare_target(self, 
                       boxes: List[List[float]], 
                       labels: List[int], 
                       image_shape: tuple) -> Dict[str, torch.Tensor]:
        """
        Prepare target dictionary for Faster R-CNN.
        
        Args:
            boxes: Bounding boxes [[x1, y1, x2, y2], ...]
            labels: Class labels
            image_shape: Image shape (C, H, W)
            
        Returns:
            Target dictionary
        """
        if len(boxes) == 0:
            # Empty target
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor(0, dtype=torch.int64),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
        
        # Convert boxes to tensor
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        # Torchvision detection models expect class labels in [1..num_classes], 0 reserved for background
        labels_tensor = torch.tensor(labels, dtype=torch.int64) + 1
        
        # Calculate areas
        areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        
        # Create target dictionary
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor(0, dtype=torch.int64),  # Single image per batch
            'area': areas,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return target
    
    def _create_empty_sample(self) -> Dict[str, Any]:
        """Create an empty sample for error cases."""
        return {
            'image': torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32),
            'target': {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor(0, dtype=torch.int64),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
        }


def rcnn_collate_fn(batch):
    """Custom collate function for Faster R-CNN."""
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
    
    return images, targets


def create_rcnn_data_loaders(samples: Dict[str, List[Sample]],
                           augmentation_pipelines: Dict[str, Optional[Callable]],
                           config,
                           num_workers: int = 4) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for Faster R-CNN training.
    
    Args:
        samples: Dictionary with 'train' and 'val' sample lists
        augmentation_pipelines: Dictionary with 'train' and 'val' augmentation pipelines
        config: RCNN configuration
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with 'train' and 'val' DataLoaders
    """
    # Create datasets
    train_dataset = RCNNDataset(
        samples=samples['train'],
        augmentation_pipeline=augmentation_pipelines['train'],
        image_size=config.imgsz
    )
    
    val_dataset = RCNNDataset(
        samples=samples['val'],
        augmentation_pipeline=augmentation_pipelines['val'],
        image_size=config.imgsz
    )
    
    # Create data loaders (disable pin_memory for MPS)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rcnn_collate_fn,
        pin_memory=False  # Disabled for MPS compatibility
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rcnn_collate_fn,
        pin_memory=False  # Disabled for MPS compatibility
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


def get_dataset_stats(dataset: RCNNDataset) -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        dataset: RCNNDataset instance
        
    Returns:
        Dictionary with dataset statistics
    """
    total_images = len(dataset)
    total_boxes = sum(len(sample.boxes) for sample in dataset.samples)
    
    # Box statistics
    box_areas = []
    box_heights = []
    box_widths = []
    
    for sample in dataset.samples:
        for box in sample.boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            box_areas.append(area)
            box_heights.append(height)
            box_widths.append(width)
    
    return {
        'total_images': total_images,
        'total_boxes': total_boxes,
        'boxes_per_image': total_boxes / total_images if total_images > 0 else 0,
        'box_area_mean': sum(box_areas) / len(box_areas) if box_areas else 0,
        'box_height_mean': sum(box_heights) / len(box_heights) if box_heights else 0,
        'box_width_mean': sum(box_widths) / len(box_widths) if box_widths else 0,
        'image_size': dataset.image_size
    }
