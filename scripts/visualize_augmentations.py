#!/usr/bin/env python3
"""
Augmentation Visualization Tool

This script creates a visual comparison of original images and their augmented versions
to help debug and tune augmentation parameters.
"""

import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import load_config, AugmentationConfig, DataConfig
from datasets.canonical import load_canonical_dataset, Sample
from datasets.augment import build_augmentation_pipelines, apply_augmentation, AugmentationResult


def draw_boxes(ax, image, boxes, labels, color='red'):
    """Draw bounding boxes on the image."""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Class {labels[i]}", color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))


def visualize_augmentations(config_path: str, num_images: int = 3, num_augmentations: int = 3):
    """
    Visualize augmentation effects on sample images.
    
    Args:
        config_path: Path to configuration YAML file
        num_images: Number of random images to display
        num_augmentations: Number of augmented versions per image
    """
    print("Current Augmentation Configuration:")
    print("=" * 40)
    cfg = load_config(config_path)
    aug_cfg: AugmentationConfig = cfg.aug
    data_cfg: DataConfig = cfg.data

    print("Geometry:")
    print(f"  Rotation: ±{aug_cfg.degrees}°")
    print(f"  Translation: ±{aug_cfg.translate*100}%")
    print(f"  Scale: {aug_cfg.scale_min:.2f} - {aug_cfg.scale_max:.2f}")
    print(f"  Shear: ±{aug_cfg.shear}°")
    print(f"  Horizontal Flip: {aug_cfg.fliplr_p*100}%")
    print(f"  Vertical Flip: {aug_cfg.flipud_p*100}%")
    print("\nPhotometric:")
    print(f"  Brightness: ±{aug_cfg.brightness*100}%")
    print(f"  Contrast: ±{aug_cfg.contrast*100}%")
    print(f"  Gamma: {aug_cfg.gamma[0]:.2f} - {aug_cfg.gamma[1]:.2f}")
    print(f"  Motion Blur: {aug_cfg.motion_blur_p*100}%")
    print(f"  Noise: {aug_cfg.noise_p*100}%")
    print(f"  CLAHE: {aug_cfg.clahe_p*100}%")
    print(f"  Glare: {aug_cfg.glare_p*100}%")
    print(f"  Vignette: {aug_cfg.vignette_p*100}%")
    print("\nPadding:")
    print(f"  Pixels: {aug_cfg.pixels}")
    print(f"  Reflect: {aug_cfg.use_reflect}")
    print("\nGenerating visualization...")

    # Load dataset
    samples = load_canonical_dataset(data_cfg.labels_json, data_cfg.splits_dir)
    all_samples = samples.train + samples.val + (samples.test if samples.test else [])
    if not all_samples:
        print("No samples found in the dataset.")
        return

    # Select random images
    selected_samples = random.sample(all_samples, min(num_images, len(all_samples)))

    # Build augmentation pipelines
    train_pipeline, _ = build_augmentation_pipelines(aug_cfg, imgsz=cfg.yolo.imgsz)

    # Create figure
    fig, axes = plt.subplots(num_images, num_augmentations + 1, 
                            figsize=(4 * (num_augmentations + 1), 4 * num_images))
    
    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(selected_samples):
        print(f"Processing image {i+1}/{len(selected_samples)}: {Path(sample.image_path).name}")
        
        # Original image
        image_orig = Image.open(sample.image_path).convert('RGB')
        image_np_orig = np.array(image_orig)

        # Plot original
        ax = axes[i, 0]
        ax.imshow(image_np_orig)
        draw_boxes(ax, image_np_orig, sample.boxes, sample.labels, color='red')
        ax.set_title(f"Original ({len(sample.boxes)} boxes)", fontsize=10)
        ax.axis('off')

        # Plot augmented versions
        for j in range(num_augmentations):
            result = apply_augmentation(
                pipeline=train_pipeline,
                image=image_np_orig,
                boxes=sample.boxes,
                labels=sample.labels,
                image_path=sample.image_path
            )

            # Convert image back to HWC for plotting if it's a tensor
            if isinstance(result.image, torch.Tensor):
                img_to_plot = result.image.permute(1, 2, 0).cpu().numpy()
                if img_to_plot.max() <= 1.0:
                    img_to_plot = (img_to_plot * 255).astype(np.uint8)
            else:
                img_to_plot = result.image

            ax = axes[i, j + 1]
            ax.imshow(img_to_plot)
            draw_boxes(ax, img_to_plot, result.boxes, result.labels, color='blue')
            ax.set_title(f"Augmented {j+1} ({len(result.boxes)} boxes)", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    
    # Save the visualization
    output_filename = "augmentation_visualization.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_filename}")
    
    # Also save a high-resolution version
    plt.savefig("augmentation_visualization_hd.png", dpi=300, bbox_inches='tight')
    print(f"High-resolution version saved to: augmentation_visualization_hd.png")
    
    # Display the plot
    print("Displaying visualization...")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize augmentation pipelines")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration YAML file")
    parser.add_argument("--images", type=int, default=3, 
                       help="Number of random images to display")
    parser.add_argument("--augmentations", type=int, default=3, 
                       help="Number of augmented versions per image")
    args = parser.parse_args()

    visualize_augmentations(args.config, args.images, args.augmentations)
