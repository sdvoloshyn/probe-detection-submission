"""
Main training facade for probe detection.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import load_config
from trainers.yolo_trainer import YOLOTrainer
from trainers.rcnn_trainer import RCNNTrainer


def train_model(config_path: str, dry_run: bool = False, writer=None, run_dir: str | None = None) -> str:
    """
    Train a model based on configuration.
    
    Args:
        config_path: Path to configuration YAML file
        dry_run: If True, only validate configuration without training
        
    Returns:
        Path to trained model weights
    """
    print("🚀 PROBE DETECTION TRAINING")
    print("=" * 50)
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Print configuration summary
    print(f"Model: {config.model}")
    print(f"Device: {config.device}")
    print(f"Classes: {config.data.class_names}")
    print(f"Images: {config.data.images_root}")
    print(f"Labels: {config.data.labels_json}")
    print(f"Splits: {config.data.splits_dir}")
    
    if dry_run:
        print("🔍 DRY RUN - No training will be performed")
        return ""
    
    # Select trainer based on model type
    if str(config.model).startswith("yolo"):
        print(f"Using YOLO trainer for {config.model}")
        trainer = YOLOTrainer(config, writer=writer, run_dir=run_dir)
    elif config.model == "faster_rcnn":
        print("Using Faster R-CNN trainer")
        trainer = RCNNTrainer(config, run_dir=run_dir)
        # Attach optional TensorBoard writer if trainer supports it
        if hasattr(trainer, 'writer') and writer is not None:
            trainer.writer = writer
    else:
        raise ValueError(f"Unknown model type: {config.model}")
    
    # Train model
    weights_path = trainer.train()
    
    # Print model info
    model_info = trainer.get_model_info()
    print(f"\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    return weights_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train probe detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )
    
    args = parser.parse_args()
    
    weights_path = train_model(args.config, args.dry_run)
    if weights_path:
        print(f"\n✅ Training completed successfully!")
        print(f"Model weights: {weights_path}")
    else:
        print(f"\n✅ Configuration validated successfully!")


if __name__ == "__main__":
    main()
