#!/usr/bin/env python3
"""
Simple training script for the new clean architecture.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import train_model
from dataclasses import asdict
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train probe detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional alias for --config"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs",
        help="Base directory for TensorBoard logs and run artifacts"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name to create subfolder under log_dir"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )
    parser.add_argument(
        "--lomo-cv",
        action="store_true",
        help="Run 5-fold Leave-One-Mission-Out cross-validation"
    )
    parser.add_argument(
        "--splits-root",
        type=str,
        default="data/splits_lomo_test_val_only",
        help="Directory to write/read LOMO split files"
    )
    
    args = parser.parse_args()
    cfg_path = args.config_path or args.config
    
    try:
        # Prepare run directory and writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"run_{timestamp}"
        run_dir = Path(args.log_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Load config (to persist JSON and pass through main)
        from src.config import load_config
        config_obj = load_config(cfg_path)

        # Point trainers' project dir to run_dir
        config_obj.project = str(run_dir)

        # Persist config as JSON for reproducibility
        with open(run_dir / "config.json", "w") as f:
            json.dump(asdict(config_obj), f, indent=2)

        if args.lomo_cv:
            # Run LOMO CV
            from src.cv.runner import run_lomo_cv
            from src.cv.lomo import load_lomo_folds_from_dir

            # Ensure splits exist; if not, guide user to create once
            existing_folds = load_lomo_folds_from_dir(args.splits_root)
            if not existing_folds:
                print(
                    "\n❌ No LOMO splits found under '{}'".format(args.splits_root)
                )
                print(
                    "Please create them once using:\n  python data/prepare_data_lomo_splits.py --labels-json {} --output {}".format(
                        config_obj.data.labels_json, args.splits_root
                    )
                )
                sys.exit(2)
            summary = run_lomo_cv(
                config_path=cfg_path,
                labels_json=config_obj.data.labels_json,
                splits_root=args.splits_root,
                base_log_dir=str(run_dir),
                run_timestamp=timestamp,
            )
            with open(run_dir / "lomo_cv_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print("\n✅ LOMO cross-validation completed. Summary saved to lomo_cv_summary.json")
        else:
            # TensorBoard writer under run_dir/tensorboard
            tb_dir = run_dir / "tensorboard"
            writer = SummaryWriter(str(tb_dir))

            # Kick off training
            weights_path = train_model(cfg_path, args.dry_run, writer=writer, run_dir=str(run_dir))

            # Close writer
            writer.close()
            if weights_path:
                print(f"\n✅ Training completed successfully!")
                print(f"Model weights: {weights_path}")
            else:
                print(f"\n✅ Configuration validated successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
