#!/usr/bin/env python3
"""
One-time utility to create Leave-One-Mission-Out (LOMO) dataset splits.

Usage:
  python data/prepare_data_lomo_splits.py --labels-json data/labels_internal.json --output data/splits_lomo
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src is importable when executed from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cv.lomo import create_lomo_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LOMO dataset splits (one-time)")
    parser.add_argument("--labels-json", type=str, required=True, help="Path to labels_internal.json")
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "data" / "splits_lomo"), help="Output directory for splits")
    args = parser.parse_args()

    labels_path = Path(args.labels_json)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = create_lomo_splits(labels_path, out_dir)

    if not folds:
        print(f"\n❌ No folds created under '{out_dir}'. Please verify labels JSON.")
        sys.exit(1)

    print(f"\n✅ Created {len(folds)} LOMO folds under '{out_dir}'.")
    for f in folds:
        print(f" - fold_{f.fold_index} ({f.mission_id}): train={f.train_count}, val={f.val_count}")


if __name__ == "__main__":
    main()


