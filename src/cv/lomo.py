"""
Leave-One-Mission-Out (LOMO) cross-validation split generation.

This utility groups images by their mission identifier parsed from the
image filename and creates one fold per mission, where that mission is used
as the validation set and all remaining missions form the training set.

Outputs a directory per fold containing train.txt and val.txt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class LomoFold:
    fold_index: int
    mission_id: str
    fold_dir: Path
    train_count: int
    val_count: int


def _parse_mission_id(image_path: str) -> str:
    """Extract mission id from image filename. Assumes format 'MISSION_...jpg'."""
    name = Path(image_path).stem
    # Mission id is the first token before the first underscore
    return name.split("_")[0]


def _load_labels(labels_json_path: Path) -> List[dict]:
    with open(labels_json_path, "r") as f:
        return json.load(f)


def create_lomo_splits(labels_json: str | Path, output_root: str | Path) -> List[LomoFold]:
    """
    Create Leave-One-Mission-Out splits and write train/val files per fold.

    Args:
        labels_json: Path to labels_internal.json
        output_root: Root directory to create fold_<idx>_<mission>/ with split files

    Returns:
        List of LomoFold metadata in deterministic order of missions.
    """
    labels_json_path = Path(labels_json)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    records = _load_labels(labels_json_path)

    # Group image paths by mission id
    mission_to_images: Dict[str, List[str]] = {}
    for rec in records:
        img_path = rec["image_path"]
        mission_id = _parse_mission_id(img_path)
        mission_to_images.setdefault(mission_id, []).append(img_path)

    # Build folds: one per mission
    mission_ids = sorted(mission_to_images.keys())
    folds: List[LomoFold] = []

    # Create a global list of all images for quick complement selection
    all_images = [rec["image_path"] for rec in records]

    for idx, mission_id in enumerate(mission_ids, start=1):
        val_images = mission_to_images[mission_id]
        val_set = set(val_images)
        train_images = [p for p in all_images if p not in val_set]

        # Write split files
        fold_dir = output_root_path / f"fold_{idx}_{mission_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_txt = fold_dir / "train.txt"
        val_txt = fold_dir / "val.txt"

        with open(train_txt, "w") as f:
            for p in train_images:
                f.write(f"{p}\n")

        with open(val_txt, "w") as f:
            for p in val_images:
                f.write(f"{p}\n")

        folds.append(
            LomoFold(
                fold_index=idx,
                mission_id=mission_id,
                fold_dir=fold_dir,
                train_count=len(train_images),
                val_count=len(val_images),
            )
        )

    return folds


def load_lomo_folds_from_dir(splits_root: str | Path) -> List[LomoFold]:
    """
    Load existing LOMO folds from a directory previously created by
    create_lomo_splits().

    Each fold is expected to live under a subdirectory named
    'fold_<idx>_<mission_id>' and contain 'train.txt' and 'val.txt'.

    Args:
        splits_root: Directory containing the fold_* subdirectories

    Returns:
        List of LomoFold metadata for all valid folds found, sorted by index.
    """
    root = Path(splits_root)
    if not root.exists() or not root.is_dir():
        return []

    folds: List[LomoFold] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("fold_"):
            continue
        # Expect pattern fold_<idx>_<mission>
        remainder = name[len("fold_"):]
        parts = remainder.split("_", 1)
        if len(parts) != 2:
            continue
        idx_str, mission_id = parts[0], parts[1]
        try:
            idx = int(idx_str)
        except ValueError:
            continue

        train_txt = child / "train.txt"
        val_txt = child / "val.txt"
        if not (train_txt.exists() and val_txt.exists()):
            continue

        # Count lines for metadata; robust to trailing newlines
        def _count_lines(p: Path) -> int:
            try:
                with open(p, "r") as f:
                    return sum(1 for _ in f if _.strip())
            except Exception:
                return 0

        train_count = _count_lines(train_txt)
        val_count = _count_lines(val_txt)

        folds.append(
            LomoFold(
                fold_index=idx,
                mission_id=mission_id,
                fold_dir=child,
                train_count=train_count,
                val_count=val_count,
            )
        )

    # Sort by fold index to ensure deterministic order
    folds.sort(key=lambda f: f.fold_index)
    return folds


