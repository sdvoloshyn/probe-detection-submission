#!/usr/bin/env python3
"""
Data preparation script for probe detection dataset.

Converts probe_labels.json (COCO format) to labels_internal.json (internal format)
and creates train/val splits with the following constraints:
- Keep whole orbits together (don't split within orbit)
- Maintain equal fraction of variants (ending with _0 and _2)
- Target 80/20 train/val split
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import argparse


def parse_filename(filename: str) -> Tuple[str, str, str, str, str]:
    """Parse filename to extract mission, orbit, pass, time, variant."""
    # Pattern: MISSION_ORBIT_PASS_1flight_TIME_VARIANT.jpg
    pattern = r'([A-Z0-9]+)_(\d+)_(\d+)_1flight_(\d+)_([02])\.jpg'
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")
    return match.groups()


def load_coco_data(labels_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load COCO format data."""
    with open(labels_path, 'r') as f:
        data = json.load(f)
    return data['images'], data['annotations']


def group_by_orbit(images: List[Dict]) -> Dict[str, List[Dict]]:
    """Group images by orbit (mission_orbit_pass)."""
    orbit_groups = defaultdict(list)
    
    for img in images:
        try:
            mission, orbit, pass_, time, variant = parse_filename(img['file_name'])
            orbit_key = f"{mission}_{orbit}_{pass_}"
            orbit_groups[orbit_key].append(img)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    return dict(orbit_groups)


def analyze_variants(orbit_groups: Dict[str, List[Dict]]) -> Dict[str, Counter]:
    """Analyze variant distribution per orbit."""
    orbit_variants = {}
    for orbit, images in orbit_groups.items():
        variants = []
        for img in images:
            try:
                mission, orbit_num, pass_, time, variant = parse_filename(img['file_name'])
                variants.append(variant)
            except ValueError:
                continue
        orbit_variants[orbit] = Counter(variants)
    
    return orbit_variants


def create_balanced_splits(orbit_groups: Dict[str, List[Dict]], 
                          target_train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Create balanced train/val splits keeping orbits together and maintaining variant balance.
    
    Args:
        orbit_groups: Dictionary mapping orbit keys to image lists
        target_train_ratio: Target ratio for training set (default: 0.8)
    
    Returns:
        Tuple of (train_image_paths, val_image_paths)
    """
    # Analyze variant distribution per orbit
    orbit_variants = analyze_variants(orbit_groups)
    
    # Calculate total variants
    total_variants = Counter()
    for variants in orbit_variants.values():
        total_variants.update(variants)
    
    print(f"Total variant distribution: {dict(total_variants)}")
    target_train_variants = {
        variant: int(count * target_train_ratio) 
        for variant, count in total_variants.items()
    }
    print(f"Target train variant distribution: {target_train_variants}")
    
    # Sort orbits by total images (largest first for better balance)
    sorted_orbits = sorted(orbit_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    train_orbits = []
    val_orbits = []
    train_variants = Counter()
    
    for orbit_key, images in sorted_orbits:
        orbit_variant_counts = orbit_variants[orbit_key]
        
        # Check if adding this orbit to train would exceed targets
        would_exceed = False
        for variant, count in orbit_variant_counts.items():
            if train_variants[variant] + count > target_train_variants[variant]:
                would_exceed = True
                break
        
        if would_exceed:
            val_orbits.append(orbit_key)
        else:
            train_orbits.append(orbit_key)
            train_variants.update(orbit_variant_counts)
    
    print(f"Selected {len(train_orbits)} orbits for training, {len(val_orbits)} for validation")
    print(f"Actual train variant distribution: {dict(train_variants)}")
    
    # Collect image paths
    train_paths = []
    val_paths = []
    
    for orbit_key in train_orbits:
        for img in orbit_groups[orbit_key]:
            # Create full path as expected by the training code
            full_path = str(Path("data/probe_images") / img['file_name'])
            train_paths.append(full_path)
    
    for orbit_key in val_orbits:
        for img in orbit_groups[orbit_key]:
            full_path = str(Path("data/probe_images") / img['file_name'])
            val_paths.append(full_path)
    
    return train_paths, val_paths


def convert_to_internal_format(images: List[Dict], annotations: List[Dict]) -> List[Dict]:
    """
    Convert COCO format to internal format expected by training code.
    
    Args:
        images: List of image dictionaries from COCO format
        annotations: List of annotation dictionaries from COCO format
    
    Returns:
        List of records in internal format
    """
    # Create image_id to image mapping
    image_id_to_image = {img['id']: img for img in images}
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[ann['image_id']].append(ann)
    
    internal_records = []
    
    for img in images:
        image_id = img['id']
        img_annotations = annotations_by_image.get(image_id, [])
        
        # Convert COCO bbox format [x, y, width, height] to xyxy format [x1, y1, x2, y2]
        bboxes = []
        for ann in img_annotations:
            x, y, width, height = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            bboxes.append({
                'xyxy': [x1, y1, x2, y2],
                'label': 'probe'  # Single class
            })
        
        # Create internal format record
        record = {
            'image_path': str(Path("data/probe_images") / img['file_name']),
            'width': img['width'],
            'height': img['height'],
            'bboxes': bboxes
        }
        
        internal_records.append(record)
    
    return internal_records


def write_split_files(train_paths: List[str], val_paths: List[str], output_dir: str):
    """Write train.txt and val.txt files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write train.txt
    with open(output_path / "train.txt", 'w') as f:
        for path in train_paths:
            f.write(f"{path}\n")
    
    # Write val.txt
    with open(output_path / "val.txt", 'w') as f:
        for path in val_paths:
            f.write(f"{path}\n")
    
    print(f"✅ Created {output_path / 'train.txt'} with {len(train_paths)} images")
    print(f"✅ Created {output_path / 'val.txt'} with {len(val_paths)} images")


def write_internal_labels(internal_records: List[Dict], output_path: str):
    """Write labels_internal.json file."""
    with open(output_path, 'w') as f:
        json.dump(internal_records, f, indent=2)
    
    print(f"✅ Created {output_path} with {len(internal_records)} records")


def main():
    parser = argparse.ArgumentParser(description="Prepare probe detection dataset")
    parser.add_argument("--labels", default="data/probe_labels.json", 
                       help="Path to input COCO labels file")
    parser.add_argument("--output-dir", default="data", 
                       help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Target training set ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("🚀 PROBE DATASET PREPARATION")
    print("=" * 50)
    
    # Load data
    print(f"Loading data from {args.labels}...")
    images, annotations = load_coco_data(args.labels)
    print(f"Loaded {len(images)} images and {len(annotations)} annotations")
    
    # Group by orbit
    print("Grouping images by orbit...")
    orbit_groups = group_by_orbit(images)
    print(f"Found {len(orbit_groups)} orbits")
    
    # Analyze variant distribution
    orbit_variants = analyze_variants(orbit_groups)
    print("Orbit variant analysis:")
    for orbit, variants in list(orbit_variants.items())[:5]:
        print(f"  {orbit}: {dict(variants)}")
    
    # Create balanced splits
    print(f"\nCreating balanced splits (target train ratio: {args.train_ratio})...")
    train_paths, val_paths = create_balanced_splits(orbit_groups, args.train_ratio)
    
    # Calculate actual ratios
    total_images = len(train_paths) + len(val_paths)
    actual_train_ratio = len(train_paths) / total_images
    print(f"Actual split: {len(train_paths)} train ({actual_train_ratio:.1%}), {len(val_paths)} val ({(1-actual_train_ratio):.1%})")
    
    # Convert to internal format
    print("\nConverting to internal format...")
    internal_records = convert_to_internal_format(images, annotations)
    
    # Write output files
    print("\nWriting output files...")
    write_split_files(train_paths, val_paths, f"{args.output_dir}/splits")
    write_internal_labels(internal_records, f"{args.output_dir}/labels_internal.json")
    
    print("\n✅ Data preparation complete!")
    print(f"   - Created splits: {args.output_dir}/splits/")
    print(f"   - Created labels: {args.output_dir}/labels_internal.json")


if __name__ == "__main__":
    main()
