#!/usr/bin/env python3
"""
Prepare seeded background negatives and inject into splits.

Supports two modes:
- LOMO CV folds: splits_root contains subdirs named fold_*/ with train.txt and val.txt
- Single split: splits_root contains train.txt and val.txt at the root

Negatives are appended to the labels JSON and split files are written to an
output splits directory (configurable).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image


@dataclass
class NegCandidate:
    x: int
    y: int
    w: int
    h: int
    score: float


def _read_split_file(p: Path) -> List[str]:
    with open(p, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _write_split_file(p: Path, paths: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for line in paths:
            f.write(f"{line}\n")


def _parse_mission_id(image_path: str) -> str:
    stem = Path(image_path).stem
    return stem.split("_")[0]


def _clamp_aspect(target_ar: Optional[float], img_w: int, img_h: int) -> float:
    if target_ar is not None and target_ar > 0:
        return float(target_ar)
    # Infer from frame if not provided
    return float(img_w) / float(img_h)


def _dilate_forbidden_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return mask
    k = max(1, int(margin))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def _boxes_to_mask(h: int, w: int, boxes: List[List[float]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        xi1 = max(0, int(math.floor(x1)))
        yi1 = max(0, int(math.floor(y1)))
        xi2 = min(w, int(math.ceil(x2)))
        yi2 = min(h, int(math.ceil(y2)))
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = 1
    return mask


def _integral(img: np.ndarray) -> np.ndarray:
    return cv2.integral(img, sdepth=cv2.CV_64F)


def _window_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    # integral has shape (H+1, W+1)
    x2 = x + w
    y2 = y + h
    return (
        integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
    )


def _sample_candidates(
    allowed_mask: np.ndarray,
    sizes: List[Tuple[int, int]],
    grid_step_px: int,
    edge_touch_k: int,
) -> List[Tuple[int, int, int, int]]:
    H, W = allowed_mask.shape
    integral_forbidden = _integral(1 - allowed_mask)  # 1s where forbidden
    candidates: List[Tuple[int, int, int, int]] = []
    for (cw, ch) in sizes:
        step = max(8, grid_step_px)
        found_any = False
        for y in range(0, H - ch + 1, step):
            if edge_touch_k > 0 and (y < edge_touch_k or (y + ch) > (H - edge_touch_k)):
                continue
            for x in range(0, W - cw + 1, step):
                s = _window_sum(integral_forbidden, x, y, cw, ch)
                if s == 0:  # No overlap with forbidden
                    candidates.append((x, y, cw, ch))
                    found_any = True
        # If none at this size, try next smaller; if some found, continue to also allow smaller
        _ = found_any
    return candidates


def _score_candidates(allowed_mask: np.ndarray, candidates: List[Tuple[int, int, int, int]]) -> List[NegCandidate]:
    # Distance transform on allowed region (allowed>0 -> foreground). Distance to nearest forbidden (zeros).
    dist = cv2.distanceTransform((allowed_mask > 0).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)
    scored: List[NegCandidate] = []
    for (x, y, w, h) in candidates:
        sub = dist[y:y+h:4, x:x+w:4]
        if sub.size == 0:
            continue
        score = float(sub.mean())
        scored.append(NegCandidate(x=x, y=y, w=w, h=h, score=score))
    return scored


def _softmax_choice(rng: np.random.RandomState, items: List[NegCandidate], temperature: float = 0.5) -> Optional[NegCandidate]:
    if not items:
        return None
    scores = np.array([c.score for c in items], dtype=np.float64)
    s = scores / max(1e-9, temperature)
    s = s - s.max()
    p = np.exp(s)
    p = p / p.sum()
    idx = rng.choice(len(items), p=p)
    return items[idx]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_jpg(image: np.ndarray, path: Path, target_size: Tuple[int, int] = (640, 400)) -> None:
    _ensure_dir(path.parent)
    # Resize image to target size to match positive images
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(path), resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def _save_empty_label(path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w") as f:
        f.write("")


def _generate_negatives_for_images(
    images: List[str], 
    mission: str, 
    fold_idx: int,
    rng: np.random.RandomState,
    repo_root: Path,
    img_to_boxes: Dict[str, List[List[float]]],
    out_neg_root: Path,
    args,
    t_ar: float,
    labels_list: List[Dict],
    existing_image_paths: set,
    new_label_records: List[Dict],
    rows: List[Dict],
    fold_rows: List[Dict]
) -> List[str]:
    """Generate negatives for a list of images and return the negative paths."""
    negatives = []
    positives = len(images)
    quota = int(math.ceil(args.per_mission_neg_frac * positives))
    picked = 0

    order = list(range(len(images)))
    rng.shuffle(order)

    for idx in order:
        if picked >= quota:
            break
        src_rel = images[idx]
        src_abs = repo_root / src_rel
        try:
            with Image.open(src_abs) as im:
                W, H = im.size
                image_np = np.array(im.convert("RGB"))
        except Exception:
            continue

        boxes = img_to_boxes.get(src_rel, [])
        forb = _boxes_to_mask(H, W, boxes)
        forb = _dilate_forbidden_mask(forb, args.margin)
        allowed = (forb == 0).astype(np.uint8)
        if allowed.sum() < 128 * 128:
            continue

        m = min(H, W)
        sizes = []
        for ratio_str in args.size_ladder.split(","):
            r = float(ratio_str.strip())
            h_c = int(round(m * r))
            w_c = int(round(h_c * t_ar))
            if w_c > W:
                w_c = W
                h_c = int(round(W / max(1e-9, t_ar)))
            if h_c > H:
                h_c = H
                w_c = int(round(H * t_ar))
            ar = (w_c / max(1, h_c))
            if abs(ar - t_ar) > args.ar_tol:
                continue
            if min(h_c, w_c) < 128:
                continue
            sizes.append((w_c, h_c))
        if not sizes:
            continue

        step_px = max(8, int(round(args.grid_step * m)))
        cand_rects = _sample_candidates(allowed, sizes, step_px, args.edge_touch_k)
        if not cand_rects:
            continue

        scored = _score_candidates(allowed, cand_rects)
        chosen = _softmax_choice(rng, scored, temperature=0.5)
        if chosen is None:
            continue

        x, y, w, h = chosen.x, chosen.y, chosen.w, chosen.h
        crop = image_np[y:y+h, x:x+w]

        src_stem = Path(src_rel).stem
        neg_name = f"{src_stem}__neg.jpg"

        # Save negative image (absolute path), then compute repo-relative for split file
        neg_abs_path = out_neg_root / mission / neg_name
        target_size = (640, 400)  # Match positive image dimensions
        _save_jpg(crop, neg_abs_path, target_size)
        neg_rel_path = Path(os.path.relpath(neg_abs_path, start=repo_root)).as_posix()

        # Add to labels JSON if not present (idempotent)
        if neg_rel_path not in existing_image_paths:
            new_rec = {
                "image_path": neg_rel_path,
                "width": target_size[0],
                "height": target_size[1],
                "bboxes": [],
            }
            labels_list.append(new_rec)
            existing_image_paths.add(neg_rel_path)
            new_label_records.append(new_rec)

        rec = {
            "mission": mission,
            "fold": fold_idx,
            "src_image": src_rel,
            "neg_image_path": neg_rel_path,
            "crop_xywh": [int(x), int(y), int(w), int(h)],
            "resized_xywh": [0, 0, target_size[0], target_size[1]],
            "crop_ar": float(w) / float(max(1, h)),
            "resized_ar": float(target_size[0]) / float(target_size[1]),
            "score": float(chosen.score),
            "size_ratio": float(min(h, w)) / float(min(H, W)),
            "kept": True,
        }
        rows.append(rec)
        fold_rows.append(rec)
        negatives.append(neg_rel_path)
        picked += 1

    return negatives


def _infer_target_ar(images: List[str], rng: np.random.RandomState, repo_root: Path) -> float:
    # pick a random frame and compute W/H
    for _ in range(min(10, len(images))):
        rel = images[rng.randint(0, len(images))]
        p = repo_root / rel
        try:
            with Image.open(p) as img:
                w, h = img.size
                if h > 0:
                    return float(w) / float(h)
        except Exception:
            continue
    return 16.0 / 9.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare seeded background negatives and inject into splits (LOMO folds or single split)")
    parser.add_argument("--images-root", type=str, required=True)
    parser.add_argument("--labels-json", type=str, required=True)
    parser.add_argument("--out-labels-json", type=str, default="", help="If set, write updated labels to this path instead of overwriting labels-json")
    parser.add_argument("--splits-root", type=str, required=True)
    parser.add_argument("--out-neg-root", type=str, required=True)
    parser.add_argument("--yolo-labels-root", type=str, default="", help="Deprecated; negatives now added to labels JSON")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--target-ar", type=float, default=-1.0)
    parser.add_argument("--ar-tol", type=float, default=0.1)
    parser.add_argument("--margin", type=int, default=6)
    parser.add_argument("--size-ladder", type=str, default="1.0,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60, 0.55, 0.5")
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--one-per-image", action="store_true", default=True)
    parser.add_argument("--per-mission-neg-frac", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-touch-k", type=int, default=0)
    parser.add_argument("--out-splits-root", type=str, default="")
    parser.add_argument("--neg-in-train", action="store_true", default=False, help="Add negatives to training set")
    parser.add_argument("--neg-in-val", action="store_true", default=False, help="Add negatives to validation set")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    splits_root = repo_root / args.splits_root
    out_splits_root = Path(args.out_splits_root) if args.out_splits_root else splits_root.parent / f"{splits_root.name}_neg"
    out_neg_root = repo_root / args.out_neg_root
    # Deprecated path, no longer used
    yolo_labels_root = repo_root / (args.yolo_labels_root or "data/yolo_labels/negatives")

    # Load labels JSON to get boxes per image (keys are project-relative paths in split files)
    in_labels_path = repo_root / args.labels_json
    with open(in_labels_path, "r") as f:
        labels_list = json.load(f)
    img_to_boxes: Dict[str, List[List[float]]] = {}
    existing_image_paths = set()
    for rec in labels_list:
        img_to_boxes[rec["image_path"]] = [b["xyxy"] for b in rec.get("bboxes", [])]
        existing_image_paths.add(rec["image_path"])
    new_label_records: List[Dict] = []

    # Iterate folds or handle single split
    fold_dirs = sorted([p for p in splits_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    single_split_mode = False
    if not fold_dirs:
        # Single-split support: expect train.txt and val.txt at root
        if (splits_root / "train.txt").exists() and (splits_root / "val.txt").exists():
            single_split_mode = True
            fold_dirs = [splits_root]
        else:
            raise FileNotFoundError(f"No fold_* subdirectories and missing train.txt/val.txt under {splits_root}")

    # QA outputs
    qa_dir = repo_root / "runs" / "negatives"
    summary_csv = qa_dir / "neg_summary.csv"
    summary_json = qa_dir / "neg_summary.json"
    previews_dir = qa_dir / "previews"
    _ensure_dir(qa_dir)
    _ensure_dir(previews_dir)

    rows: List[Dict] = []

    for fold_idx, fold_dir in enumerate(fold_dirs, start=1):
        fold_name = "single" if single_split_mode else fold_dir.name
        print(f"Processing fold {fold_idx}/{len(fold_dirs)}: {fold_name}")
        train_list = _read_split_file(fold_dir / "train.txt")
        val_list = _read_split_file(fold_dir / "val.txt")
        print(f"  Train: {len(train_list)} images")
        print(f"  Val: {len(val_list)} images")
        print(f"  Negatives in train: {args.neg_in_train}")
        print(f"  Negatives in val: {args.neg_in_val}")

        # Determine target AR if needed
        t_ar = args.target_ar if args.target_ar > 0 else _infer_target_ar(train_list, rng, repo_root)

        # Group by mission
        mission_to_train_images: Dict[str, List[str]] = {}
        for p in train_list:
            mission_to_train_images.setdefault(_parse_mission_id(p), []).append(p)

        injected_train = list(train_list)
        injected_val = list(val_list)
        fold_rows: List[Dict] = []

        # Per mission quota (TRAIN) - only if --neg-in-train is set
        train_negatives_count = 0
        if args.neg_in_train:
            for mission, src_images in mission_to_train_images.items():
                train_negatives = _generate_negatives_for_images(
                    src_images, mission, fold_idx, rng, repo_root, img_to_boxes, 
                    out_neg_root, args, t_ar, labels_list, existing_image_paths, 
                    new_label_records, rows, fold_rows
                )
                injected_train.extend(train_negatives)
                train_negatives_count += len(train_negatives)

        # If requested, also generate negatives for VALIDATION from the hold-out mission
        val_negatives_count = 0
        if args.neg_in_val:
            val_mission_to_images: Dict[str, List[str]] = {}
            for p in val_list:
                val_mission_to_images.setdefault(_parse_mission_id(p), []).append(p)

            for mission, src_images in val_mission_to_images.items():
                val_negatives = _generate_negatives_for_images(
                    src_images, mission, fold_idx, rng, repo_root, img_to_boxes, 
                    out_neg_root, args, t_ar, labels_list, existing_image_paths, 
                    new_label_records, rows, fold_rows
                )
                injected_val.extend(val_negatives)
                val_negatives_count += len(val_negatives)

        # Calculate and print statistics
        train_original_count = len(train_list)
        val_original_count = len(val_list)
        train_total_count = len(injected_train)
        val_total_count = len(injected_val)
        
        train_neg_pct = (train_negatives_count / train_original_count * 100) if train_original_count > 0 else 0
        val_neg_pct = (val_negatives_count / val_original_count * 100) if val_original_count > 0 else 0
        
        print(f"  Train: {train_original_count} → {train_total_count} (+{train_negatives_count}, +{train_neg_pct:.1f}%)")
        print(f"  Val: {val_original_count} → {val_total_count} (+{val_negatives_count}, +{val_neg_pct:.1f}%)")

        # Write out new fold splits
        out_fold = out_splits_root if single_split_mode else (out_splits_root / fold_dir.name)
        _ensure_dir(out_fold)
        
        # Write training split (with or without negatives based on --neg-in-train)
        if args.neg_in_train:
            _write_split_file(out_fold / "train.txt", injected_train)
        else:
            _write_split_file(out_fold / "train.txt", train_list)
            
        # Write validation split (with or without negatives based on --neg-in-val)
        if args.neg_in_val:
            _write_split_file(out_fold / "val.txt", injected_val)
        else:
            _write_split_file(out_fold / "val.txt", val_list)

        # Create a small preview montage with up to 10 negatives for this fold
        try:
            if fold_rows:
                fold_rng = np.random.RandomState(args.seed + fold_idx)
                idxs = list(range(len(fold_rows)))
                fold_rng.shuffle(idxs)
                idxs = idxs[:10]

                tiles_per_row = 5
                rows_grid = 2
                tile = int(args.imgsz)
                montage = np.zeros((rows_grid * tile, tiles_per_row * tile, 3), dtype=np.uint8)

                for i, ridx in enumerate(idxs):
                    r = fold_rows[ridx]
                    src_rel = r["src_image"]
                    x, y, w, h = r["crop_xywh"]
                    src_abs = repo_root / src_rel
                    img = cv2.imread(str(src_abs))
                    if img is None:
                        continue
                    Hs, Ws = img.shape[:2]
                    scale = min(tile / max(1, Hs), tile / max(1, Ws))
                    newW = int(round(Ws * scale))
                    newH = int(round(Hs * scale))
                    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
                    tile_img = np.zeros((tile, tile, 3), dtype=np.uint8)
                    dx = (tile - newW) // 2
                    dy = (tile - newH) // 2
                    tile_img[dy:dy+newH, dx:dx+newW] = resized
                    x1 = int(round(x * scale + dx))
                    y1 = int(round(y * scale + dy))
                    x2 = int(round((x + w) * scale + dx))
                    y2 = int(round((y + h) * scale + dy))
                    cv2.rectangle(tile_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    row_i = i // tiles_per_row
                    col_i = i % tiles_per_row
                    montage[row_i*tile:(row_i+1)*tile, col_i*tile:(col_i+1)*tile] = tile_img

                prev_path = previews_dir / ("single.jpg" if single_split_mode else f"fold_{fold_idx}.jpg")
                _ensure_dir(prev_path.parent)
                cv2.imwrite(str(prev_path), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except Exception:
            pass

    # Save summaries
    _ensure_dir(summary_csv.parent)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mission","fold","src_image","neg_image_path","crop_xywh","resized_xywh","crop_ar","resized_ar","score","size_ratio","kept"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    # Persist updated labels JSON (with negatives appended)
    out_labels_path = Path(args.out_labels_json) if args.out_labels_json else in_labels_path
    _ensure_dir(out_labels_path.parent)
    with open(out_labels_path, "w") as f:
        json.dump(labels_list, f)

    print(f"✅ Negatives prepared. Splits written to: {out_splits_root}")
    if args.out_labels_json:
        print(f"✅ Updated labels written to: {out_labels_path}")


if __name__ == "__main__":
    main()


