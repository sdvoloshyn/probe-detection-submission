#!/usr/bin/env python3
"""
Unified operating point selection for YOLO and Faster R-CNN models.

This module provides consistent confidence threshold selection across different
detection frameworks by:
1. Building a validation cache with low-confidence predictions
2. Sweeping confidence thresholds to find optimal operating point
3. Supporting various metrics (F1, precision, recall) and constraints
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image

from ..datasets.canonical import load_canonical_dataset


@dataclass
class Detection:
    """Single detection with confidence score and bounding box."""
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_path: str
    
    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def touches_edge(self, img_width: int, img_height: int, min_pixels: int = 0) -> bool:
        """Check if detection touches image edge within min_pixels tolerance."""
        if min_pixels <= 0:
            return True  # No edge constraint
        
        return (self.x1 <= min_pixels or 
                self.y1 <= min_pixels or 
                self.x2 >= (img_width - min_pixels) or 
                self.y2 >= (img_height - min_pixels))


@dataclass
class ThresholdMetrics:
    """Metrics for a specific confidence threshold."""
    threshold: float
    precision: float
    recall: float
    f1: float
    mean_iou: float
    tp: int
    fp: int
    fn: int
    num_predictions: int
    fp_rate_neg: float


@dataclass
class OperatingPoint:
    """Selected operating point with best threshold and metrics."""
    best_threshold: float
    best_metrics: ThresholdMetrics
    all_thresholds: List[ThresholdMetrics]
    selection_metric: str
    has_negatives: bool


def build_val_cache(
    model: torch.nn.Module,
    val_paths: List[str],
    labels_dict: Dict[str, Any],
    imgsz: int = 640,
    base_conf: float = 0.01,
    nms_iou: float = 0.7,
    device: Optional[torch.device] = None
) -> Dict[str, List[Detection]]:
    """
    Build validation cache by running model once at very low confidence.
    
    Args:
        model: YOLO or RCNN model in eval mode
        val_paths: List of validation image paths
        labels_dict: Dictionary mapping image_path -> labels
        imgsz: Image size for inference
        base_conf: Low confidence threshold to capture all detections
        nms_iou: NMS IoU threshold
        device: Device for inference
        
    Returns:
        Dict mapping image_path -> List[Detection]
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    pred_cache = {}
    
    print(f"Building validation cache for {len(val_paths)} images (conf={base_conf})...")
    
    with torch.no_grad():
        for i, img_path in enumerate(val_paths):
            if i % 50 == 0:
                print(f"  Processing {i+1}/{len(val_paths)}...")
            
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_width, img_height = img.size
            
            # Handle different model types
            if hasattr(model, 'predict'):  # YOLO
                results = model.predict(
                    img_path, 
                    conf=base_conf,
                    iou=nms_iou,
                    imgsz=imgsz,
                    verbose=False
                )
                
                detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for j in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                        conf = float(boxes.conf[j].cpu().numpy())
                        
                        detections.append(Detection(
                            conf=conf,
                            x1=float(x1), y1=float(y1),
                            x2=float(x2), y2=float(y2),
                            image_path=img_path
                        ))
            
            else:  # RCNN
                # Convert PIL to tensor
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                outputs = model(img_tensor)
                
                detections = []
                if outputs and len(outputs) > 0:
                    pred = outputs[0]
                    if 'boxes' in pred and len(pred['boxes']) > 0:
                        boxes = pred['boxes'].cpu().numpy()
                        scores = pred['scores'].cpu().numpy()
                        
                        # Filter by base confidence and apply NMS-like filtering
                        valid_indices = scores >= base_conf
                        boxes = boxes[valid_indices]
                        scores = scores[valid_indices]
                        
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = box
                            detections.append(Detection(
                                conf=float(score),
                                x1=float(x1), y1=float(y1),
                                x2=float(x2), y2=float(y2),
                                image_path=img_path
                            ))

            pred_cache[img_path] = detections
    
    total_detections = sum(len(dets) for dets in pred_cache.values())
    print(f"✅ Built cache: {total_detections} detections across {len(pred_cache)} images")
    
    return pred_cache


def _compute_iou(det: Detection, gt_box: List[float]) -> float:
    """Compute IoU between detection and ground truth box."""
    # GT box format: [x1, y1, x2, y2]
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
    
    # Intersection
    ix1 = max(det.x1, gt_x1)
    iy1 = max(det.y1, gt_y1)
    ix2 = min(det.x2, gt_x2)
    iy2 = min(det.y2, gt_y2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    
    # Union
    det_area = (det.x2 - det.x1) * (det.y2 - det.y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union = det_area + gt_area - intersection
    
    return intersection / union if union > 0 else 0.0


def select_operating_point(
    pred_cache: Dict[str, List[Detection]],
    labels_dict: Dict[str, Any],
    conf_sweep: List[float],
    iou_thresh: float = 0.5,
    max_det: int = 1,
    edge_touch_k: int = 0,
    metric: str = "f1",
    fp_rate_cap: float | None = None
) -> OperatingPoint:
    """
    Select optimal operating point by sweeping confidence thresholds.
    
    Args:
        pred_cache: Dict mapping image_path -> List[Detection]
        labels_dict: Dict mapping image_path -> labels
        conf_sweep: List of confidence thresholds to test
        iou_thresh: IoU threshold for TP/FP classification
        max_det: Maximum detections per image (1 for single-class)
        edge_touch_k: Minimum pixels from edge (0 = no constraint)
        metric: Metric to optimize ("f1", "precision", "recall")
        
    Returns:
        OperatingPoint with best threshold and metrics
    """
    print(f"Selecting operating point with {len(conf_sweep)} thresholds...")
    print(f"Config: iou_thresh={iou_thresh}, max_det={max_det}, edge_touch_k={edge_touch_k}, metric={metric}")
    
    # Check if we have any negative samples (images with no GT boxes)
    has_negatives = False
    total_gt = 0
    negative_images: List[str] = []
    
    for img_path in pred_cache.keys():
        gt_data = labels_dict[img_path]
        gt_boxes = gt_data['bboxes']
        if len(gt_boxes) == 0:
            has_negatives = True
            negative_images.append(img_path)
        total_gt += len(gt_boxes)
    
    print(f"Dataset stats: {len(pred_cache)} images, {total_gt} GT boxes, has_negatives={has_negatives}")
    
    threshold_results = []
    best_score = -1.0
    best_threshold = conf_sweep[0] if conf_sweep else 0.5
    best_metrics = None
    
    for conf_thresh in conf_sweep:
        tp, fp, fn = 0, 0, 0
        total_iou = 0.0
        matched_pairs = 0
        total_predictions = 0
        neg_fp_images = 0
        
        for img_path, detections in pred_cache.items():
            # Get ground truth
            gt_data = labels_dict[img_path]
            gt_boxes = [bbox_data['xyxy'] for bbox_data in gt_data['bboxes']]
            
            # Filter detections by confidence
            valid_dets = [d for d in detections if d.conf >= conf_thresh]
            
            # Apply edge touch filter if specified
            if edge_touch_k > 0:
                img = Image.open(img_path)
                img_width, img_height = img.size
                valid_dets = [d for d in valid_dets if d.touches_edge(img_width, img_height, edge_touch_k)]
            
            # Keep only top detections (sorted by confidence)
            valid_dets = sorted(valid_dets, key=lambda x: x.conf, reverse=True)[:max_det]
            total_predictions += len(valid_dets)
            
            if len(gt_boxes) == 0:
                # Negative image: all predictions are FP
                fp += len(valid_dets)
                if len(valid_dets) > 0:
                    neg_fp_images += 1
            else:
                # Positive image: match predictions to GT
                used_gt = set()
                
                for det in valid_dets:
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in used_gt:
                            continue
                        
                        iou = _compute_iou(det, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_thresh:
                        tp += 1
                        used_gt.add(best_gt_idx)
                        total_iou += best_iou
                        matched_pairs += 1
                    else:
                        fp += 1
                
                # Unmatched GT boxes are FN
                fn += len(gt_boxes) - len(used_gt)
        
        # Compute metrics
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        mean_iou = total_iou / max(matched_pairs, 1e-8)
        
        fp_rate_neg = (neg_fp_images / max(len(negative_images), 1)) if has_negatives else 0.0
        metrics = ThresholdMetrics(
            threshold=conf_thresh,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=mean_iou,
            tp=tp,
            fp=fp,
            fn=fn,
            num_predictions=total_predictions,
            fp_rate_neg=fp_rate_neg
        )
        threshold_results.append(metrics)
        
        print(f"  thresh={conf_thresh:.3f}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, IoU={mean_iou:.4f} (TP={tp}, FP={fp}, FN={fn})")
        
        # Check if this is the best threshold
        if metric == "f1":
            score = f1
        elif metric == "precision":
            score = precision
        elif metric == "recall":
            score = recall
        else:
            score = f1  # Default to F1
        
        # Enforce FP-rate cap if provided
        within_cap = True
        if fp_rate_cap is not None:
            within_cap = metrics.fp_rate_neg <= fp_rate_cap
        
        # Update best if score is better (within cap), or tie-break on precision
        if within_cap and (score > best_score or 
            (abs(score - best_score) < 1e-6 and precision > (best_metrics.precision if best_metrics else -1))):
            best_score = score
            best_threshold = conf_thresh
            best_metrics = metrics
    
    # Special case: if no negatives, prefer lowest confidence (favor recall)
    if not has_negatives and threshold_results:
        best_threshold = min(conf_sweep)
        best_metrics = next(m for m in threshold_results if m.threshold == best_threshold)
        print(f"  No negatives detected → using lowest confidence: {best_threshold:.3f}")
    
    # If FP cap filtered out all thresholds, pick the one with lowest fp_rate_neg, then highest F1
    if best_metrics is None and threshold_results:
        threshold_results_sorted = sorted(
            threshold_results,
            key=lambda m: (m.fp_rate_neg, -m.f1)
        )
        best_metrics = threshold_results_sorted[0]
        best_threshold = best_metrics.threshold
        print("⚠️  No threshold satisfied fp_rate_cap; selecting by minimum fp_rate_neg then max F1.")

    print(f"✅ Selected operating point: conf={best_threshold:.3f} ({metric}={getattr(best_metrics, metric):.4f})")
    
    return OperatingPoint(
        best_threshold=best_threshold,
        best_metrics=best_metrics,
        all_thresholds=threshold_results,
        selection_metric=metric,
        has_negatives=has_negatives
    )


def save_operating_point_results(
    op_point: OperatingPoint,
    save_dir: Union[str, Path],
    prefix: str = ""
) -> None:
    """Save operating point selection results to JSON files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_path = save_dir / f"{prefix}op_select_summary.json"
    summary_data = {
        "best_threshold": op_point.best_threshold,
        "selection_metric": op_point.selection_metric,
        "has_negatives": op_point.has_negatives,
        "best_metrics": asdict(op_point.best_metrics)
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save per-threshold results
    per_thresh_path = save_dir / f"{prefix}per_threshold.json"
    per_thresh_data = {
        "thresholds": [asdict(m) for m in op_point.all_thresholds],
        "config": {
            "selection_metric": op_point.selection_metric,
            "has_negatives": op_point.has_negatives
        }
    }
    
    with open(per_thresh_path, 'w') as f:
        json.dump(per_thresh_data, f, indent=2)
    
    print(f"✅ Saved operating point results to {save_dir}/")


def save_best_metadata(
    op_point: OperatingPoint,
    weights_path: Union[str, Path],
    additional_config: Optional[Dict] = None
) -> None:
    """Save best.meta.json alongside model weights."""
    weights_path = Path(weights_path)
    meta_path = weights_path.parent / "best.meta.json"
    
    meta_data = {
        "conf": op_point.best_threshold,
        "metrics": asdict(op_point.best_metrics),
        "selection_metric": op_point.selection_metric,
        "has_negatives": op_point.has_negatives
    }
    
    if additional_config:
        meta_data.update(additional_config)
    
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    print(f"✅ Saved model metadata to {meta_path}")
