from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from .filters import edge_touch_filter


def _pr_f1_at_thresholds(
    pred_scores: List[float],
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    conf_thresholds: List[float],
    iou_thresh: float,
    image_height: int,
    image_width: int,
    edge_k: Optional[int] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Compute simple Precision/Recall/F1 per confidence threshold with greedy IoU matching.
    Assumes single-class and at-most-one detection policy will be applied upstream.
    """
    # Optionally filter by edge-touch
    if edge_k is not None:
        keep_mask = edge_touch_filter(pred_boxes, image_height, image_width, edge_k)
        pred_boxes = [b for b, k in zip(pred_boxes, keep_mask) if k]
        pred_scores = [s for s, k in zip(pred_scores, keep_mask) if k]

    # Pre-sort predictions by score desc
    order = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in order]
    pred_scores = [pred_scores[i] for i in order]

    # Greedy IoU matcher
    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter + 1e-9
        return inter / union

    results: Dict[float, Dict[str, float]] = {}
    for t in conf_thresholds:
        # Apply threshold and max_det=1 policy
        filtered = [(b, s) for b, s in zip(pred_boxes, pred_scores) if s >= t]
        if len(filtered) > 1:
            filtered = [max(filtered, key=lambda x: x[1])]

        tp = 0; fp = 0; fn = 0
        matched_gt = [False] * len(gt_boxes)
        for b, s in filtered:
            # Find best match
            best_iou = 0.0; best_j = -1
            for j, g in enumerate(gt_boxes):
                if matched_gt[j]:
                    continue
                i = iou(b, g)
                if i > best_iou:
                    best_iou = i; best_j = j
            if best_iou >= iou_thresh and best_j >= 0:
                tp += 1
                matched_gt[best_j] = True
            else:
                fp += 1
        fn = matched_gt.count(False)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        results[t] = {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}
    return results


def select_operating_point(
    conf_sweep: List[float],
    metric: str,
    per_image_predictions: List[Tuple[List[float], List[List[float]], int, int]],
    per_image_gt_boxes: List[List[List[float]]],
    iou_thresh: float,
    edge_k: Optional[int] = None,
) -> Dict:
    """
    Given predictions and GT per image, sweep confidence thresholds and select best metric.
    per_image_predictions: list of (scores, boxes_xyxy, image_height)
    per_image_gt_boxes: list of gt boxes per image
    Returns dict with best_conf and metrics at best.
    """
    # Aggregate over images by summing counts
    agg_by_t = {t: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for t in conf_sweep}

    for (scores, boxes, H, W), gt_boxes in zip(per_image_predictions, per_image_gt_boxes):
        per_t = _pr_f1_at_thresholds(scores, boxes, gt_boxes, conf_sweep, iou_thresh, H, W, edge_k)
        for t, m in per_t.items():
            agg = agg_by_t[t]
            agg["tp"] += m.get("tp", 0.0)
            agg["fp"] += m.get("fp", 0.0)
            agg["fn"] += m.get("fn", 0.0)

    # Reduce to metrics and keep counts
    summary = {}
    for t in conf_sweep:
        tp = agg_by_t[t]["tp"]; fp = agg_by_t[t]["fp"]; fn = agg_by_t[t]["fn"]
        precision = float(tp / (tp + fp + 1e-9))
        recall = float(tp / (tp + fn + 1e-9))
        f1 = float(2 * precision * recall / (precision + recall + 1e-9))
        summary[t] = {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}

    # Select best by metric with tie-break to lower conf
    key = metric.lower()
    best_conf = None
    best_val = -1.0
    for t in sorted(conf_sweep):
        val = summary[t].get(key, 0.0)
        if (val > best_val) or (abs(val - best_val) < 1e-12 and (best_conf is None or t < best_conf)):
            best_val = val
            best_conf = t
    best_metrics = summary.get(best_conf, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0})
    return {"best_conf": best_conf, "metrics": best_metrics, "per_threshold": summary}


