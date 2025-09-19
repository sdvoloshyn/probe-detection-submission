"""
Evaluation metrics helpers.

Utilities for computing IoU statistics over matched true positives only.
"""

from __future__ import annotations

from typing import Iterable, Dict, Tuple, Any
import numpy as np


class Match:
    """Lightweight container for a matched GT-Pred pair.

    Attributes:
        gt_id: ground-truth index within image (optional)
        pred_id: prediction index within image (optional)
        iou: IoU value for the match (float)
        conf: prediction confidence (float)
        img_path: path to the image for this match (str)
        extra: optional dictionary for additional metadata
    """

    def __init__(self, gt_id: int | None, pred_id: int | None, iou: float | None,
                 conf: float | None, img_path: str | None, extra: Dict[str, Any] | None = None):
        self.gt_id = gt_id
        self.pred_id = pred_id
        self.iou = iou
        self.conf = conf
        self.img_path = img_path
        self.extra = extra or {}


def mean_iou_tp_only(matches: Iterable[Match]) -> float:
    """
    Compute mean IoU over matched pairs only.

    Args:
        matches: iterable of Match objects or tuples with an .iou attribute/field

    Returns:
        Mean IoU over matches with valid IoU; 0.0 if none.
    """
    ious = []
    for m in matches:
        iou_val = m.iou if hasattr(m, 'iou') else m[2]
        ious.append(float(iou_val))
    return float(np.mean(ious)) if ious else 0.0


def iou_percentiles_tp_only(matches: Iterable[Match], qs: Tuple[float, ...] = (0.5, 0.75, 0.9)) -> Dict[float, float]:
    """
    Compute IoU percentiles over matched TPs only.

    Args:
        matches: iterable of Match objects
        qs: tuple of quantiles in [0,1]

    Returns:
        Dict mapping quantile -> value; 0.0 for empty input.
    """
    ious = []
    for m in matches:
        iou_val = m.iou if hasattr(m, 'iou') else m[2]
        ious.append(float(iou_val))

    if not ious:
        return {q: 0.0 for q in qs}
    arr = np.asarray(ious, dtype=np.float32)
    return {q: float(np.quantile(arr, q)) for q in qs}


