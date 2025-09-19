from __future__ import annotations

from typing import List


def edge_touch_filter(
    boxes_xyxy: List[List[float]],
    image_height: int,
    image_width: int,
    k: int,
) -> List[bool]:
    """
    Return a boolean mask indicating which boxes touch any image edge by at
    least k pixels (top, bottom, left, or right).
    """
    mask: List[bool] = []
    if k is None or k <= 0:
        return [True] * len(boxes_xyxy)
    for x1, y1, x2, y2 in boxes_xyxy:
        touches_top = y1 <= k
        touches_bottom = (image_height - y2) <= k
        touches_left = x1 <= k
        touches_right = (image_width - x2) <= k
        mask.append(bool(touches_top or touches_bottom or touches_left or touches_right))
    return mask


