"""Geometry helpers used by person-quality checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.algorithm_config import algorithm_config
from utils.keyframe.person_crop import (
    PersonCropResult,
    project_rect_to_person_crop,
)

def _image_hw(image_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(image_shape) < 2:
        return 0, 0
    return max(0, _safe_int(image_shape[0])), max(0, _safe_int(image_shape[1]))

def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0

def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number

def _rect_area(rect: dict[str, Any]) -> int:
    return max(0, _safe_int(rect.get("w"))) * max(0, _safe_int(rect.get("h")))

def _rect_xyxy(rect: dict[str, Any]) -> tuple[int, int, int, int]:
    x = _safe_int(rect.get("x"))
    y = _safe_int(rect.get("y"))
    w = _safe_int(rect.get("w"))
    h = _safe_int(rect.get("h"))
    return x, y, x + w, y + h

def _project_rect_to_person_crop(
    target: np.ndarray,
    rect: dict[str, Any],
    crop_result: PersonCropResult,
) -> None:
    """Conservatively mark a whole-image rectangle on a person crop canvas."""
    target |= project_rect_to_person_crop(rect, crop_result)

def _image_edge_touch_count(rect: dict[str, Any], img_w: int, img_h: int) -> int:
    """Return the number of image edges near the clipped bbox (diagnostic only)."""
    x1, y1, x2, y2 = _rect_xyxy(rect)
    margin = max(
        algorithm_config.person.quality.edge.image_edge_margin_min_px,
        int(round(min(img_w, img_h) * algorithm_config.person.quality.edge.image_edge_margin_ratio)),
    )
    return sum(
        [
            x1 <= margin,
            y1 <= margin,
            x2 >= img_w - margin,
            y2 >= img_h - margin,
        ]
    )

def _image_edge_contact_directions(
    rect: dict[str, Any], img_w: int, img_h: int
) -> frozenset[str]:
    """Return image edges actually contacted by the clipped bbox."""
    if img_w <= 0 or img_h <= 0:
        return frozenset()
    x1, y1, x2, y2 = _rect_xyxy(rect)
    epsilon = 1
    contacts = []
    if x1 <= epsilon:
        contacts.append("left")
    if y1 <= epsilon:
        contacts.append("top")
    if x2 >= img_w - epsilon:
        contacts.append("right")
    if y2 >= img_h - epsilon:
        contacts.append("bottom")
    return frozenset(contacts)

def _raw_bbox_overflow(
    xyxy: Any, img_w: int, img_h: int
) -> tuple[float, frozenset[str]]:
    """Return clipped-away raw bbox area and the directions that exceed the frame."""
    try:
        values = np.asarray(xyxy, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return 0.0, frozenset()
    if values.size < 4 or not bool(np.all(np.isfinite(values[:4]))):
        return 0.0, frozenset()
    x1, y1, x2, y2 = (float(value) for value in values[:4])
    raw_width = max(0.0, x2 - x1)
    raw_height = max(0.0, y2 - y1)
    raw_area = raw_width * raw_height
    if raw_area <= 0.0 or img_w <= 0 or img_h <= 0:
        return 0.0, frozenset()

    clipped_x1 = max(0.0, min(float(img_w), x1))
    clipped_y1 = max(0.0, min(float(img_h), y1))
    clipped_x2 = max(0.0, min(float(img_w), x2))
    clipped_y2 = max(0.0, min(float(img_h), y2))
    clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(
        0.0, clipped_y2 - clipped_y1
    )
    overflow_ratio = max(0.0, min(1.0, 1.0 - clipped_area / raw_area))
    directions = []
    if x1 < 0.0:
        directions.append("left")
    if y1 < 0.0:
        directions.append("top")
    if x2 > img_w:
        directions.append("right")
    if y2 > img_h:
        directions.append("bottom")
    return float(overflow_ratio), frozenset(directions)

def _max_other_person_overlap_ratio(
    rect: dict[str, Any],
    all_segments: list[dict[str, Any]],
    *,
    segment_index: int | None,
) -> float:
    current_area = _rect_area(rect)
    if current_area <= 0:
        return 0.0
    x1, y1, x2, y2 = _rect_xyxy(rect)
    max_ratio = 0.0
    for index, other in enumerate(all_segments):
        if segment_index is not None and index == segment_index:
            continue
        if segment_index is None and other.get("rect") is rect:
            continue
        other_rect = other.get("rect") or {}
        ox1, oy1, ox2, oy2 = _rect_xyxy(other_rect)
        inter_w = max(0, min(x2, ox2) - max(x1, ox1))
        inter_h = max(0, min(y2, oy2) - max(y1, oy1))
        max_ratio = max(max_ratio, (inter_w * inter_h) / current_area)
    return max_ratio
