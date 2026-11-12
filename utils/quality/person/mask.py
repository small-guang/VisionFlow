"""Mask validation and shape-analysis helpers for person quality."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.algorithm_config import algorithm_config
from utils.person_mask import PersonMask, as_person_mask

from .geometry import _rect_xyxy

def _valid_mask(mask: Any, img_h: int, img_w: int) -> PersonMask | None:
    if mask is None or img_h <= 0 or img_w <= 0:
        return None
    region = as_person_mask(mask, (img_h, img_w))
    if region is None:
        return None
    return PersonMask(region.data.astype(bool, copy=False), region.origin, region.image_shape)

def _connected_component_areas(mask: PersonMask | np.ndarray) -> list[int]:
    data = mask.data if isinstance(mask, PersonMask) else mask
    cv2_module = _cv2_module()
    if cv2_module is not None:
        connected = getattr(cv2_module, "connectedComponentsWithStats", None)
        if connected is not None:
            _, _, stats, _ = connected(data.astype(np.uint8), 8)
            return [int(area) for area in stats[1:, 4].tolist() if int(area) > 0]
    return _connected_component_areas_fallback(data)

def _cv2_module() -> Any:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2

def _connected_component_areas_fallback(mask: np.ndarray) -> list[int]:
    visited = np.zeros(mask.shape, dtype=bool)
    ys, xs = np.nonzero(mask)
    areas: list[int] = []
    height, width = mask.shape
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        areas.append(area)
    return areas

def _mask_bbox_edge_stats(
    mask: PersonMask | np.ndarray, rect: dict[str, Any]
) -> tuple[float, int]:
    region = as_person_mask(mask)
    if region is None:
        return 0.0, 0

    crop = region.window(_rect_xyxy(rect))
    mask_area = int(np.count_nonzero(crop))
    if mask_area <= 0:
        return 0.0, 0

    band = max(1, int(round(min(crop.shape[0], crop.shape[1]) * 0.03)))
    top = crop[:band, :]
    bottom = crop[-band:, :]
    left = crop[:, :band]
    right = crop[:, -band:]
    edge_counts = [
        int(np.count_nonzero(top)),
        int(np.count_nonzero(bottom)),
        int(np.count_nonzero(left)),
        int(np.count_nonzero(right)),
    ]
    touched_threshold = max(
        algorithm_config.person.quality.edge.mask_bbox_touch_min_pixels,
        int(round(mask_area * algorithm_config.person.quality.edge.mask_bbox_touch_ratio)),
    )
    touched_edges = sum(count >= touched_threshold for count in edge_counts)

    edge_mask = np.zeros_like(crop, dtype=bool)
    edge_mask[:band, :] = True
    edge_mask[-band:, :] = True
    edge_mask[:, :band] = True
    edge_mask[:, -band:] = True
    edge_ratio = int(np.count_nonzero(crop & edge_mask)) / mask_area
    return float(edge_ratio), int(touched_edges)

def _mask_image_edge_directions(mask: PersonMask | np.ndarray) -> frozenset[str]:
    """Return original-image edges with meaningful foreground support."""
    region = as_person_mask(mask)
    if region is None:
        return frozenset()
    mask_area = region.count()
    if mask_area <= 0:
        return frozenset()
    height, width = region.image_shape
    band = max(
        algorithm_config.person.quality.edge.mask_image_band_min_px,
        int(round(min(height, width) * algorithm_config.person.quality.edge.mask_image_band_ratio)),
    )
    touched_threshold = max(
        algorithm_config.person.quality.edge.mask_image_touch_min_pixels,
        int(round(mask_area * algorithm_config.person.quality.edge.mask_image_touch_ratio)),
    )
    edge_counts = {
        "left": region.count((0, 0, band, height)),
        "top": region.count((0, 0, width, band)),
        "right": region.count((width - band, 0, width, height)),
        "bottom": region.count((0, height - band, width, height)),
    }
    return frozenset(
        edge for edge, count in edge_counts.items() if count >= touched_threshold
    )
