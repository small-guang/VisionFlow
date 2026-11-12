"""Local person masks with explicit original-image coordinates.

Only window() materializes missing zero pixels, and only for the requested
window. There is deliberately no implicit conversion to a full-image array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class PersonMask:
    data: np.ndarray
    origin: tuple[int, int]
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 2 or not self.data.size:
            raise ValueError("mask data must be a non-empty two-dimensional array")
        for name, values in (("origin", self.origin), ("image_shape", self.image_shape)):
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ValueError(f"{name} must contain two integers")
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise ValueError(f"{name} must contain two integers")
            object.__setattr__(self, name, tuple(int(value) for value in values))
        x, y = self.origin
        height, width = self.image_shape
        if (
            x < 0 or y < 0 or height <= 0 or width <= 0
            or x + self.data.shape[1] > width
            or y + self.data.shape[0] > height
        ):
            raise ValueError("mask support must lie inside the original image")

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        x, y = self.origin
        return x, y, x + self.data.shape[1], y + self.data.shape[0]

    def _overlap(self, bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bounds
        return max(x1, bounds[0]), max(y1, bounds[1]), min(x2, bounds[2]), min(y2, bounds[3])

    def _view(self, bounds: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bounds
        x, y = self.origin
        return self.data[y1 - y:y2 - y, x1 - x:x2 - x]

    def window(self, bounds: tuple[int, int, int, int]) -> np.ndarray:
        """Return an image-clipped global window, filling absent support with zeros."""
        height, width = self.image_shape
        x1, y1, x2, y2 = bounds
        x1, x2 = (max(0, min(width, int(value))) for value in (x1, x2))
        y1, y2 = (max(0, min(height, int(value))) for value in (y1, y2))
        clipped = (x1, y1, x2, y2)
        left, top, right, bottom = self._overlap(clipped)
        if right > left and bottom > top and (left, top, right, bottom) == clipped:
            return self._view(clipped)
        result = np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=self.data.dtype)
        if right > left and bottom > top:
            result[top - y1:bottom - y1, left - x1:right - x1] = self._view(
                (left, top, right, bottom)
            )
        return result

    def count(self, bounds: tuple[int, int, int, int] | None = None) -> int:
        """Count foreground without allocating a zero-padded window."""
        if bounds is None:
            return int(np.count_nonzero(self.data))
        left, top, right, bottom = self._overlap(bounds)
        if right <= left or bottom <= top:
            return 0
        return int(np.count_nonzero(self._view((left, top, right, bottom))))

    def intersection_count(self, other: PersonMask) -> int:
        if self.image_shape != other.image_shape:
            raise ValueError("mask intersection requires the same image shape")
        bounds = self._overlap(other.bounds)
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return 0
        return int(np.count_nonzero(np.logical_and(self._view(bounds), other._view(bounds))))


def as_person_mask(value: Any, image_shape: tuple[int, ...] | None = None) -> PersonMask | None:
    """Validate a region or wrap a legacy dense image mask without copying it."""
    if value is None:
        return None
    try:
        if isinstance(value, PersonMask):
            result = value
        else:
            data = np.asarray(value)
            if data.ndim != 2 or not data.size:
                return None
            result = PersonMask(data, (0, 0), tuple(data.shape))
        if image_shape is not None and tuple(image_shape[:2]) != result.image_shape:
            return None
        return result
    except (TypeError, ValueError):
        return None

