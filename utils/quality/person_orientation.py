"""Pure COCO-17 torso-orientation classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from core.algorithm_config import algorithm_config
from utils.keyframe.person_pose_geometry import estimate_person_torso_axis


class PersonOrientation(str, Enum):
    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    UNKNOWN = "unknown"


PERSON_ORIENTATION_VALUES = frozenset(item.value for item in PersonOrientation)


def normalize_person_orientation(value: object) -> str:
    """Return a canonical public label, or ``unknown`` for invalid input."""
    if isinstance(value, PersonOrientation):
        return value.value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in PERSON_ORIENTATION_VALUES:
            return normalized
    return PersonOrientation.UNKNOWN.value


@dataclass(frozen=True)
class PersonOrientationResult:
    orientation: PersonOrientation = PersonOrientation.UNKNOWN
    signed_span_ratio: Optional[float] = None
    shoulder_span_ratio: Optional[float] = None
    hip_span_ratio: Optional[float] = None
    evidence_count: int = 0
    reason: str = "classified"

    @property
    def label(self) -> str:
        return normalize_person_orientation(self.orientation)


def _unknown(
    reason: str,
    *,
    shoulder_span_ratio: Optional[float] = None,
    hip_span_ratio: Optional[float] = None,
    evidence_count: int = 0,
    signed_span_ratio: Optional[float] = None,
) -> PersonOrientationResult:
    return PersonOrientationResult(
        orientation=PersonOrientation.UNKNOWN,
        signed_span_ratio=signed_span_ratio,
        shoulder_span_ratio=shoulder_span_ratio,
        hip_span_ratio=hip_span_ratio,
        evidence_count=evidence_count,
        reason=reason,
    )


def _image_dimensions(image_shape: Any) -> Optional[tuple[float, float]]:
    try:
        if image_shape is None or len(image_shape) < 2:
            return None
        height = float(image_shape[0])
        width = float(image_shape[1])
    except (TypeError, ValueError, OverflowError, IndexError):
        return None
    if not math.isfinite(height) or not math.isfinite(width):
        return None
    if height <= 0.0 or width <= 0.0:
        return None
    return height, width


def _visible_points(keypoints: Any, image_shape: Any) -> Optional[np.ndarray]:
    """Copy keypoints and hide every point that cannot be trusted."""
    try:
        points = np.asarray(keypoints, dtype=np.float64)
    except Exception:
        return None
    if points.shape != (17, 3):
        return None
    dimensions = _image_dimensions(image_shape)
    if dimensions is None:
        return None
    height, width = dimensions
    sanitized = points.copy()
    for index, point in enumerate(sanitized):
        x, y, confidence = (float(value) for value in point)
        if not (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(confidence)
            and 0.0 <= x < width
            and 0.0 <= y < height
            and confidence >= algorithm_config.person.pose_geometry.torso_visible_threshold
        ):
            sanitized[index] = np.nan
    return sanitized


def _pair_ratio(
    points: np.ndarray,
    left_index: int,
    right_index: int,
    local_right: np.ndarray,
    torso_scale: float,
) -> Optional[tuple[float, float]]:
    left = points[left_index]
    right = points[right_index]
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            span = left[:2] - right[:2]
            numerator = float(np.dot(span, local_right))
        weight = float(min(left[2], right[2]))
        ratio = numerator / float(torso_scale)
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None
    if not math.isfinite(ratio) or not math.isfinite(weight) or weight <= 0.0:
        return None
    return ratio, weight


def estimate_person_orientation(
    keypoints: Any,
    rect: Any,
    image_shape: Any,
) -> PersonOrientationResult:
    """Estimate front/side/back from reliable shoulder and hip spans.

    The function is deliberately fail-closed: malformed or numerically
    unstable data returns an ``UNKNOWN`` result and never mutates its input.
    """
    side_span_ratio = algorithm_config.person.orientation.side_span_ratio
    try:
        points = _visible_points(keypoints, image_shape)
        if points is None:
            return _unknown("pose_axis_missing")

        axis_result = estimate_person_torso_axis(points, rect)
        if not axis_result.valid:
            return _unknown(axis_result.reason)
        try:
            down_axis = np.asarray(axis_result.down_axis, dtype=np.float64)
            torso_scale = float(axis_result.torso_scale)
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return _unknown("invalid_score")
        if (
            down_axis.shape != (2,)
            or not np.isfinite(down_axis).all()
            or not math.isfinite(torso_scale)
            or torso_scale <= 0.0
        ):
            return _unknown("invalid_score")
        try:
            local_right = np.asarray((down_axis[1], -down_axis[0]), dtype=np.float64)
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return _unknown("invalid_score")
        if not np.isfinite(local_right).all():
            return _unknown("invalid_score")

        pair_values: list[tuple[str, float, float]] = []
        invalid_pair = False
        for name, left_index, right_index in (
            ("shoulder", 5, 6),
            ("hip", 11, 12),
        ):
            endpoints_visible = np.isfinite(points[[left_index, right_index]]).all()
            if not endpoints_visible:
                continue
            pair = _pair_ratio(
                points, left_index, right_index, local_right, torso_scale
            )
            if pair is None:
                invalid_pair = True
                continue
            ratio, weight = pair
            pair_values.append((name, ratio, weight))

        if invalid_pair:
            return _unknown(
                "invalid_score",
                shoulder_span_ratio=next(
                    (ratio for name, ratio, _ in pair_values if name == "shoulder"),
                    None,
                ),
                hip_span_ratio=next(
                    (ratio for name, ratio, _ in pair_values if name == "hip"),
                    None,
                ),
                evidence_count=len(pair_values),
            )
        if not pair_values:
            return _unknown("horizontal_pair_missing")

        shoulder_ratio = next(
            (ratio for name, ratio, _ in pair_values if name == "shoulder"), None
        )
        hip_ratio = next(
            (ratio for name, ratio, _ in pair_values if name == "hip"), None
        )
        if len(pair_values) == 2:
            first_ratio, second_ratio = pair_values[0][1], pair_values[1][1]
            if (
                abs(first_ratio) > side_span_ratio
                and abs(second_ratio) > side_span_ratio
                and first_ratio * second_ratio < 0.0
            ):
                return _unknown(
                    "pair_conflict",
                    shoulder_span_ratio=shoulder_ratio,
                    hip_span_ratio=hip_ratio,
                    evidence_count=2,
                )

        maximum_weight = max(weight for _, _, weight in pair_values)
        if not math.isfinite(maximum_weight) or maximum_weight <= 0.0:
            return _unknown(
                "invalid_score",
                shoulder_span_ratio=shoulder_ratio,
                hip_span_ratio=hip_ratio,
                evidence_count=len(pair_values),
            )
        scaled_weights = [weight / maximum_weight for _, _, weight in pair_values]
        total_scaled_weight = math.fsum(scaled_weights)
        score = math.fsum(
            ratio * scaled_weight
            for (_, ratio, _), scaled_weight in zip(pair_values, scaled_weights)
        ) / total_scaled_weight
        if not math.isfinite(score):
            return _unknown(
                "invalid_score",
                shoulder_span_ratio=shoulder_ratio,
                hip_span_ratio=hip_ratio,
                evidence_count=len(pair_values),
            )
        if abs(score) <= side_span_ratio:
            orientation = PersonOrientation.SIDE
        elif score > 0.0:
            orientation = PersonOrientation.FRONT
        else:
            orientation = PersonOrientation.BACK
        return PersonOrientationResult(
            orientation=orientation,
            signed_span_ratio=float(score),
            shoulder_span_ratio=shoulder_ratio,
            hip_span_ratio=hip_ratio,
            evidence_count=len(pair_values),
            reason="classified",
        )
    except Exception:
        # Expected malformed/numeric input is handled at each stage above and
        # retains its specific reason.  This final guard contains only
        # genuinely unexpected failures from untrusted inputs or dependencies.
        return _unknown("internal_error")


__all__ = [
    "PERSON_ORIENTATION_VALUES",
    "PersonOrientation",
    "PersonOrientationResult",
    "estimate_person_orientation",
    "normalize_person_orientation",
]
