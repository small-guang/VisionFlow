"""Pure face-orientation classification built on head-pose angles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

from core.algorithm_config import algorithm_config
from .face_quality import calculate_pose_from_landmarks


class FaceOrientation(str, Enum):
    FRONT = "front"
    SIDE = "side"
    NO_FACE = "no_face"
    UNKNOWN = "unknown"


FACE_ORIENTATION_VALUES = frozenset(item.value for item in FaceOrientation)


def normalize_face_orientation(value: object) -> str:
    """Return a canonical public orientation label, or ``unknown``."""
    if isinstance(value, FaceOrientation):
        return value.value
    if isinstance(value, str):
        label = value.strip().casefold()
        if label in FACE_ORIENTATION_VALUES:
            return label
    return FaceOrientation.UNKNOWN.value


def _finite_float(value: object) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def classify_face_orientation(
    pitch: object,
    yaw: object,
    roll: object,
    side_yaw_deg: float = algorithm_config.face.orientation.side_yaw_degrees,
) -> FaceOrientation:
    """Classify a pose using yaw; pitch and roll are validated but not labels.

    A zero threshold is valid and classifies every finite yaw as ``SIDE``.
    """
    threshold = _finite_float(side_yaw_deg)
    if threshold is None or threshold < 0:
        raise ValueError("side_yaw_deg must be a finite, non-negative number")
    pitch_value = _finite_float(pitch)
    yaw_value = _finite_float(yaw)
    roll_value = _finite_float(roll)
    if pitch_value is None or yaw_value is None or roll_value is None:
        return FaceOrientation.UNKNOWN
    return FaceOrientation.FRONT if abs(yaw_value) < threshold else FaceOrientation.SIDE


@dataclass(frozen=True)
class FaceOrientationResult:
    orientation: FaceOrientation
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None

    @property
    def label(self) -> str:
        return self.orientation.value


def estimate_face_orientation(
    landmarks: Optional[Sequence[Sequence[float]]],
    image_shape: Optional[Tuple[int, ...]],
) -> FaceOrientationResult:
    """Estimate pose safely and classify valid three-angle results."""
    if landmarks is None:
        return FaceOrientationResult(FaceOrientation.UNKNOWN)
    try:
        pose = calculate_pose_from_landmarks(landmarks, image_shape)
    except Exception:
        return FaceOrientationResult(FaceOrientation.UNKNOWN)
    if pose is None:
        return FaceOrientationResult(FaceOrientation.UNKNOWN)
    try:
        if len(pose) != 3:
            return FaceOrientationResult(FaceOrientation.UNKNOWN)
        angles = tuple(_finite_float(value) for value in pose)
    except (TypeError, ValueError, OverflowError):
        return FaceOrientationResult(FaceOrientation.UNKNOWN)
    if any(value is None for value in angles):
        return FaceOrientationResult(FaceOrientation.UNKNOWN)
    pitch, yaw, roll = angles
    orientation = classify_face_orientation(pitch, yaw, roll)
    return FaceOrientationResult(orientation, pitch, yaw, roll)


__all__ = [
    "FACE_ORIENTATION_VALUES",
    "FaceOrientation",
    "FaceOrientationResult",
    "classify_face_orientation",
    "estimate_face_orientation",
    "normalize_face_orientation",
]
