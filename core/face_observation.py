from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np

from utils.quality.face_orientation import normalize_face_orientation


class FaceProcessingStage(str, Enum):
    DETECTION = "detection"
    LANDMARKS = "landmarks"
    QUALITY = "quality"
    FEATURE = "feature"
    IDENTITY = "identity"


def is_supported_face_id_scalar(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return not isinstance(value, bool) and isinstance(value, Integral)


def normalize_face_id_scalar(value: object) -> str:
    """Normalize the supported raw face-ID scalar contract to a string ID."""
    if not is_supported_face_id_scalar(value):
        return "0"
    if isinstance(value, str):
        return value.strip()
    return str(int(value))


def _readonly_copy(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    copied = np.ascontiguousarray(np.asarray(value, dtype=np.float32)).copy()
    copied.setflags(write=False)
    return copied


def _validated_rect(value: tuple[int, ...], name: str) -> tuple[int, int, int, int]:
    try:
        if len(value) != 4:
            raise ValueError(f"{name} must contain four values")
        return tuple(int(v) for v in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain four values") from exc


@dataclass(frozen=True)
class FrameFaceObservation:
    source_rect: tuple[int, int, int, int]
    confidence: float
    expanded_source_rect: tuple[int, int, int, int] | None = None
    source_landmarks: np.ndarray | None = None
    orientation: str = "unknown"
    quality_score: float | None = None
    normalized_embedding: np.ndarray | None = None
    raw_embedding_norm: float | None = None
    assigned_face_id: str = "0"
    recognition_score: float = 0.0
    completed_stage: FaceProcessingStage = FaceProcessingStage.DETECTION
    failure_reason: str = ""

    @property
    def assigned_face_id_is_valid(self) -> bool:
        return bool(getattr(self, "_assigned_face_id_is_valid", False))

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_rect", _validated_rect(self.source_rect, "source_rect"))
        if self.expanded_source_rect is not None:
            object.__setattr__(
                self,
                "expanded_source_rect",
                _validated_rect(self.expanded_source_rect, "expanded_source_rect"),
            )
        object.__setattr__(self, "source_landmarks", _readonly_copy(self.source_landmarks))
        embedding = _readonly_copy(self.normalized_embedding)
        if embedding is not None:
            if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
                raise ValueError("normalized_embedding must be a finite one-dimensional vector")
        object.__setattr__(self, "normalized_embedding", embedding)
        if self.raw_embedding_norm is not None:
            try:
                norm = float(self.raw_embedding_norm)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("raw_embedding_norm must be finite") from exc
            if not math.isfinite(norm):
                raise ValueError("raw_embedding_norm must be finite")
            object.__setattr__(self, "raw_embedding_norm", norm)
        object.__setattr__(self, "orientation", normalize_face_orientation(self.orientation))
        raw_assigned_face_id = self.assigned_face_id
        object.__setattr__(
            self,
            "_assigned_face_id_is_valid",
            is_supported_face_id_scalar(raw_assigned_face_id),
        )
        object.__setattr__(
            self,
            "assigned_face_id",
            normalize_face_id_scalar(raw_assigned_face_id),
        )


def landmarks_in_expanded_crop(
    observation: FrameFaceObservation,
) -> np.ndarray | None:
    if observation.source_landmarks is None or observation.expanded_source_rect is None:
        return None
    points = np.asarray(observation.source_landmarks, dtype=np.float32)
    if points.shape != (5, 2):
        return None
    x1, y1, x2, y2 = observation.expanded_source_rect
    if x2 <= x1 or y2 <= y1:
        return None
    translated = points.copy()
    translated[:, 0] -= x1
    translated[:, 1] -= y1
    return translated
