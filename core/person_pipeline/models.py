"""Immutable values shared by person-pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from utils.keyframe.person_crop import PersonCropResult


@dataclass(frozen=True)
class PersonPipelineRuntime:
    encode_image: Callable[[np.ndarray], bytes | None]
    evaluate_visual_quality: Callable[[np.ndarray, np.ndarray | None], Any]
    match_face: Callable[..., Any]
    pose_keypoints_to_crop: Callable[[Any, "PersonCropResult"], Any]


@dataclass(frozen=True)
class PersonObservation:
    segment_index: int
    segment: dict[str, Any]
    people_rect: dict[str, int]
    mask_pixel_count: int
    confidence: float
    person_keypoints: Any
    person_orientation_result: Any
    quality_decision: Any
    ai_id: str
    detect_time: int

    @property
    def person_orientation(self) -> str:
        try:
            label = self.person_orientation_result.label
        except (AttributeError, TypeError):
            return "unknown"
        return label if isinstance(label, str) else "unknown"


@dataclass(frozen=True)
class PreparedPerson:
    observation: PersonObservation
    crop_result: Any
    quality_decision: Any
    crop_info: str
    visual_info: str = ""
    image_bytes: bytes | None = None
    crop_path: str = ""

    @property
    def cropped(self) -> np.ndarray:
        return self.crop_result.image

    @property
    def mask_crop(self) -> np.ndarray | None:
        return self.crop_result.mask


@dataclass(frozen=True)
class PersonEmbeddingBundle:
    sbs: np.ndarray | None = None
    clip: np.ndarray | None = None
    sbs_failure: str = ""
    sbs_failure_reason: str = ""
    clip_failure: str = ""


@dataclass(frozen=True)
class FaceMatchOutcome:
    face_id: str
    info: str
    orientation: str = "unknown"
    source: str = "local_fallback"
    fallback_reason: str = ""
    shared_face_id: str = "0"
    shared_orientation: str = "unknown"
    shared_source: str = ""
    shared_fallback_reason: str = ""


@dataclass(frozen=True)
class PersonAnalysis:
    """Per-person preparation and inference, awaiting ordered identity submission."""

    prepared: PreparedPerson
    face: FaceMatchOutcome
    embeddings: PersonEmbeddingBundle | None = None


@dataclass(frozen=True)
class PersonPreparationFailure:
    observation: PersonObservation
    info: str
