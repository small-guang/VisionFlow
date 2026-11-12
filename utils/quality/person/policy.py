"""Shared scoring and action policy for person quality."""

from __future__ import annotations

from typing import Iterable

from core.algorithm_config import algorithm_config
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
)

from ..quality_reasons import PoseQualityReason, QualityReason


PERSON_ACTION_DISCARD = "discard"
PERSON_ACTION_IDENTIFY_ONLY = "identify_only"
PERSON_ACTION_BUILD_GALLERY = "build_gallery"

def _combined_person_quality_score(
    pose_score: float,
    mask_score: float,
    crop_score: float,
) -> float:
    general_config = algorithm_config.person.quality.general
    return float(
        general_config.pose_weight
        * max(0.0, min(float(pose_score), general_config.score_max))
        + general_config.mask_weight
        * max(0.0, min(float(mask_score), general_config.score_max))
        + general_config.crop_weight
        * max(0.0, min(float(crop_score), general_config.score_max))
    )

def person_quality_action_from_level(level: int) -> str:
    if level >= QUALITY_LEVEL_ENROLL:
        return PERSON_ACTION_BUILD_GALLERY
    if level >= QUALITY_LEVEL_RECOGNITION:
        return PERSON_ACTION_IDENTIFY_ONLY
    return PERSON_ACTION_DISCARD

def _deduplicate_reasons(
    reasons: Iterable[QualityReason | PoseQualityReason],
) -> list[QualityReason | PoseQualityReason]:
    unique: list[QualityReason | PoseQualityReason] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return unique
