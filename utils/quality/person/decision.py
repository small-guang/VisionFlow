"""Decision merging and forced-drop behavior for person quality."""

from __future__ import annotations

from typing import Any, Iterable

from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    person_quality_level_from_score,
    person_quality_score_from_mask,
)

from ..quality_reasons import PoseQualityReason, QualityReason
from .models import (
    PersonCropQualityDecision,
    PersonQualityDecision,
    PersonVisualQualityDecision,
)
from .policy import (
    PERSON_ACTION_DISCARD,
    _combined_person_quality_score,
    _deduplicate_reasons,
    person_quality_action_from_level,
)


def merge_person_crop_quality(
    quality: PersonQualityDecision,
    crop: PersonCropQualityDecision,
) -> PersonQualityDecision:
    reasons = _deduplicate_reasons([*quality.quality_reasons, *crop.quality_reasons])
    crop_score = min(quality.crop_score, crop.crop_score)

    if crop.reject:
        final_score = 0.0
        final_level = QUALITY_LEVEL_DROP
        crop_score = 0.0
    elif quality.person_quality_level == QUALITY_LEVEL_DROP:
        final_score = 0.0
        final_level = QUALITY_LEVEL_DROP
    else:
        final_score = _combined_person_quality_score(
            quality.pose_score,
            quality.mask_score,
            crop_score,
        )
        final_level = min(
            quality.person_quality_level,
            crop.cap_level,
            person_quality_level_from_score(final_score),
        )
    return PersonQualityDecision(
        person_quality_score=float(final_score),
        person_quality_level=int(final_level),
        person_quality_action=person_quality_action_from_level(final_level),
        mask_score=quality.mask_score,
        crop_score=float(crop_score),
        quality_reasons=reasons,
        pose_score=quality.pose_score,
        visible_keypoint_count=quality.visible_keypoint_count,
        torso_keypoint_count=quality.torso_keypoint_count,
        shoulder_hip_chain_count=quality.shoulder_hip_chain_count,
        half_body_pose_eligible=quality.half_body_pose_eligible,
        body_completeness=quality.body_completeness,
        is_half_body=quality.is_half_body,
        is_truncated=quality.is_truncated,
        has_second_person=(
            quality.has_second_person or crop.has_second_person
        ),
        foreign_person_ratio=crop.metrics.foreign_person_ratio,
        foreign_person_core_ratio=crop.metrics.foreign_person_core_ratio,
        foreign_person_pixel_count=crop.metrics.foreign_person_pixel_count,
        foreign_person_core_pixel_count=(
            crop.metrics.foreign_person_core_pixel_count
        ),
        foreign_person_bbox_fallback_ratio=(
            crop.metrics.foreign_person_bbox_fallback_ratio
        ),
        foreign_person_bbox_fallback_pixel_count=(
            crop.metrics.foreign_person_bbox_fallback_pixel_count
        ),
        foreign_person_edge_dominant=(
            crop.metrics.foreign_person_edge_dominant
        ),
        foreign_person_evidence_source=(
            crop.metrics.foreign_person_evidence_source
        ),
        padding_ratio=crop.metrics.padding_ratio,
        letterbox_padding_ratio=crop.metrics.letterbox_padding_ratio,
        source_overflow_ratio=crop.metrics.source_overflow_ratio,
        bbox_overflow_ratio=quality.bbox_overflow_ratio,
        near_image_edge_count=quality.near_image_edge_count,
        truncated_edge_count=quality.truncated_edge_count,
    )

def merge_person_visual_quality(
    quality: PersonQualityDecision,
    visual: PersonVisualQualityDecision,
) -> PersonQualityDecision:
    reasons = _deduplicate_reasons(
        [*quality.quality_reasons, *visual.quality_reasons]
    )
    if visual.reject:
        final_score = 0.0
        final_level = QUALITY_LEVEL_DROP
        crop_score = 0.0
    else:
        crop_score = min(quality.crop_score, visual.crop_score)
        final_score = _combined_person_quality_score(
            quality.pose_score,
            quality.mask_score,
            crop_score,
        )
        final_level = min(
            quality.person_quality_level,
            visual.cap_level,
            person_quality_level_from_score(final_score),
        )
    return PersonQualityDecision(
        person_quality_score=float(final_score),
        person_quality_level=int(final_level),
        person_quality_action=person_quality_action_from_level(final_level),
        mask_score=quality.mask_score,
        crop_score=float(crop_score),
        quality_reasons=reasons,
        pose_score=quality.pose_score,
        visible_keypoint_count=quality.visible_keypoint_count,
        torso_keypoint_count=quality.torso_keypoint_count,
        shoulder_hip_chain_count=quality.shoulder_hip_chain_count,
        half_body_pose_eligible=quality.half_body_pose_eligible,
        body_completeness=quality.body_completeness,
        is_half_body=quality.is_half_body,
        is_truncated=quality.is_truncated,
        has_second_person=quality.has_second_person,
        foreign_person_ratio=quality.foreign_person_ratio,
        foreign_person_core_ratio=quality.foreign_person_core_ratio,
        foreign_person_pixel_count=quality.foreign_person_pixel_count,
        foreign_person_core_pixel_count=(
            quality.foreign_person_core_pixel_count
        ),
        foreign_person_bbox_fallback_ratio=(
            quality.foreign_person_bbox_fallback_ratio
        ),
        foreign_person_bbox_fallback_pixel_count=(
            quality.foreign_person_bbox_fallback_pixel_count
        ),
        foreign_person_edge_dominant=quality.foreign_person_edge_dominant,
        foreign_person_evidence_source=quality.foreign_person_evidence_source,
        padding_ratio=quality.padding_ratio,
        letterbox_padding_ratio=quality.letterbox_padding_ratio,
        source_overflow_ratio=quality.source_overflow_ratio,
        bbox_overflow_ratio=quality.bbox_overflow_ratio,
        near_image_edge_count=quality.near_image_edge_count,
        truncated_edge_count=quality.truncated_edge_count,
    )

def drop_person_quality_decision(
    reason: QualityReason | PoseQualityReason,
    *,
    mask_pixel_count: Any = 0,
    extra_reasons: Iterable[QualityReason | PoseQualityReason] | None = None,
    base_decision: PersonQualityDecision | None = None,
) -> PersonQualityDecision:
    reasons = [reason]
    if base_decision is not None:
        reasons.extend(base_decision.quality_reasons)
    if extra_reasons:
        reasons.extend(extra_reasons)
    reasons = _deduplicate_reasons(reasons)
    return PersonQualityDecision(
        person_quality_score=0.0,
        person_quality_level=QUALITY_LEVEL_DROP,
        person_quality_action=PERSON_ACTION_DISCARD,
        mask_score=(
            base_decision.mask_score
            if base_decision is not None
            else person_quality_score_from_mask(mask_pixel_count)
        ),
        crop_score=0.0,
        quality_reasons=reasons,
        pose_score=base_decision.pose_score if base_decision is not None else 0.0,
        visible_keypoint_count=(
            base_decision.visible_keypoint_count if base_decision is not None else 0
        ),
        torso_keypoint_count=(
            base_decision.torso_keypoint_count if base_decision is not None else 0
        ),
        shoulder_hip_chain_count=(
            base_decision.shoulder_hip_chain_count
            if base_decision is not None
            else 0
        ),
        half_body_pose_eligible=(
            base_decision.half_body_pose_eligible
            if base_decision is not None
            else False
        ),
        body_completeness=(
            base_decision.body_completeness
            if base_decision is not None
            else "unknown"
        ),
        is_half_body=(
            base_decision.is_half_body if base_decision is not None else False
        ),
        is_truncated=(
            base_decision.is_truncated if base_decision is not None else False
        ),
        has_second_person=(
            base_decision.has_second_person if base_decision is not None else False
        ),
        foreign_person_ratio=(
            base_decision.foreign_person_ratio if base_decision is not None else 0.0
        ),
        foreign_person_core_ratio=(
            base_decision.foreign_person_core_ratio
            if base_decision is not None
            else 0.0
        ),
        foreign_person_pixel_count=(
            base_decision.foreign_person_pixel_count
            if base_decision is not None
            else 0
        ),
        foreign_person_core_pixel_count=(
            base_decision.foreign_person_core_pixel_count
            if base_decision is not None
            else 0
        ),
        foreign_person_bbox_fallback_ratio=(
            base_decision.foreign_person_bbox_fallback_ratio
            if base_decision is not None
            else 0.0
        ),
        foreign_person_bbox_fallback_pixel_count=(
            base_decision.foreign_person_bbox_fallback_pixel_count
            if base_decision is not None
            else 0
        ),
        foreign_person_edge_dominant=(
            base_decision.foreign_person_edge_dominant
            if base_decision is not None
            else False
        ),
        foreign_person_evidence_source=(
            base_decision.foreign_person_evidence_source
            if base_decision is not None
            else "none"
        ),
        padding_ratio=(
            base_decision.padding_ratio if base_decision is not None else 0.0
        ),
        letterbox_padding_ratio=(
            base_decision.letterbox_padding_ratio
            if base_decision is not None
            else 0.0
        ),
        source_overflow_ratio=(
            base_decision.source_overflow_ratio
            if base_decision is not None
            else 0.0
        ),
        bbox_overflow_ratio=(
            base_decision.bbox_overflow_ratio if base_decision is not None else 0.0
        ),
        near_image_edge_count=(
            base_decision.near_image_edge_count if base_decision is not None else 0
        ),
        truncated_edge_count=(
            base_decision.truncated_edge_count if base_decision is not None else 0
        ),
    )
