"""Primary structural evaluator for person quality."""

from __future__ import annotations

from typing import Any

from core.algorithm_config import algorithm_config
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
    person_quality_level_from_score,
    person_quality_score_from_mask,
)
from utils.reid.person_body import (
    BODY_HALF_LOWER,
    BODY_HALF_UPPER,
    BODY_PARTIAL,
    BODY_THREE_QUARTER_LOWER,
)

from ..quality_reasons import PoseQualityReason, QualityReason
from .geometry import (
    _image_edge_contact_directions,
    _image_edge_touch_count,
    _image_hw,
    _max_other_person_overlap_ratio,
    _raw_bbox_overflow,
    _rect_area,
    _safe_float,
    _safe_int,
)
from .mask import (
    _connected_component_areas,
    _mask_bbox_edge_stats,
    _mask_image_edge_directions,
    _valid_mask,
)
from .models import PersonPoseMetrics, PersonQualityDecision
from .policy import (
    _combined_person_quality_score,
    person_quality_action_from_level,
)
from .pose import _invalid_pose_metrics, calculate_person_pose_metrics


def _is_half_upper_pose_gallery_eligible(
    pose_metrics: PersonPoseMetrics,
) -> bool:
    """Return whether an upper-half pose is structurally safe for gallery use."""
    return bool(
        pose_metrics.body_completeness == BODY_HALF_UPPER
        and pose_metrics.visible_keypoint_count
        >= algorithm_config.person.quality.edge.half_upper_gallery_min_visible_keypoints
        and pose_metrics.torso_keypoint_count == 4
        and pose_metrics.shoulder_hip_chain_count == 2
    )


def evaluate_person_quality(
    segment: dict[str, Any],
    all_segments: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    *,
    segment_index: int | None = None,
) -> PersonQualityDecision:
    general_config = algorithm_config.person.quality.general
    edge_config = algorithm_config.person.quality.edge
    pose_config = algorithm_config.person.quality.pose
    rect = segment.get("rect") or {}
    img_h, img_w = _image_hw(image_shape)
    bbox_area = _rect_area(rect)
    image_area = max(1, img_h * img_w)
    mask_pixel_count = segment.get("maskPixelCount")
    mask_score = person_quality_score_from_mask(mask_pixel_count)
    crop_score = general_config.score_max
    cap_level = QUALITY_LEVEL_ENROLL
    reject = False
    reasons: list[QualityReason | PoseQualityReason] = []
    near_image_edge_count = 0
    bbox_overflow_ratio = 0.0
    bbox_contact_edges: frozenset[str] = frozenset()
    raw_overflow_edges: frozenset[str] = frozenset()
    confirmed_truncated_edges: frozenset[str] = frozenset()

    def add_reject(
        reason: QualityReason | PoseQualityReason, *, crop: bool = False
    ) -> None:
        nonlocal reject, crop_score
        if reason not in reasons:
            reasons.append(reason)
        reject = True
        if crop:
            crop_score = 0.0

    def add_reason(reason: QualityReason | PoseQualityReason) -> None:
        if reason not in reasons:
            reasons.append(reason)

    def add_cap(
        reason: QualityReason | PoseQualityReason, *, crop: bool = False
    ) -> None:
        nonlocal cap_level, crop_score
        if reason not in reasons:
            reasons.append(reason)
        cap_level = min(cap_level, QUALITY_LEVEL_RECOGNITION)
        if crop:
            crop_score = min(
                crop_score,
                general_config.enroll_min_score - general_config.recognition_score_gap,
            )

    mask = _valid_mask(segment.get("mask"), img_h, img_w)
    pose = segment.get("pose")
    pose_metrics = _invalid_pose_metrics()
    half_upper_gallery_pose = False
    if segment.get("poseInferenceFailed"):
        add_reject(PoseQualityReason.POSE_INFERENCE_FAILED)
    elif pose is None:
        add_reject(PoseQualityReason.POSE_NOT_MATCHED)
    else:
        pose_metrics = calculate_person_pose_metrics(
            pose.get("keypoints"),
            rect,
            image_shape,
            mask,
        )
        if not pose_metrics.valid:
            add_reject(PoseQualityReason.POSE_KEYPOINTS_INVALID)
        else:
            half_upper_gallery_pose = _is_half_upper_pose_gallery_eligible(
                pose_metrics
            )
            if (
                pose_metrics.visible_keypoint_count
                < pose_config.min_visible_keypoints
            ):
                add_reject(PoseQualityReason.POSE_VISIBLE_KEYPOINTS_TOO_FEW)
            if pose_metrics.torso_keypoint_count < pose_config.min_torso_keypoints:
                add_reject(PoseQualityReason.POSE_TORSO_INCOMPLETE)
            if pose_metrics.shoulder_hip_chain_count == 0:
                add_cap(PoseQualityReason.POSE_SHOULDER_HIP_CHAIN_INCOMPLETE)
            if pose_metrics.hip_knee_chain_count == 0:
                if half_upper_gallery_pose:
                    add_reason(PoseQualityReason.POSE_LOWER_BODY_INCOMPLETE)
                else:
                    add_cap(PoseQualityReason.POSE_LOWER_BODY_INCOMPLETE)
            if pose_metrics.body_completeness == BODY_THREE_QUARTER_LOWER:
                add_cap(PoseQualityReason.POSE_HEAD_INCOMPLETE)
            if pose_metrics.body_completeness == BODY_HALF_UPPER:
                if half_upper_gallery_pose:
                    add_reason(PoseQualityReason.POSE_HALF_BODY)
                else:
                    add_cap(PoseQualityReason.POSE_HALF_BODY)
            if pose_metrics.body_completeness in {BODY_HALF_LOWER, BODY_PARTIAL}:
                add_reject(PoseQualityReason.POSE_BODY_TYPE_NOT_SEARCHABLE)
            if (
                pose_metrics.vertical_span_ratio
                < pose_config.min_vertical_span_ratio
            ):
                add_cap(PoseQualityReason.POSE_GEOMETRY_UNSTABLE)
            if pose_metrics.bbox_agreement < pose_config.min_bbox_agreement:
                add_cap(PoseQualityReason.POSE_KEYPOINTS_OUTSIDE_BBOX)
            if (
                pose_metrics.mask_agreement is not None
                and pose_metrics.mask_agreement < pose_config.min_mask_agreement
            ):
                add_cap(PoseQualityReason.POSE_MASK_MISMATCH)

    if mask_score < general_config.recognition_min_score:
        add_reject(QualityReason.MASK_AREA_TOO_SMALL)
    elif mask_score < general_config.enroll_min_score:
        add_cap(QualityReason.MASK_AREA_BELOW_ENROLL)

    confidence = _safe_float(segment.get("conf"))
    if confidence is None or confidence <= 0:
        add_reject(QualityReason.INVALID_CONFIDENCE, crop=True)

    if img_h <= 0 or img_w <= 0:
        add_reject(QualityReason.INVALID_IMAGE_SHAPE, crop=True)

    w = _safe_int(rect.get("w"))
    h = _safe_int(rect.get("h"))
    if bbox_area <= 0 or w <= 0 or h <= 0:
        add_reject(QualityReason.INVALID_BBOX, crop=True)
    else:
        bbox_image_ratio = bbox_area / image_area
        aspect_ratio = h / max(w, 1)
        if w < edge_config.min_bbox_width_px or h < edge_config.min_bbox_height_px:
            add_reject(QualityReason.BBOX_TOO_SMALL, crop=True)
        if bbox_image_ratio < edge_config.min_bbox_image_ratio:
            add_reject(QualityReason.BBOX_AREA_TOO_SMALL, crop=True)
        if bbox_image_ratio > edge_config.max_bbox_image_ratio:
            add_cap(QualityReason.BBOX_AREA_TOO_LARGE, crop=True)
        if (
            aspect_ratio < edge_config.min_bbox_aspect_ratio
            or aspect_ratio > edge_config.max_bbox_aspect_ratio
        ):
            add_cap(QualityReason.BBOX_ASPECT_RATIO_ABNORMAL, crop=True)
        near_image_edge_count = _image_edge_touch_count(rect, img_w, img_h)
        if near_image_edge_count > 0:
            add_reason(QualityReason.BBOX_NEAR_IMAGE_EDGE)
        bbox_contact_edges = _image_edge_contact_directions(rect, img_w, img_h)
        bbox_overflow_ratio, raw_overflow_edges = _raw_bbox_overflow(
            segment.get("xyxy"), img_w, img_h
        )
        overlap_ratio = _max_other_person_overlap_ratio(
            rect, all_segments, segment_index=segment_index
        )
        if overlap_ratio > edge_config.other_person_overlap_reject_ratio:
            add_reject(
                QualityReason.BBOX_HEAVILY_OVERLAPS_OTHER_PERSON,
                crop=True,
            )
        elif overlap_ratio > edge_config.other_person_overlap_cap_ratio:
            add_cap(QualityReason.BBOX_OVERLAPS_OTHER_PERSON, crop=True)

    if mask is None:
        if segment.get("mask") is None:
            reasons.append(QualityReason.MASK_MISSING)
        else:
            reasons.append(QualityReason.MASK_SHAPE_INVALID)
    elif bbox_area > 0:
        mask_area = mask.count()
        mask_bbox_ratio = mask_area / max(1, bbox_area)
        if mask_bbox_ratio < edge_config.mask_bbox_reject_ratio:
            add_reject(QualityReason.MASK_BBOX_RATIO_TOO_LOW)
        elif mask_bbox_ratio < edge_config.mask_bbox_cap_ratio:
            add_cap(QualityReason.MASK_BBOX_RATIO_LOW)

        component_areas = _connected_component_areas(mask)
        if component_areas:
            max_component_area = max(component_areas)
            if (
                max_component_area / max(mask_area, 1)
                < edge_config.largest_component_min_ratio
            ):
                add_reject(QualityReason.MASK_FRAGMENTED)
            large_component_threshold = max(
                edge_config.large_component_min_pixels,
                int(round(mask_area * edge_config.large_component_ratio)),
            )
            large_component_count = sum(
                area >= large_component_threshold for area in component_areas
            )
            if large_component_count >= edge_config.fragmented_component_count:
                add_cap(QualityReason.MASK_MULTIPLE_COMPONENTS)

        edge_ratio, touched_edges = _mask_bbox_edge_stats(mask, rect)
        if (
            touched_edges >= edge_config.mask_edge_reject_count
            and edge_ratio >= edge_config.mask_edge_reject_ratio
        ):
            add_reason(QualityReason.MASK_TOUCHING_BBOX_EDGE_SEVERE)
        elif (
            touched_edges >= edge_config.mask_edge_cap_count
            and edge_ratio >= edge_config.mask_edge_cap_ratio
        ):
            add_reason(QualityReason.MASK_TOUCHING_BBOX_EDGE)
        elif touched_edges >= 1 or edge_ratio >= edge_config.mask_edge_cap_ratio:
            add_reason(QualityReason.MASK_TOUCHING_BBOX_EDGE)

        mask_image_edges = _mask_image_edge_directions(mask)
        bbox_evidence_edges = bbox_contact_edges | raw_overflow_edges
        confirmed = set()
        for edge in bbox_evidence_edges & mask_image_edges:
            if edge in raw_overflow_edges:
                confirmed.add(edge)
            elif edge == "top" and not pose_metrics.head_visible:
                confirmed.add(edge)
            elif edge == "bottom" and pose_metrics.full_body_chain_count == 0:
                confirmed.add(edge)
        confirmed_truncated_edges = frozenset(confirmed)

    if confirmed_truncated_edges:
        add_reason(QualityReason.PERSON_TRUNCATED)
        if len(confirmed_truncated_edges) >= edge_config.truncated_edge_reject_count:
            add_reject(QualityReason.PERSON_TRUNCATED_SEVERE, crop=True)
        if bbox_overflow_ratio > edge_config.bbox_overflow_reject_ratio:
            add_reject(QualityReason.BBOX_CONTENT_OVERFLOW_TOO_HIGH, crop=True)
        elif bbox_overflow_ratio > edge_config.bbox_overflow_cap_ratio:
            add_cap(QualityReason.BBOX_CONTENT_OVERFLOW_HIGH, crop=True)

    if reject:
        final_score = 0.0
        final_level = QUALITY_LEVEL_DROP
    else:
        final_score = _combined_person_quality_score(
            pose_metrics.pose_score,
            mask_score,
            crop_score,
        )
        final_level = min(person_quality_level_from_score(final_score), cap_level)

    reason_set = set(reasons)
    is_truncated = bool(confirmed_truncated_edges)
    has_second_person = bool(
        reason_set
        & {
            QualityReason.BBOX_HEAVILY_OVERLAPS_OTHER_PERSON,
            QualityReason.BBOX_OVERLAPS_OTHER_PERSON,
            QualityReason.MASK_MULTIPLE_COMPONENTS,
        }
    )

    return PersonQualityDecision(
        person_quality_score=float(final_score),
        person_quality_level=int(final_level),
        person_quality_action=person_quality_action_from_level(final_level),
        mask_score=float(mask_score),
        crop_score=float(crop_score),
        quality_reasons=reasons,
        pose_score=pose_metrics.pose_score,
        visible_keypoint_count=pose_metrics.visible_keypoint_count,
        torso_keypoint_count=pose_metrics.torso_keypoint_count,
        shoulder_hip_chain_count=pose_metrics.shoulder_hip_chain_count,
        half_body_pose_eligible=half_upper_gallery_pose,
        body_completeness=pose_metrics.body_completeness,
        is_half_body=pose_metrics.is_half_body,
        is_truncated=is_truncated,
        has_second_person=has_second_person,
        bbox_overflow_ratio=float(bbox_overflow_ratio),
        near_image_edge_count=int(near_image_edge_count),
        truncated_edge_count=len(confirmed_truncated_edges),
    )
