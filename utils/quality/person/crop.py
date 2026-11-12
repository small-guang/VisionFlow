"""Crop padding and foreign-person quality checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.algorithm_config import algorithm_config
from utils.keyframe.person_crop import PersonCropResult, project_mask_to_person_crop
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    QUALITY_LEVEL_ENROLL,
    QUALITY_LEVEL_RECOGNITION,
)

from ..quality_reasons import QualityReason
from .geometry import _image_hw, _project_rect_to_person_crop
from .mask import _valid_mask
from .models import PersonCropMetrics, PersonCropQualityDecision


def _target_core_region(target_mask: np.ndarray) -> np.ndarray:
    """Return the full-height central portion of the target-mask bounding box."""
    core = np.zeros(target_mask.shape, dtype=bool)
    ys, xs = np.nonzero(target_mask)
    if xs.size == 0 or ys.size == 0:
        return core

    x1 = int(np.min(xs))
    x2 = int(np.max(xs)) + 1
    y1 = int(np.min(ys))
    y2 = int(np.max(ys)) + 1
    width = x2 - x1
    core_width = max(
        1, int(round(width * algorithm_config.person.quality.crop.foreign_core_width_ratio))
    )
    core_x1 = x1 + max(0, (width - core_width) // 2)
    core_x2 = min(x2, core_x1 + core_width)
    core[y1:y2, core_x1:core_x2] = True
    return core


def _foreign_evidence_source(mask_pixels: int, fallback_pixels: int) -> str:
    if mask_pixels > 0 and fallback_pixels > 0:
        return "mixed"
    if mask_pixels > 0:
        return "mask"
    if fallback_pixels > 0:
        return "bbox_fallback"
    return "none"


def evaluate_person_crop_quality(
    crop_result: PersonCropResult,
    segment: dict[str, Any],
    all_segments: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    *,
    segment_index: int | None = None,
) -> PersonCropQualityDecision:
    """Record crop geometry and gate foreign-person contamination."""
    crop_config = algorithm_config.person.quality.crop
    general_config = algorithm_config.person.quality.general
    reasons: list[QualityReason] = []
    reject = False
    cap_level = QUALITY_LEVEL_ENROLL
    crop_score = general_config.score_max
    img_h, img_w = _image_hw(image_shape)

    padding_ratio = max(0.0, min(float(crop_result.padding_ratio), 1.0))
    letterbox_padding_ratio = max(
        0.0, min(float(crop_result.letterbox_padding_ratio), 1.0)
    )
    source_overflow_ratio = max(
        0.0, min(float(crop_result.source_overflow_ratio), 1.0)
    )

    target_mask = crop_result.mask
    if (
        isinstance(target_mask, np.ndarray)
        and target_mask.ndim == 2
        and tuple(target_mask.shape) == tuple(crop_result.image.shape[:2])
    ):
        target_mask_pixel_count = int(np.count_nonzero(target_mask))
    else:
        target_mask_pixel_count = 0

    foreign_mask = np.zeros(crop_result.image.shape[:2], dtype=bool)
    foreign_bbox_fallback_mask = np.zeros(
        crop_result.image.shape[:2], dtype=bool
    )
    current_rect = segment.get("rect")
    for index, other in enumerate(all_segments):
        if segment_index is not None and index == segment_index:
            continue
        if segment_index is None and other is segment:
            continue
        if segment_index is None and other.get("rect") is current_rect:
            continue

        other_mask = _valid_mask(other.get("mask"), img_h, img_w)
        if other_mask is not None and other_mask.count() > 0:
            projected = project_mask_to_person_crop(
                other_mask,
                crop_result,
                image_shape,
            )
            if projected is not None:
                foreign_mask |= projected.astype(bool)
            continue
        _project_rect_to_person_crop(
            foreign_bbox_fallback_mask,
            other.get("rect") or {},
            crop_result,
        )

    foreign_person_pixel_count = int(np.count_nonzero(foreign_mask))
    target_core_region = (
        _target_core_region(target_mask.astype(bool))
        if target_mask_pixel_count > 0
        else np.zeros(crop_result.image.shape[:2], dtype=bool)
    )
    foreign_person_core_pixel_count = int(
        np.count_nonzero(foreign_mask & target_core_region)
    )
    foreign_person_bbox_fallback_pixel_count = int(
        np.count_nonzero(foreign_bbox_fallback_mask)
    )
    foreign_person_ratio = (
        float(foreign_person_pixel_count / target_mask_pixel_count)
        if target_mask_pixel_count > 0
        else 0.0
    )
    foreign_person_core_ratio = (
        float(foreign_person_core_pixel_count / target_mask_pixel_count)
        if target_mask_pixel_count > 0
        else 0.0
    )
    foreign_person_bbox_fallback_ratio = (
        float(
            foreign_person_bbox_fallback_pixel_count
            / target_mask_pixel_count
        )
        if target_mask_pixel_count > 0
        else 0.0
    )
    diagnostic_min_pixels = max(
        crop_config.foreign_diagnostic_min_pixels,
        int(
            round(
                target_mask_pixel_count
                * crop_config.foreign_diagnostic_ratio
            )
        ),
    )
    mask_diagnostic = foreign_person_pixel_count >= diagnostic_min_pixels
    fallback_diagnostic = (
        foreign_person_bbox_fallback_pixel_count >= diagnostic_min_pixels
    )
    reject_for_mask = bool(
        foreign_person_ratio >= crop_config.foreign_reject_ratio
        or foreign_person_core_ratio >= crop_config.foreign_core_reject_ratio
    )
    cap_for_mask = bool(
        foreign_person_ratio >= crop_config.foreign_cap_ratio
        or foreign_person_core_ratio >= crop_config.foreign_core_cap_ratio
    )
    has_second_person = bool(
        mask_diagnostic or fallback_diagnostic or reject_for_mask or cap_for_mask
    )
    outside_core_pixel_count = max(
        0,
        foreign_person_pixel_count - foreign_person_core_pixel_count,
    )
    foreign_person_edge_dominant = bool(
        mask_diagnostic
        and foreign_person_pixel_count > 0
        and outside_core_pixel_count / foreign_person_pixel_count
        >= crop_config.foreign_edge_dominant_ratio
    )
    evidence_source = _foreign_evidence_source(
        foreign_person_pixel_count,
        foreign_person_bbox_fallback_pixel_count,
    )

    if reject_for_mask:
        reasons.append(QualityReason.CROP_HEAVILY_CONTAINS_OTHER_PERSON)
        reject = True
        crop_score = 0.0
    elif cap_for_mask:
        reasons.append(QualityReason.CROP_CONTAINS_OTHER_PERSON)
        cap_level = min(cap_level, QUALITY_LEVEL_RECOGNITION)
        crop_score = min(
            crop_score,
            general_config.enroll_min_score - general_config.recognition_score_gap,
        )
    elif mask_diagnostic:
        reasons.append(QualityReason.CROP_CONTAINS_OTHER_PERSON_MILD)

    if fallback_diagnostic:
        reasons.append(QualityReason.CROP_OTHER_PERSON_BBOX_FALLBACK)

    if reject:
        cap_level = QUALITY_LEVEL_DROP

    return PersonCropQualityDecision(
        reject=reject,
        cap_level=cap_level,
        crop_score=float(crop_score),
        quality_reasons=reasons,
        metrics=PersonCropMetrics(
            padding_ratio=float(padding_ratio),
            target_mask_pixel_count=target_mask_pixel_count,
            foreign_person_pixel_count=foreign_person_pixel_count,
            foreign_person_ratio=float(foreign_person_ratio),
            foreign_person_core_pixel_count=foreign_person_core_pixel_count,
            foreign_person_core_ratio=float(foreign_person_core_ratio),
            foreign_person_bbox_fallback_pixel_count=(
                foreign_person_bbox_fallback_pixel_count
            ),
            foreign_person_bbox_fallback_ratio=float(
                foreign_person_bbox_fallback_ratio
            ),
            foreign_person_edge_dominant=foreign_person_edge_dominant,
            foreign_person_evidence_source=evidence_source,
            letterbox_padding_ratio=float(letterbox_padding_ratio),
            source_overflow_ratio=float(source_overflow_ratio),
        ),
        has_second_person=has_second_person,
    )
