"""Pose metrics and body-completeness evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.algorithm_config import algorithm_config
from utils.keyframe.detection_helpers import pose_mask_agreement
from utils.reid.person_body import (
    BODY_FULL,
    BODY_HALF_LOWER,
    BODY_HALF_UPPER,
    BODY_PARTIAL,
    BODY_THREE_QUARTER_LOWER,
    BODY_THREE_QUARTER_UPPER,
    BODY_UNKNOWN,
)

from .geometry import _image_hw, _rect_xyxy, _safe_int
from .mask import _valid_mask
from .models import PersonPoseMetrics


PERSON_POSE_KEYPOINT_COUNT = 17  # COCO 人体姿态模型应输出的关键点总数

_POSE_HEAD = (0, 1, 2, 3, 4)
_POSE_SHOULDERS = (5, 6)
_POSE_HIPS = (11, 12)
_POSE_KNEES = (13, 14)
_POSE_ANKLES = (15, 16)
_POSE_TORSO = (*_POSE_SHOULDERS, *_POSE_HIPS)
_POSE_LOWER = (*_POSE_KNEES, *_POSE_ANKLES)

def calculate_person_pose_metrics(
    keypoints: Any,
    rect: dict[str, Any],
    image_shape: tuple[int, ...],
    mask: Any = None,
) -> PersonPoseMetrics:
    """Calculate structural metrics from whole-image COCO-17 keypoints."""
    pose_config = algorithm_config.person.quality.pose
    try:
        points = np.asarray(keypoints, dtype=np.float32)
    except (TypeError, ValueError):
        points = np.empty((0, 0), dtype=np.float32)

    if points.ndim != 2 or points.shape != (PERSON_POSE_KEYPOINT_COUNT, 3):
        return _invalid_pose_metrics()

    img_h, img_w = _image_hw(image_shape)
    if img_h <= 0 or img_w <= 0:
        return _invalid_pose_metrics()

    finite = np.all(np.isfinite(points), axis=1)
    inside_image = (
        (points[:, 0] >= 0)
        & (points[:, 0] < img_w)
        & (points[:, 1] >= 0)
        & (points[:, 1] < img_h)
    )
    visible = finite & inside_image & (points[:, 2] >= pose_config.visible_threshold)
    visible_count = int(np.count_nonzero(visible))
    torso_count = int(np.count_nonzero(visible[list(_POSE_TORSO)]))
    lower_count = int(np.count_nonzero(visible[list(_POSE_LOWER)]))

    shoulder_hip_chains = [
        bool(visible[_POSE_SHOULDERS[side]] and visible[_POSE_HIPS[side]])
        for side in range(2)
    ]
    hip_knee_chains = [
        bool(visible[_POSE_HIPS[side]] and visible[_POSE_KNEES[side]])
        for side in range(2)
    ]
    full_lower_chains = [
        bool(hip_knee_chains[side] and visible[_POSE_ANKLES[side]])
        for side in range(2)
    ]
    shoulder_hip_knee_chains = [
        bool(shoulder_hip_chains[side] and visible[_POSE_KNEES[side]])
        for side in range(2)
    ]
    full_body_chains = [
        bool(shoulder_hip_knee_chains[side] and visible[_POSE_ANKLES[side]])
        for side in range(2)
    ]
    shoulder_hip_count = sum(shoulder_hip_chains)
    hip_knee_count = sum(hip_knee_chains)
    full_lower_count = sum(full_lower_chains)
    shoulder_hip_knee_count = sum(shoulder_hip_knee_chains)
    full_body_chain_count = sum(full_body_chains)
    head_visible = bool(np.any(visible[list(_POSE_HEAD)]))

    rect_height = max(1, _safe_int(rect.get("h")))
    if visible_count >= 2:
        visible_y = points[visible, 1]
        vertical_span_ratio = float(
            max(0.0, float(np.max(visible_y) - np.min(visible_y))) / rect_height
        )
    else:
        vertical_span_ratio = 0.0

    x1, y1, x2, y2 = _rect_xyxy(rect)
    if visible_count:
        visible_points = points[visible]
        inside_bbox = (
            (visible_points[:, 0] >= x1)
            & (visible_points[:, 0] <= x2)
            & (visible_points[:, 1] >= y1)
            & (visible_points[:, 1] <= y2)
        )
        bbox_agreement = float(np.mean(inside_bbox))
    else:
        bbox_agreement = 0.0

    valid_mask = _valid_mask(mask, img_h, img_w)
    mask_agreement = (
        pose_mask_agreement(points[visible], valid_mask, rect)
        if valid_mask is not None and visible_count
        else None
    )

    if head_visible and full_body_chain_count > 0:
        body_completeness = BODY_FULL
    elif head_visible and shoulder_hip_knee_count > 0:
        body_completeness = BODY_THREE_QUARTER_UPPER
    elif not head_visible and full_body_chain_count > 0:
        body_completeness = BODY_THREE_QUARTER_LOWER
    elif head_visible and shoulder_hip_count > 0:
        body_completeness = BODY_HALF_UPPER
    elif not head_visible and shoulder_hip_count == 0 and full_lower_count > 0:
        body_completeness = BODY_HALF_LOWER
    else:
        body_completeness = BODY_PARTIAL

    pose_score = (
        pose_config.visible_score_weight * visible_count / PERSON_POSE_KEYPOINT_COUNT
        + pose_config.torso_score_weight * torso_count / len(_POSE_TORSO)
        + pose_config.lower_score_weight * lower_count / len(_POSE_LOWER)
        + pose_config.shoulder_hip_score_weight * shoulder_hip_count / 2.0
        + pose_config.vertical_span_score_weight
        * min(vertical_span_ratio / pose_config.full_vertical_span_ratio, 1.0)
    )
    return PersonPoseMetrics(
        valid=True,
        pose_score=float(
            min(max(pose_score, 0.0), algorithm_config.person.quality.general.score_max)
        ),
        visible_keypoint_count=visible_count,
        torso_keypoint_count=torso_count,
        shoulder_hip_chain_count=shoulder_hip_count,
        hip_knee_chain_count=hip_knee_count,
        full_lower_chain_count=full_lower_count,
        shoulder_hip_knee_chain_count=shoulder_hip_knee_count,
        full_body_chain_count=full_body_chain_count,
        vertical_span_ratio=vertical_span_ratio,
        bbox_agreement=bbox_agreement,
        mask_agreement=mask_agreement,
        body_completeness=body_completeness,
        is_half_body=body_completeness in {BODY_HALF_UPPER, BODY_HALF_LOWER},
        head_visible=head_visible,
    )

def _invalid_pose_metrics() -> PersonPoseMetrics:
    return PersonPoseMetrics(
        valid=False,
        pose_score=0.0,
        visible_keypoint_count=0,
        torso_keypoint_count=0,
        shoulder_hip_chain_count=0,
        hip_knee_chain_count=0,
        full_lower_chain_count=0,
        shoulder_hip_knee_chain_count=0,
        full_body_chain_count=0,
        vertical_span_ratio=0.0,
        bbox_agreement=0.0,
        mask_agreement=None,
        body_completeness=BODY_UNKNOWN,
        is_half_body=False,
        head_visible=False,
    )
