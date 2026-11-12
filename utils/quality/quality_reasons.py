"""Stable quality reason codes shared by face and person flows."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class QualityReason(StrEnum):
    FACE_WIDTH_TOO_NARROW = "face_width_too_narrow"
    KEYPOINTS_OUTSIDE_BBOX = "keypoints_outside_bbox"
    NOSE_OUTSIDE_EYES_HORIZONTAL = "nose_outside_eyes_horizontal"
    NOSE_OUTSIDE_EYES_MOUTH_VERTICAL = "nose_outside_eyes_mouth_vertical"
    BAD_POSE = "bad_pose"
    GEOMETRIC_INSTABILITY = "geometric_instability"
    BLURRY_LAPLACIAN = "blurry_laplacian"
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    LOW_CONTRAST = "low_contrast"
    LOW_FEATURE_QUALITY_NORM = "low_feature_quality_norm"

    MASK_AREA_TOO_SMALL = "mask_area_too_small"
    MASK_AREA_BELOW_ENROLL = "mask_area_below_enroll"
    INVALID_CONFIDENCE = "invalid_confidence"
    INVALID_IMAGE_SHAPE = "invalid_image_shape"
    INVALID_BBOX = "invalid_bbox"
    EMPTY_CROP = "empty_crop"
    BBOX_TOO_SMALL = "bbox_too_small"
    BBOX_AREA_TOO_SMALL = "bbox_area_too_small"
    BBOX_AREA_TOO_LARGE = "bbox_area_too_large"
    BBOX_ASPECT_RATIO_ABNORMAL = "bbox_aspect_ratio_abnormal"
    BBOX_THREE_IMAGE_EDGES = "bbox_three_image_edges"
    BBOX_NEAR_IMAGE_EDGE = "bbox_near_image_edge"
    BBOX_CONTENT_OVERFLOW_HIGH = "bbox_content_overflow_high"
    BBOX_CONTENT_OVERFLOW_TOO_HIGH = "bbox_content_overflow_too_high"
    PERSON_TRUNCATED = "person_truncated"
    PERSON_TRUNCATED_SEVERE = "person_truncated_severe"
    BBOX_HEAVILY_OVERLAPS_OTHER_PERSON = "bbox_heavily_overlaps_other_person"
    BBOX_OVERLAPS_OTHER_PERSON = "bbox_overlaps_other_person"
    MASK_MISSING = "mask_missing"
    MASK_SHAPE_INVALID = "mask_shape_invalid"
    MASK_BBOX_RATIO_TOO_LOW = "mask_bbox_ratio_too_low"
    MASK_BBOX_RATIO_LOW = "mask_bbox_ratio_low"
    MASK_FRAGMENTED = "mask_fragmented"
    MASK_MULTIPLE_COMPONENTS = "mask_multiple_components"
    MASK_TOUCHING_BBOX_EDGE_SEVERE = "mask_touching_bbox_edge_severe"
    MASK_TOUCHING_BBOX_EDGE = "mask_touching_bbox_edge"
    CROP_PADDING_RATIO_HIGH = "crop_padding_ratio_high"
    CROP_PADDING_RATIO_TOO_HIGH = "crop_padding_ratio_too_high"
    CROP_CONTAINS_OTHER_PERSON_MILD = "crop_contains_other_person_mild"
    CROP_OTHER_PERSON_BBOX_FALLBACK = "crop_other_person_bbox_fallback"
    CROP_CONTAINS_OTHER_PERSON = "crop_contains_other_person"
    CROP_HEAVILY_CONTAINS_OTHER_PERSON = "crop_heavily_contains_other_person"


class PoseQualityReason(StrEnum):
    POSE_INFERENCE_FAILED = "pose_inference_failed"
    POSE_NOT_MATCHED = "pose_not_matched"
    POSE_KEYPOINTS_INVALID = "pose_keypoints_invalid"
    POSE_VISIBLE_KEYPOINTS_TOO_FEW = "pose_visible_keypoints_too_few"
    POSE_TORSO_INCOMPLETE = "pose_torso_incomplete"
    POSE_SHOULDER_HIP_CHAIN_INCOMPLETE = "pose_shoulder_hip_chain_incomplete"
    POSE_LOWER_BODY_INCOMPLETE = "pose_lower_body_incomplete"
    POSE_HALF_BODY = "pose_half_body"
    POSE_HEAD_INCOMPLETE = "pose_head_incomplete"
    POSE_BODY_TYPE_NOT_SEARCHABLE = "pose_body_type_not_searchable"
    POSE_KEYPOINTS_OUTSIDE_BBOX = "pose_keypoints_outside_bbox"
    POSE_MASK_MISMATCH = "pose_mask_mismatch"
    POSE_GEOMETRY_UNSTABLE = "pose_geometry_unstable"


def quality_reason_value(reason: Any) -> str:
    if isinstance(reason, (QualityReason, PoseQualityReason)):
        return reason.value
    return str(reason)
