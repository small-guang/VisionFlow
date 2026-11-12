"""Data models shared by person-quality evaluators."""

from __future__ import annotations

from dataclasses import dataclass

from ..quality_reasons import PoseQualityReason, QualityReason


@dataclass(frozen=True)
class PersonQualityDecision:
    person_quality_score: float
    person_quality_level: int
    person_quality_action: str
    mask_score: float
    crop_score: float
    quality_reasons: list[QualityReason | PoseQualityReason]
    pose_score: float = 0.0
    visible_keypoint_count: int = 0
    torso_keypoint_count: int = 0
    shoulder_hip_chain_count: int = 0
    half_body_pose_eligible: bool = False
    body_completeness: str = "unknown"
    is_half_body: bool = False
    is_truncated: bool = False
    has_second_person: bool = False
    foreign_person_ratio: float = 0.0
    foreign_person_core_ratio: float = 0.0
    foreign_person_pixel_count: int = 0
    foreign_person_core_pixel_count: int = 0
    foreign_person_bbox_fallback_ratio: float = 0.0
    foreign_person_bbox_fallback_pixel_count: int = 0
    foreign_person_edge_dominant: bool = False
    foreign_person_evidence_source: str = "none"
    padding_ratio: float = 0.0
    letterbox_padding_ratio: float = 0.0
    source_overflow_ratio: float = 0.0
    bbox_overflow_ratio: float = 0.0
    near_image_edge_count: int = 0
    truncated_edge_count: int = 0

@dataclass(frozen=True)
class PersonPoseMetrics:
    valid: bool
    pose_score: float
    visible_keypoint_count: int
    torso_keypoint_count: int
    shoulder_hip_chain_count: int
    hip_knee_chain_count: int
    full_lower_chain_count: int
    shoulder_hip_knee_chain_count: int
    full_body_chain_count: int
    vertical_span_ratio: float
    bbox_agreement: float
    mask_agreement: float | None
    body_completeness: str
    is_half_body: bool
    head_visible: bool = False

@dataclass(frozen=True)
class PersonVisualMetrics:
    laplacian_variance: float
    black_clip_ratio: float
    dark_ratio: float
    white_clip_ratio: float
    bright_ratio: float
    p5: float
    p10: float
    p90: float
    p95: float
    contrast_span: float
    used_foreground: bool
    sample_pixel_count: int

@dataclass(frozen=True)
class PersonVisualQualityDecision:
    reject: bool
    cap_level: int
    crop_score: float
    quality_reasons: list[QualityReason]
    metrics: PersonVisualMetrics

@dataclass(frozen=True)
class PersonCropMetrics:
    padding_ratio: float
    target_mask_pixel_count: int
    foreign_person_pixel_count: int
    foreign_person_ratio: float
    foreign_person_core_pixel_count: int = 0
    foreign_person_core_ratio: float = 0.0
    foreign_person_bbox_fallback_pixel_count: int = 0
    foreign_person_bbox_fallback_ratio: float = 0.0
    foreign_person_edge_dominant: bool = False
    foreign_person_evidence_source: str = "none"
    letterbox_padding_ratio: float = 0.0
    source_overflow_ratio: float = 0.0

@dataclass(frozen=True)
class PersonCropQualityDecision:
    reject: bool
    cap_level: int
    crop_score: float
    quality_reasons: list[QualityReason]
    metrics: PersonCropMetrics
    has_second_person: bool = False
