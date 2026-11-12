"""Strict, immutable configuration contract for algorithm thresholds."""

import math
import os
import tomllib
from pathlib import Path
from typing import Annotated

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALGORITHM_CONFIG_PATH = PROJECT_ROOT / "config" / "algorithm.toml"


class StrictFrozenModel(BaseModel):
    """Reject unknown or non-strict values and prevent mutations."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class AlgorithmConfigError(RuntimeError):
    """Raised when the selected algorithm configuration cannot be loaded."""


Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
ByteValue = Annotated[int, Field(ge=0, le=255)]


def _require_sum(actual: float, expected: float, description: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{description} must sum to {expected}")


class FaceMatchingConfig(StrictFrozenModel):
    general_similarity_threshold: Ratio
    single_sample_similarity_threshold: Ratio


class FaceGalleryConfig(StrictFrozenModel):
    max_vectors_per_id: Annotated[int, Field(ge=1, le=1000)]


class FaceCropConfig(StrictFrozenModel):
    scale_factor: Annotated[float, Field(ge=1.0)]


class FaceQualityConfig(StrictFrozenModel):
    recognition_min_score: NonNegativeFloat
    enroll_min_score: NonNegativeFloat
    min_face_width_px: PositiveFloat
    max_pose_abs_degrees: Annotated[float, Field(gt=0.0, le=180.0)]
    geometry_std_ratio_threshold: Ratio
    laplacian_threshold: NonNegativeFloat
    feature_norm_threshold: PositiveFloat

    @model_validator(mode="after")
    def validate_score_order(self):
        if self.recognition_min_score > self.enroll_min_score:
            raise ValueError("recognition_min_score must be <= enroll_min_score")
        return self


class FaceOrientationConfig(StrictFrozenModel):
    side_yaw_degrees: Annotated[float, Field(ge=0.0, le=180.0)]


class PersonAssignmentConfig(StrictFrozenModel):
    pose_match_min_iou: Ratio
    pose_match_min_mask_agreement: Ratio
    pose_match_mask_weight: Ratio
    pose_match_iou_weight: Ratio
    pose_keypoint_visible_threshold: Ratio
    pose_mask_sample_radius_min_px: PositiveInt
    pose_mask_sample_radius_ratio: Ratio
    reid_duplicate_bbox_iou: Ratio
    reid_duplicate_mask_iou: Ratio

    @model_validator(mode="after")
    def validate_weights(self):
        _require_sum(
            self.pose_match_mask_weight + self.pose_match_iou_weight,
            1.0,
            "pose_match_mask_weight and pose_match_iou_weight",
        )
        return self


class PersonCropConfig(StrictFrozenModel):
    margin_ratio: Ratio
    support_margin_ratio: Ratio
    support_min_margin_px: PositiveInt
    border_bgr: tuple[ByteValue, ByteValue, ByteValue]
    support_visible_threshold: Ratio
    upright_dead_zone_degrees: Annotated[float, Field(ge=0.0, le=180.0)]

    @field_validator("border_bgr", mode="before")
    @classmethod
    def validate_border_bgr(cls, value):
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(channel) is not int for channel in value)
        ):
            raise ValueError("border_bgr must be a TOML list of exactly three strict integers")
        return tuple(value)


class PersonPoseGeometryConfig(StrictFrozenModel):
    torso_visible_threshold: Ratio
    torso_min_axis_diagonal_ratio: Ratio
    torso_max_chain_disagreement_degrees: Annotated[float, Field(ge=0.0, le=180.0)]


class PersonOrientationConfig(StrictFrozenModel):
    side_span_ratio: Ratio


class PersonQualityGeneralConfig(StrictFrozenModel):
    recognition_min_score: NonNegativeFloat
    enroll_min_score: NonNegativeFloat
    max_mask_pixels: PositiveInt
    score_max: PositiveFloat
    recognition_score_gap: PositiveFloat
    pose_weight: Ratio
    mask_weight: Ratio
    crop_weight: Ratio

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.recognition_min_score > self.enroll_min_score:
            raise ValueError("recognition_min_score must be <= enroll_min_score")
        if self.recognition_score_gap >= self.enroll_min_score:
            raise ValueError("recognition_score_gap must be < enroll_min_score")
        _require_sum(
            self.pose_weight + self.mask_weight + self.crop_weight,
            1.0,
            "pose_weight, mask_weight, and crop_weight",
        )
        return self


class PersonQualityCropConfig(StrictFrozenModel):
    foreign_diagnostic_ratio: Ratio
    foreign_diagnostic_min_pixels: PositiveInt
    foreign_cap_ratio: Ratio
    foreign_reject_ratio: Ratio
    foreign_core_cap_ratio: Ratio
    foreign_core_reject_ratio: Ratio
    foreign_core_width_ratio: Ratio
    foreign_edge_dominant_ratio: Ratio

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.foreign_cap_ratio > self.foreign_reject_ratio:
            raise ValueError("foreign_cap_ratio must be <= foreign_reject_ratio")
        if self.foreign_core_cap_ratio > self.foreign_core_reject_ratio:
            raise ValueError("foreign_core_cap_ratio must be <= foreign_core_reject_ratio")
        return self


class PersonQualityEdgeConfig(StrictFrozenModel):
    mask_edge_cap_count: PositiveInt
    mask_edge_cap_ratio: Ratio
    mask_edge_reject_count: PositiveInt
    mask_edge_reject_ratio: Ratio
    bbox_overflow_cap_ratio: Ratio
    bbox_overflow_reject_ratio: Ratio
    truncated_edge_reject_count: PositiveInt
    half_upper_gallery_min_visible_keypoints: PositiveInt
    min_bbox_width_px: PositiveInt
    min_bbox_height_px: PositiveInt
    min_bbox_image_ratio: Ratio
    max_bbox_image_ratio: Ratio
    min_bbox_aspect_ratio: PositiveFloat
    max_bbox_aspect_ratio: PositiveFloat
    other_person_overlap_cap_ratio: Ratio
    other_person_overlap_reject_ratio: Ratio
    mask_bbox_cap_ratio: Ratio
    mask_bbox_reject_ratio: Ratio
    largest_component_min_ratio: Ratio
    large_component_min_pixels: PositiveInt
    large_component_ratio: Ratio
    fragmented_component_count: PositiveInt
    image_edge_margin_min_px: PositiveInt
    image_edge_margin_ratio: Ratio
    mask_bbox_touch_min_pixels: PositiveInt
    mask_bbox_touch_ratio: Ratio
    mask_image_band_min_px: PositiveInt
    mask_image_band_ratio: Ratio
    mask_image_touch_min_pixels: PositiveInt
    mask_image_touch_ratio: Ratio

    @model_validator(mode="after")
    def validate_relationships(self):
        comparisons = (
            (self.mask_edge_cap_count, self.mask_edge_reject_count, "mask_edge_cap_count", "mask_edge_reject_count"),
            (self.mask_edge_cap_ratio, self.mask_edge_reject_ratio, "mask_edge_cap_ratio", "mask_edge_reject_ratio"),
            (self.bbox_overflow_cap_ratio, self.bbox_overflow_reject_ratio, "bbox_overflow_cap_ratio", "bbox_overflow_reject_ratio"),
            (self.other_person_overlap_cap_ratio, self.other_person_overlap_reject_ratio, "other_person_overlap_cap_ratio", "other_person_overlap_reject_ratio"),
            (self.mask_bbox_reject_ratio, self.mask_bbox_cap_ratio, "mask_bbox_reject_ratio", "mask_bbox_cap_ratio"),
        )
        for lower, upper, lower_name, upper_name in comparisons:
            if lower > upper:
                raise ValueError(f"{lower_name} must be <= {upper_name}")
        if self.min_bbox_image_ratio >= self.max_bbox_image_ratio:
            raise ValueError("min_bbox_image_ratio must be < max_bbox_image_ratio")
        if self.min_bbox_aspect_ratio >= self.max_bbox_aspect_ratio:
            raise ValueError("min_bbox_aspect_ratio must be < max_bbox_aspect_ratio")
        return self


class PersonQualityPoseConfig(StrictFrozenModel):
    visible_threshold: Ratio
    min_visible_keypoints: PositiveInt
    min_torso_keypoints: PositiveInt
    min_bbox_agreement: Ratio
    min_mask_agreement: Ratio
    min_vertical_span_ratio: Ratio
    full_vertical_span_ratio: Ratio
    visible_score_weight: NonNegativeFloat
    torso_score_weight: NonNegativeFloat
    lower_score_weight: NonNegativeFloat
    shoulder_hip_score_weight: NonNegativeFloat
    vertical_span_score_weight: NonNegativeFloat

    @model_validator(mode="after")
    def validate_span_order(self):
        if self.min_vertical_span_ratio > self.full_vertical_span_ratio:
            raise ValueError("min_vertical_span_ratio must be <= full_vertical_span_ratio")
        return self


class PersonQualityVisualConfig(StrictFrozenModel):
    min_foreground_pixels: PositiveInt
    mask_erosion_kernel: PositiveInt
    mask_erosion_iterations: PositiveInt
    laplacian_reject_threshold: NonNegativeFloat
    laplacian_cap_threshold: NonNegativeFloat
    black_clip_threshold: ByteValue
    dark_threshold: ByteValue
    white_clip_threshold: ByteValue
    bright_threshold: ByteValue
    exposure_reject_ratio: Ratio
    exposure_cap_ratio: Ratio
    underexposed_reject_p95: NonNegativeFloat
    underexposed_cap_p90: NonNegativeFloat
    overexposed_reject_p5: NonNegativeFloat
    overexposed_cap_p10: NonNegativeFloat
    contrast_reject_threshold: NonNegativeFloat
    contrast_cap_threshold: NonNegativeFloat

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.mask_erosion_kernel % 2 != 1:
            raise ValueError("mask_erosion_kernel must be odd")
        comparisons = (
            (self.laplacian_reject_threshold, self.laplacian_cap_threshold, "laplacian_reject_threshold", "laplacian_cap_threshold"),
            (self.black_clip_threshold, self.dark_threshold, "black_clip_threshold", "dark_threshold"),
            (self.bright_threshold, self.white_clip_threshold, "bright_threshold", "white_clip_threshold"),
            (self.exposure_cap_ratio, self.exposure_reject_ratio, "exposure_cap_ratio", "exposure_reject_ratio"),
            (self.underexposed_reject_p95, self.underexposed_cap_p90, "underexposed_reject_p95", "underexposed_cap_p90"),
            (self.overexposed_cap_p10, self.overexposed_reject_p5, "overexposed_cap_p10", "overexposed_reject_p5"),
            (self.contrast_reject_threshold, self.contrast_cap_threshold, "contrast_reject_threshold", "contrast_cap_threshold"),
        )
        for lower, upper, lower_name, upper_name in comparisons:
            if lower > upper:
                raise ValueError(f"{lower_name} must be <= {upper_name}")
        return self


class PersonQualityConfig(StrictFrozenModel):
    general: PersonQualityGeneralConfig
    crop: PersonQualityCropConfig
    edge: PersonQualityEdgeConfig
    pose: PersonQualityPoseConfig
    visual: PersonQualityVisualConfig

    @model_validator(mode="after")
    def validate_pose_score_total(self):
        _require_sum(
            self.pose.visible_score_weight
            + self.pose.torso_score_weight
            + self.pose.lower_score_weight
            + self.pose.shoulder_hip_score_weight
            + self.pose.vertical_span_score_weight,
            self.general.score_max,
            "person quality pose score weights",
        )
        return self


class PersonFaceOwnershipConfig(StrictFrozenModel):
    mask_min_coverage: Ratio
    pose_visible_threshold: Ratio
    head_bbox_margin_ratio: Ratio


class PersonFaceBindingConfig(StrictFrozenModel):
    crop_scale_factor: Annotated[float, Field(ge=1.0)]
    min_face_width_px: PositiveInt
    laplacian_threshold: NonNegativeFloat
    max_pose_abs_degrees: Annotated[float, Field(gt=0.0, le=180.0)]
    geometry_std_ratio_threshold: Ratio
    feature_norm_threshold: PositiveFloat
    quality_min_score: NonNegativeFloat


class ReidBodyConfig(StrictFrozenModel):
    same_body_threshold: Ratio
    enrollable_cross_body_threshold: Ratio
    other_cross_body_threshold: Ratio


class ReidClassificationConfig(StrictFrozenModel):
    class_top_k: PositiveInt
    face_class_margin: Ratio
    body_class_margin: Ratio
    body_local_top1: Ratio
    body_cross_device_top1: Ratio
    body_local_mean: Ratio
    body_cross_device_mean: Ratio
    unique_anchor_cross_device_top1: Ratio
    unique_anchor_raw_margin: Ratio
    query_window_days: PositiveInt
    best_score_weight: Ratio
    mean_score_weight: Ratio

    @model_validator(mode="after")
    def validate_weights(self):
        _require_sum(
            self.best_score_weight + self.mean_score_weight,
            1.0,
            "best_score_weight and mean_score_weight",
        )
        return self


class ReidPrototypeConfig(StrictFrozenModel):
    max_per_reid: PositiveInt
    max_per_domain: PositiveInt
    duplicate_threshold: Ratio
    class_floor: Ratio
    replace_redundancy_floor: Ratio

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.max_per_domain > self.max_per_reid:
            raise ValueError("max_per_domain must be <= max_per_reid")
        if not self.class_floor <= self.replace_redundancy_floor <= self.duplicate_threshold:
            raise ValueError(
                "class_floor must be <= replace_redundancy_floor <= duplicate_threshold"
            )
        return self


class FaceConfig(StrictFrozenModel):
    matching: FaceMatchingConfig
    gallery: FaceGalleryConfig
    crop: FaceCropConfig
    quality: FaceQualityConfig
    orientation: FaceOrientationConfig


class PersonConfig(StrictFrozenModel):
    assignment: PersonAssignmentConfig
    crop: PersonCropConfig
    pose_geometry: PersonPoseGeometryConfig
    orientation: PersonOrientationConfig
    quality: PersonQualityConfig


class PersonFaceConfig(StrictFrozenModel):
    ownership: PersonFaceOwnershipConfig
    binding: PersonFaceBindingConfig


class ReidConfig(StrictFrozenModel):
    body: ReidBodyConfig
    classification: ReidClassificationConfig
    prototype: ReidPrototypeConfig


class AlgorithmConfig(StrictFrozenModel):
    face: FaceConfig
    person: PersonConfig
    person_face: PersonFaceConfig
    reid: ReidConfig


def _resolve_algorithm_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        selected = path
    elif "ALGORITHM_CONFIG_PATH" in os.environ:
        selected = os.environ["ALGORITHM_CONFIG_PATH"]
    else:
        selected = dotenv_values(PROJECT_ROOT / ".env").get("ALGORITHM_CONFIG_PATH")
    candidate = Path(selected) if selected else DEFAULT_ALGORITHM_CONFIG_PATH
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def load_algorithm_config(path: str | Path | None = None) -> AlgorithmConfig:
    """Load and validate the selected immutable algorithm configuration."""
    legacy_key = "FACE_MAX_VECTORS_PER_ID"
    if legacy_key in os.environ or legacy_key in dotenv_values(PROJECT_ROOT / ".env"):
        raise AlgorithmConfigError(
            f"{legacy_key} is no longer supported; set "
            "face.gallery.max_vectors_per_id in the algorithm TOML profile"
        )
    selected = _resolve_algorithm_config_path(path)
    try:
        with selected.open("rb") as stream:
            raw = tomllib.load(stream)
        return AlgorithmConfig.model_validate(raw)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise AlgorithmConfigError(
            f"Invalid algorithm configuration {selected}: {error}"
        ) from error


algorithm_config = load_algorithm_config()
