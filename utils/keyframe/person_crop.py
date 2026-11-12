"""Person crop geometry, upright rotation, and coordinate projection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.algorithm_config import algorithm_config
from utils.person_mask import as_person_mask

from .person_pose_geometry import estimate_person_torso_axis


# Supports a full 4K frame plus the default margin while bounding a uint8
# three-channel crop canvas to 48 MiB before optional masks and downstream work.
_PERSON_CROP_MAX_OUTPUT_DIMENSION = 8192
_PERSON_CROP_MAX_OUTPUT_PIXELS = 16 * 1024 * 1024


@dataclass(frozen=True)
class PersonRotationDecision:
    angle: float
    should_rotate: bool
    reason: str


@dataclass(frozen=True)
class PersonCropResult:
    image: np.ndarray
    mask: Optional[np.ndarray]
    requested_xyxy: Tuple[int, int, int, int]
    source_xyxy: Tuple[int, int, int, int]
    content_xyxy: Tuple[int, int, int, int]
    padding: Tuple[int, int, int, int]
    padding_ratio: float
    letterbox_padding_ratio: float = 0.0
    source_overflow_ratio: float = 0.0
    source_to_crop: Optional[np.ndarray] = None
    rotation_angle: float = 0.0
    rotation_applied: bool = False
    rotation_reason: str = "disabled"


def estimate_person_upright_rotation(
    keypoints: Any,
    rect: dict[str, Any],
    *,
    enabled: bool,
) -> PersonRotationDecision:
    """Estimate the rotation that makes the COCO-17 shoulder-to-hip axis upright."""
    if not enabled:
        return PersonRotationDecision(0.0, False, "disabled")

    axis_result = estimate_person_torso_axis(keypoints, rect)
    if not axis_result.valid:
        return PersonRotationDecision(0.0, False, axis_result.reason)
    down_axis = axis_result.down_axis
    if down_axis is None:
        return PersonRotationDecision(0.0, False, "pose_axis_missing")

    angle = -math.degrees(
        math.atan2(float(down_axis[0]), float(down_axis[1]))
    )
    angle = (angle + 180.0) % 360.0 - 180.0
    if abs(angle) < algorithm_config.person.crop.upright_dead_zone_degrees:
        return PersonRotationDecision(float(angle), False, "upright_dead_zone")
    return PersonRotationDecision(float(angle), True, "applied")


def build_person_crop(
    image: np.ndarray,
    rect: Dict[str, int],
    mask: Any = None,
    *,
    margin_ratio: float = algorithm_config.person.crop.margin_ratio,
    keypoints: Any = None,
    rotation_enabled: bool = False,
) -> Optional[PersonCropResult]:
    """Build a person crop, optionally rotating a reliable Pose axis upright."""
    decision = estimate_person_upright_rotation(
        keypoints,
        rect,
        enabled=rotation_enabled,
    )
    if not decision.should_rotate:
        return _build_unrotated_person_crop(
            image,
            rect,
            mask,
            margin_ratio=margin_ratio,
            rotation_angle=decision.angle,
            rotation_reason=decision.reason,
        )
    try:
        return _build_rotated_person_crop(
            image,
            rect,
            mask,
            margin_ratio=margin_ratio,
            decision=decision,
            keypoints=keypoints,
        )
    except Exception:
        return _build_unrotated_person_crop(
            image,
            rect,
            mask,
            margin_ratio=margin_ratio,
            rotation_angle=0.0,
            rotation_reason="rotation_failed",
        )


def _affine_to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    return np.vstack(
        [np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]]
    )


def _warp_mask_nearest(
    mask: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Warp a mask while preserving dtypes unsupported by OpenCV remap."""
    original_dtype = mask.dtype
    supported_dtypes = (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.int16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    )
    warp_input = mask
    if original_dtype == np.dtype(bool):
        warp_input = mask.astype(np.uint8)
    elif original_dtype not in supported_dtypes:
        warp_input = mask.astype(np.float64)
    warped = cv2.warpAffine(
        warp_input,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if warped.dtype != original_dtype:
        warped = warped.astype(original_dtype)
    return warped


def _warp_source_view_nearest(
    mask: Any,
    matrix: np.ndarray,
    source_xyxy: Tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> Optional[np.ndarray]:
    """Warp only a validated source window with a global affine matrix."""
    region = as_person_mask(mask)
    try:
        affine = np.asarray(matrix, dtype=np.float64)
        source_values = tuple(source_xyxy)
        output_values = tuple(output_size)
    except (TypeError, ValueError):
        return None
    if (
        region is None
        or affine.shape != (2, 3)
        or not bool(np.all(np.isfinite(affine)))
        or len(source_values) != 4
        or len(output_values) != 2
    ):
        return None

    try:
        source_bounds = tuple(int(value) for value in source_values)
        output_width, output_height = (
            int(value) for value in output_values
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        any(
            isinstance(value, (bool, np.bool_))
            or value != converted
            for value, converted in zip(source_values, source_bounds)
        )
        or any(
            isinstance(value, (bool, np.bool_))
            or value != converted
            for value, converted in zip(
                output_values,
                (output_width, output_height),
            )
        )
    ):
        return None

    source_x1, source_y1, source_x2, source_y2 = source_bounds
    if (
        source_x1 < 0
        or source_y1 < 0
        or source_x2 <= source_x1
        or source_y2 <= source_y1
        or source_x2 > region.image_shape[1]
        or source_y2 > region.image_shape[0]
        or output_width <= 0
        or output_height <= 0
    ):
        return None

    # Keep the original source window and matrix, including its zero border,
    # so nearest-neighbor rounding matches a dense image mask exactly.
    source_view = region.window(source_bounds)
    source_view_to_global = np.asarray(
        [
            [1.0, 0.0, float(source_x1)],
            [0.0, 1.0, float(source_y1)],
        ],
        dtype=np.float64,
    )
    local_matrix = (
        _affine_to_homogeneous(affine)
        @ _affine_to_homogeneous(source_view_to_global)
    )[:2]
    warped = _warp_mask_nearest(
        source_view,
        local_matrix,
        (output_width, output_height),
    )
    if (
        not isinstance(warped, np.ndarray)
        or warped.shape != (output_height, output_width)
    ):
        raise ValueError("source-view mask warp shape mismatch")
    return np.ascontiguousarray(warped)


def _person_crop_output_shape_is_safe(height: int, width: int) -> bool:
    """Return whether a requested crop is inside the allocation budget."""
    return (
        0 < height <= _PERSON_CROP_MAX_OUTPUT_DIMENSION
        and 0 < width <= _PERSON_CROP_MAX_OUTPUT_DIMENSION
        and height <= _PERSON_CROP_MAX_OUTPUT_PIXELS // width
    )


def _nonzero_xyxy(array: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return the half-open tight bounds of nonzero pixels in a 2D array."""
    values = np.asarray(array)
    if values.ndim != 2:
        return None
    row_indices = np.flatnonzero(np.any(values, axis=1))
    if row_indices.size == 0:
        return None
    column_indices = np.flatnonzero(np.any(values, axis=0))
    return (
        int(column_indices[0]),
        int(row_indices[0]),
        int(column_indices[-1]) + 1,
        int(row_indices[-1]) + 1,
    )


def _points_xyxy(points: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return half-open bounds for finite two-dimensional point coordinates."""
    try:
        coordinates = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        return None
    finite = coordinates[np.all(np.isfinite(coordinates), axis=1)]
    if finite.shape[0] == 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        bounds = (
            float(np.min(finite[:, 0])),
            float(np.min(finite[:, 1])),
            float(np.max(finite[:, 0]) + 1.0),
            float(np.max(finite[:, 1]) + 1.0),
        )
    if not all(math.isfinite(value) for value in bounds):
        return None
    return bounds


def _transformed_visible_points(
    keypoints: Any,
    matrix: np.ndarray,
) -> np.ndarray:
    """Transform reliable finite COCO-17 points with an affine matrix."""
    try:
        points = np.asarray(keypoints, dtype=np.float64)
        affine = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)
    if (
        points.shape != (17, 3)
        or affine.shape != (2, 3)
        or not bool(np.all(np.isfinite(affine)))
    ):
        return np.empty((0, 2), dtype=np.float64)
    visible = np.all(np.isfinite(points), axis=1) & (
        points[:, 2] >= algorithm_config.person.crop.support_visible_threshold
    )
    if not bool(np.any(visible)):
        return np.empty((0, 2), dtype=np.float64)
    homogeneous = np.column_stack(
        [points[visible, :2], np.ones(int(np.count_nonzero(visible)))]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = homogeneous @ affine.T
    return transformed[np.all(np.isfinite(transformed), axis=1)]


def _union_xyxy(
    first: Optional[Tuple[float, float, float, float]],
    second: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """Return the smallest box containing both optional boxes."""
    if first is None:
        return second
    if second is None:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _intersect_xyxy(
    first: Optional[Tuple[float, float, float, float]],
    second: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """Return the nonempty intersection of two optional boxes."""
    if first is None or second is None:
        return None
    intersection = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
        return None
    return intersection


def _expand_clip_xyxy(
    bounds: Optional[Tuple[float, float, float, float]],
    canvas_width: int,
    canvas_height: int,
) -> Tuple[int, int, int, int]:
    """Expand support by a safe margin and clip it to the rotated canvas."""
    full_canvas = (0, 0, int(canvas_width), int(canvas_height))
    if canvas_width <= 0 or canvas_height <= 0 or bounds is None:
        return full_canvas
    try:
        x1, y1, x2, y2 = (float(value) for value in bounds)
    except (TypeError, ValueError, OverflowError):
        return full_canvas
    if (
        not all(math.isfinite(value) for value in (x1, y1, x2, y2))
        or x2 <= x1
        or y2 <= y1
    ):
        return full_canvas
    crop = algorithm_config.person.crop
    margin_x = max(
        float(crop.support_min_margin_px),
        (x2 - x1) * crop.support_margin_ratio,
    )
    margin_y = max(
        float(crop.support_min_margin_px),
        (y2 - y1) * crop.support_margin_ratio,
    )
    expanded = (
        max(0, int(math.floor(x1 - margin_x))),
        max(0, int(math.floor(y1 - margin_y))),
        min(canvas_width, int(math.ceil(x2 + margin_x))),
        min(canvas_height, int(math.ceil(y2 + margin_y))),
    )
    if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
        return full_canvas
    return expanded


def _build_rotated_person_crop(
    image: np.ndarray,
    rect: Dict[str, int],
    mask: Any,
    *,
    margin_ratio: float,
    decision: PersonRotationDecision,
    keypoints: Any,
) -> Optional[PersonCropResult]:
    x, y = int(rect["x"]), int(rect["y"])
    width, height = int(rect["w"]), int(rect["h"])
    margin_ratio = float(margin_ratio)
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
        or width <= 0
        or height <= 0
        or not math.isfinite(margin_ratio)
        or margin_ratio < 0.0
    ):
        raise ValueError("invalid rotated person crop input")

    image_height, image_width = image.shape[:2]
    margin_x = int(math.ceil(width * margin_ratio))
    margin_y = int(math.ceil(height * margin_ratio))
    natural_x1, natural_y1 = x - margin_x, y - margin_y
    natural_x2 = x + width + margin_x
    natural_y2 = y + height + margin_y
    natural_width = natural_x2 - natural_x1
    natural_height = natural_y2 - natural_y1
    if not _person_crop_output_shape_is_safe(natural_height, natural_width):
        raise ValueError("unsafe natural person crop shape")
    source_x1, source_y1 = max(0, natural_x1), max(0, natural_y1)
    source_x2, source_y2 = min(image_width, natural_x2), min(
        image_height, natural_y2
    )
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return None

    dx, dy = source_x1 - natural_x1, source_y1 - natural_y1
    source_width = source_x2 - source_x1
    source_height = source_y2 - source_y1
    natural = np.empty((natural_height, natural_width, 3), dtype=image.dtype)
    border_bgr = tuple(algorithm_config.person.crop.border_bgr)
    natural[...] = np.asarray(border_bgr, dtype=image.dtype)
    natural[dy : dy + source_height, dx : dx + source_width] = image[
        source_y1:source_y2,
        source_x1:source_x2,
    ]

    region = as_person_mask(mask, image.shape)

    center = ((natural_width - 1) / 2.0, (natural_height - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, decision.angle, 1.0)
    cosine = abs(float(rotation[0, 0]))
    sine = abs(float(rotation[0, 1]))
    rotated_width = max(
        1,
        int(math.ceil(natural_height * sine + natural_width * cosine)),
    )
    rotated_height = max(
        1,
        int(math.ceil(natural_height * cosine + natural_width * sine)),
    )
    if not _person_crop_output_shape_is_safe(rotated_height, rotated_width):
        raise ValueError("unsafe rotated person crop shape")
    rotation[0, 2] += (rotated_width - 1) / 2.0 - center[0]
    rotation[1, 2] += (rotated_height - 1) / 2.0 - center[1]
    rotated = cv2.warpAffine(
        natural,
        rotation,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_bgr,
    )
    if rotated.ndim != 3 or rotated.shape[2] != 3 or rotated.size == 0:
        raise ValueError("invalid rotated person crop")

    original_to_natural = np.asarray(
        [
            [1.0, 0.0, -float(natural_x1)],
            [0.0, 1.0, -float(natural_y1)],
        ],
        dtype=np.float64,
    )
    source_to_rotated = (
        _affine_to_homogeneous(rotation)
        @ _affine_to_homogeneous(original_to_natural)
    )[:2]

    source_validity = np.ones(
        (source_height, source_width),
        dtype=np.uint8,
    )
    source_view_to_original = np.asarray(
        [
            [1.0, 0.0, float(source_x1)],
            [0.0, 1.0, float(source_y1)],
        ],
        dtype=np.float64,
    )
    source_view_to_rotated = (
        _affine_to_homogeneous(source_to_rotated)
        @ _affine_to_homogeneous(source_view_to_original)
    )[:2]
    rotated_validity = _warp_mask_nearest(
        source_validity,
        source_view_to_rotated,
        (rotated_width, rotated_height),
    )
    if rotated_validity.shape != rotated.shape[:2]:
        raise ValueError("rotated validity shape mismatch")

    rotated_mask = None
    if region is not None:
        rotated_mask = _warp_source_view_nearest(
            region,
            source_to_rotated,
            (source_x1, source_y1, source_x2, source_y2),
            (rotated_width, rotated_height),
        )
        if rotated_mask is None or rotated_mask.shape != rotated.shape[:2]:
            raise ValueError("rotated person mask shape mismatch")

    mask_support = (
        None if rotated_mask is None else _nonzero_xyxy(rotated_mask)
    )
    pose_support = _points_xyxy(
        _transformed_visible_points(keypoints, source_to_rotated)
    )
    bbox_corners = np.asarray(
        [
            [x, y, 1.0],
            [x + width - 1, y, 1.0],
            [x + width - 1, y + height - 1, 1.0],
            [x, y + height - 1, 1.0],
        ],
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        bbox_support = _points_xyxy(bbox_corners @ source_to_rotated.T)

    if mask_support is not None and pose_support is not None:
        support = _union_xyxy(mask_support, pose_support)
    elif mask_support is not None:
        support = mask_support
    elif pose_support is not None:
        support = _intersect_xyxy(pose_support, bbox_support)
        if support is None:
            support = bbox_support
    elif bbox_support is not None:
        support = bbox_support
    else:
        support = (0.0, 0.0, float(rotated_width), float(rotated_height))

    try:
        tight_x1, tight_y1, tight_x2, tight_y2 = _expand_clip_xyxy(
            support,
            rotated_width,
            rotated_height,
        )
    except (TypeError, ValueError, ArithmeticError):
        tight_x1, tight_y1 = 0, 0
        tight_x2, tight_y2 = rotated_width, rotated_height
    rotated = np.ascontiguousarray(
        rotated[tight_y1:tight_y2, tight_x1:tight_x2]
    )
    rotated_validity = np.ascontiguousarray(
        rotated_validity[tight_y1:tight_y2, tight_x1:tight_x2]
    )
    if rotated_mask is not None:
        rotated_mask = np.ascontiguousarray(
            rotated_mask[tight_y1:tight_y2, tight_x1:tight_x2]
        )
    if (
        rotated.ndim != 3
        or rotated.shape[2] != 3
        or rotated.size == 0
        or rotated_validity.shape != rotated.shape[:2]
        or (
            rotated_mask is not None
            and rotated_mask.shape != rotated.shape[:2]
        )
    ):
        raise ValueError("invalid tight rotated person crop")

    tight_translation = np.asarray(
        [
            [1.0, 0.0, -float(tight_x1)],
            [0.0, 1.0, -float(tight_y1)],
        ],
        dtype=np.float64,
    )
    source_to_crop = (
        _affine_to_homogeneous(tight_translation)
        @ _affine_to_homogeneous(source_to_rotated)
    )[:2]

    output_height, output_width = rotated.shape[:2]
    content = _nonzero_xyxy(rotated_validity)
    if content is None:
        content_xyxy = (0, 0, 0, 0)
        padding = (0, 0, output_width, output_height)
    else:
        content_xyxy = content
        padding = (
            content[0],
            content[1],
            output_width - content[2],
            output_height - content[3],
        )

    output_area = max(1, output_width * output_height)
    natural_area = max(1, natural_width * natural_height)
    source_area = source_width * source_height
    return PersonCropResult(
        image=rotated,
        mask=rotated_mask,
        requested_xyxy=(natural_x1, natural_y1, natural_x2, natural_y2),
        source_xyxy=(source_x1, source_y1, source_x2, source_y2),
        content_xyxy=content_xyxy,
        padding=padding,
        padding_ratio=float(
            np.clip(
                1.0 - np.count_nonzero(rotated_validity) / output_area,
                0.0,
                1.0,
            )
        ),
        letterbox_padding_ratio=0.0,
        source_overflow_ratio=float(
            np.clip((natural_area - source_area) / natural_area, 0.0, 1.0)
        ),
        source_to_crop=np.ascontiguousarray(source_to_crop),
        rotation_angle=decision.angle,
        rotation_applied=True,
        rotation_reason="applied",
    )


def _build_unrotated_person_crop(
    image: np.ndarray,
    rect: Dict[str, int],
    mask: Any = None,
    *,
    margin_ratio: float = algorithm_config.person.crop.margin_ratio,
    rotation_angle: float = 0.0,
    rotation_reason: str = "disabled",
) -> Optional[PersonCropResult]:
    """Build an unrotated bbox-plus-margin crop without aspect-ratio padding."""
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        return None

    try:
        x = int(rect.get("x", 0))
        y = int(rect.get("y", 0))
        width = int(rect.get("w", 0))
        height = int(rect.get("h", 0))
        margin_ratio = float(margin_ratio)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None

    if (
        width <= 0
        or height <= 0
        or not math.isfinite(margin_ratio)
        or margin_ratio < 0.0
    ):
        return None

    image_height, image_width = image.shape[:2]
    if image_height <= 0 or image_width <= 0:
        return None
    if (
        x < 0
        or y < 0
        or x + width > image_width
        or y + height > image_height
    ):
        return None

    try:
        margin_x = int(math.ceil(width * margin_ratio))
        margin_y = int(math.ceil(height * margin_ratio))
    except (OverflowError, ValueError):
        return None

    requested_x1 = x - margin_x
    requested_y1 = y - margin_y
    requested_x2 = x + width + margin_x
    requested_y2 = y + height + margin_y
    target_width = requested_x2 - requested_x1
    target_height = requested_y2 - requested_y1
    if not _person_crop_output_shape_is_safe(target_height, target_width):
        return None

    source_x1 = max(0, requested_x1)
    source_y1 = max(0, requested_y1)
    source_x2 = min(image_width, requested_x2)
    source_y2 = min(image_height, requested_y2)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return None

    destination_x1 = source_x1 - requested_x1
    destination_y1 = source_y1 - requested_y1
    destination_x2 = destination_x1 + (source_x2 - source_x1)
    destination_y2 = destination_y1 + (source_y2 - source_y1)
    padding_left = destination_x1
    padding_top = destination_y1
    padding_right = target_width - destination_x2
    padding_bottom = target_height - destination_y2

    crop = np.empty((target_height, target_width, 3), dtype=image.dtype)
    crop[...] = np.asarray(algorithm_config.person.crop.border_bgr, dtype=image.dtype)
    crop[destination_y1:destination_y2, destination_x1:destination_x2] = image[
        source_y1:source_y2,
        source_x1:source_x2,
    ]

    aligned_mask = None
    region = as_person_mask(mask, image.shape)
    if region is not None:
        aligned_mask = np.zeros(
            (target_height, target_width),
            dtype=region.data.dtype,
        )
        aligned_mask[
            destination_y1:destination_y2,
            destination_x1:destination_x2,
        ] = region.window((source_x1, source_y1, source_x2, source_y2))

    valid_area = (source_x2 - source_x1) * (source_y2 - source_y1)
    canvas_area = target_width * target_height
    invalid_ratio = float(
        max(0.0, min(1.0, 1.0 - valid_area / canvas_area))
    )
    return PersonCropResult(
        image=np.ascontiguousarray(crop),
        mask=(
            None
            if aligned_mask is None
            else np.ascontiguousarray(aligned_mask)
        ),
        requested_xyxy=(
            requested_x1,
            requested_y1,
            requested_x2,
            requested_y2,
        ),
        source_xyxy=(source_x1, source_y1, source_x2, source_y2),
        content_xyxy=(
            destination_x1,
            destination_y1,
            destination_x2,
            destination_y2,
        ),
        padding=(padding_left, padding_top, padding_right, padding_bottom),
        padding_ratio=invalid_ratio,
        letterbox_padding_ratio=0.0,
        source_overflow_ratio=invalid_ratio,
        source_to_crop=np.asarray([
            [1.0, 0.0, -float(requested_x1)],
            [0.0, 1.0, -float(requested_y1)],
        ], dtype=np.float64),
        rotation_angle=float(rotation_angle),
        rotation_applied=False,
        rotation_reason=str(rotation_reason),
    )


def _source_to_crop_matrix(crop_result: PersonCropResult) -> np.ndarray:
    try:
        matrix = np.asarray(crop_result.source_to_crop, dtype=np.float64)
    except (TypeError, ValueError):
        matrix = np.empty((0, 0), dtype=np.float64)
    if matrix.shape == (2, 3) and bool(np.all(np.isfinite(matrix))):
        return matrix
    requested_x1, requested_y1, _, _ = crop_result.requested_xyxy
    return np.asarray(
        [
            [1.0, 0.0, -float(requested_x1)],
            [0.0, 1.0, -float(requested_y1)],
        ],
        dtype=np.float64,
    )


def transform_keypoints_to_person_crop(
    keypoints: Any,
    crop_result: PersonCropResult,
) -> Optional[np.ndarray]:
    """Apply the crop's authoritative affine transform to COCO-17 keypoints."""
    try:
        points = np.asarray(keypoints, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    matrix = _source_to_crop_matrix(crop_result)
    if points.shape != (17, 3) or matrix.shape != (2, 3):
        return None
    transformed = points.copy()
    homogeneous = np.column_stack(
        [points[:, 0], points[:, 1], np.ones(17, dtype=np.float32)]
    )
    transformed[:, :2] = (homogeneous @ matrix.T).astype(np.float32)
    return transformed


def project_mask_to_person_crop(
    mask: Any,
    crop_result: PersonCropResult,
    image_shape: Tuple[int, ...],
) -> Optional[np.ndarray]:
    """Project an original-coordinate mask with the crop's affine transform."""
    region = as_person_mask(mask, image_shape)
    matrix = _source_to_crop_matrix(crop_result)
    if region is None or matrix.shape != (2, 3):
        return None

    projected = _warp_source_view_nearest(
        region,
        matrix,
        crop_result.source_xyxy,
        (crop_result.image.shape[1], crop_result.image.shape[0]),
    )
    return projected


def project_rect_to_person_crop(
    rect: dict[str, Any],
    crop_result: PersonCropResult,
) -> np.ndarray:
    """Rasterize an original-image rectangle through the crop affine transform."""
    target = np.zeros(crop_result.image.shape[:2], dtype=bool)
    try:
        x1 = int(rect.get("x", 0))
        y1 = int(rect.get("y", 0))
        x2 = x1 + int(rect.get("w", 0))
        y2 = y1 + int(rect.get("h", 0))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return target
    matrix = _source_to_crop_matrix(crop_result)
    source_x1, source_y1, source_x2, source_y2 = crop_result.source_xyxy
    x1 = max(x1, source_x1)
    y1 = max(y1, source_y1)
    x2 = min(x2, source_x2)
    y2 = min(y2, source_y2)
    if x2 <= x1 or y2 <= y1 or matrix.shape != (2, 3):
        return target
    corners = np.asarray(
        [
            [x1, y1, 1],
            [x2 - 1, y1, 1],
            [x2 - 1, y2 - 1, 1],
            [x1, y2 - 1, 1],
        ],
        dtype=np.float64,
    )
    polygon = np.rint(corners @ matrix.T).astype(np.int32)
    cv2.fillConvexPoly(target.view(np.uint8), polygon, 1)
    return target
