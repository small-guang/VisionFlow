"""Pure helpers for keyframe self-detection flows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import unquote, urlparse

import numpy as np

from core.algorithm_config import algorithm_config
from utils.person_mask import PersonMask, as_person_mask

PERSON_CLASS_ID = 0  # COCO 数据集中 person 类别的 class_id


@dataclass(frozen=True)
class KeyframeContext:
    sn: str
    timestamp: int
    object_name: str
    keyframe_id: str = ""
    is_pano_keyframe: bool = False


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_non_empty(values: Iterable[Any]) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _strip_url_and_query(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        # Load storage settings only for URLs; raw keys keep their legacy rules.
        from core.config import settings

        if getattr(settings, "STORAGE_BACKEND", "oss") == "minio":
            endpoint_scheme = "https" if settings.MINIO_SECURE else "http"
            endpoint = urlparse(f"{endpoint_scheme}://{settings.MINIO_ENDPOINT}")
            default_port = 443 if endpoint_scheme == "https" else 80
            url_default_port = 443 if parsed.scheme == "https" else 80
            endpoint_port = endpoint.port if endpoint.port is not None else default_port
            url_port = parsed.port if parsed.port is not None else url_default_port
            if (
                parsed.scheme not in ("http", "https")
                or parsed.username is not None
                or parsed.password is not None
                or parsed.hostname != endpoint.hostname
                or url_port != endpoint_port
            ):
                raise ValueError("keyframe URL does not match configured MinIO endpoint")
            path = parsed.path
            if parsed.params:
                path += f";{parsed.params}"
            bucket, separator, object_name = path.removeprefix("/").partition("/")
            if unquote(bucket) != settings.MINIO_BUCKET_NAME:
                raise ValueError("keyframe URL does not match configured MinIO bucket")
            if not separator or not object_name:
                raise ValueError("empty keyframe object name")
            return unquote(object_name)
        value = parsed.path
    else:
        value = value.split("?", 1)[0]
    return unquote(value).strip().lstrip("/")


def infer_device_sn_from_object_name(object_name: str) -> str:
    if "/" not in object_name:
        return ""
    return object_name.split("/", 1)[0].strip()


def _normalize_keyframe_object_name(
    keyframe_value: str,
    data: Dict[str, Any],
) -> str:
    object_name = _strip_url_and_query(keyframe_value)
    if not object_name:
        raise ValueError("empty keyframe object name")

    if object_name.lower().endswith((".jpg", ".jpeg")):
        return object_name

    if "/" in object_name:
        return f"{object_name}.jpg"

    device_sn = _first_non_empty(
        [data.get("sn"), data.get("devSn"), data.get("deviceSn")]
    )
    if device_sn:
        return f"{device_sn}/picture/{object_name}.jpg"
    return f"{object_name}.jpg"


def resolve_keyframe_object_name(data: Dict[str, Any]) -> str:
    keyframe_value = _first_non_empty(
        [
            data.get("keyframe"),
            data.get("keyFrame"),
            data.get("keyframeUrl"),
            data.get("keyFrameUrl"),
            data.get("imageUrl"),
            data.get("image_url"),
        ]
    )
    if not keyframe_value:
        raise ValueError("missing keyframe object name")
    return _normalize_keyframe_object_name(keyframe_value, data)


def _build_keyframe_context(
    data: Dict[str, Any],
    object_name: str,
    now_ms: Optional[int],
) -> KeyframeContext:
    sn = _first_non_empty(
        [
            data.get("sn"),
            data.get("devSn"),
            data.get("deviceSn"),
            infer_device_sn_from_object_name(object_name),
        ]
    )
    timestamp = _to_int(
        _first_non_empty([data.get("timestamp"), data.get("time"), data.get("ts")]),
        default=now_ms if now_ms is not None else int(time.time() * 1000),
    )
    keyframe_id = _first_non_empty([data.get("uId")])
    if not keyframe_id:
        raise ValueError("missing uId")
    return KeyframeContext(
        sn=sn,
        timestamp=timestamp,
        object_name=object_name,
        keyframe_id=keyframe_id,
    )


def parse_keyframe_context(
    data: Dict[str, Any],
    now_ms: Optional[int] = None,
) -> KeyframeContext:
    return _build_keyframe_context(
        data,
        resolve_keyframe_object_name(data),
        now_ms,
    )


def parse_pano_keyframe_context(
    data: Dict[str, Any],
    now_ms: Optional[int] = None,
) -> Optional[KeyframeContext]:
    pano_keyframe_value = _first_non_empty([data.get("panoKeyframeUrl")])
    if not pano_keyframe_value:
        return None
    main_ctx = parse_keyframe_context(data, now_ms=now_ms)
    return KeyframeContext(
        sn=main_ctx.sn,
        timestamp=main_ctx.timestamp,
        object_name=_normalize_keyframe_object_name(
            pano_keyframe_value,
            {**data, "sn": main_ctx.sn},
        ),
        keyframe_id=main_ctx.keyframe_id,
        is_pano_keyframe=True,
    )


def clip_xyxy_to_rect(
    xyxy: Iterable[Any], image_shape: Tuple[int, ...]
) -> Optional[Dict[str, int]]:
    values = list(xyxy)
    if len(values) < 4 or len(image_shape) < 2:
        return None

    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    x1 = int(max(0, min(img_w, round(float(values[0])))))
    y1 = int(max(0, min(img_h, round(float(values[1])))))
    x2 = int(max(0, min(img_w, round(float(values[2])))))
    y2 = int(max(0, min(img_h, round(float(values[3])))))

    if x2 <= x1 or y2 <= y1:
        return None
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def rect_to_xyxy(rect: Dict[str, int]) -> Tuple[int, int, int, int]:
    x = int(rect.get("x", 0))
    y = int(rect.get("y", 0))
    w = int(rect.get("w", 0))
    h = int(rect.get("h", 0))
    return x, y, x + w, y + h


def expand_square_rect(
    rect: Dict[str, int],
    image_shape: Tuple[int, ...],
    scale_factor: float = algorithm_config.face.crop.scale_factor,
) -> Optional[Tuple[int, int, int, int]]:
    x, y, x2, y2 = rect_to_xyxy(rect)
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    w = x2 - x
    h = y2 - y
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None

    center_x = x + w / 2
    center_y = y + h / 2
    new_side = max(w, h) * scale_factor
    safe_x1 = int(max(0, center_x - new_side / 2))
    safe_y1 = int(max(0, center_y - new_side / 2))
    safe_x2 = int(min(img_w, center_x + new_side / 2))
    safe_y2 = int(min(img_h, center_y + new_side / 2))

    if safe_x2 <= safe_x1 or safe_y2 <= safe_y1:
        return None
    return safe_x1, safe_y1, safe_x2, safe_y2


def parse_person_segments_result(
    detections: Any,
    masks: Any,
    num_detections: Any,
    image_shape: Tuple[int, ...],
    person_class_id: int = PERSON_CLASS_ID,
) -> list[Dict[str, Any]]:
    if detections is None or masks is None or num_detections is None:
        return []

    count = _to_int(np.asarray(num_detections).reshape(-1)[0], default=0)
    if count <= 0:
        return []

    rows = np.asarray(detections, dtype=np.float32)[:count]
    mask_array = np.asarray(masks)
    persons: list[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        if len(row) < 6 or int(row[5]) != person_class_id:
            continue
        if index >= len(mask_array):
            continue

        rect = clip_xyxy_to_rect(row[:4], image_shape)
        if rect is None:
            continue

        mask = np.asarray(mask_array[index], dtype=np.uint8)
        mask_pixel_count = int(np.count_nonzero(mask))
        persons.append(
            {
                "xyxy": [float(value) for value in row[:4]],
                "rect": rect,
                "conf": float(row[4]),
                "cls": int(row[5]),
                "mask": mask,
                "maskPixelCount": mask_pixel_count,
            }
        )

    return persons


def parse_person_poses_result(
    detections: Any,
    keypoints: Any,
    num_detections: Any,
    image_shape: Tuple[int, ...],
    person_class_id: int = PERSON_CLASS_ID,
) -> list[Dict[str, Any]]:
    """Parse Triton YOLO pose outputs into whole-image person poses."""
    if detections is None or num_detections is None:
        return []

    count_values = np.asarray(num_detections).reshape(-1)
    if count_values.size == 0:
        return []
    count = _to_int(count_values[0], default=0)
    if count <= 0:
        return []

    try:
        rows = np.asarray(detections, dtype=np.float32)
    except (TypeError, ValueError):
        return []
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        return []
    rows = rows[:count]
    try:
        keypoint_array = (
            None if keypoints is None else np.asarray(keypoints, dtype=np.float32)
        )
        if keypoint_array is not None and keypoint_array.ndim == 2:
            keypoint_array = keypoint_array.reshape(1, *keypoint_array.shape)
    except (TypeError, ValueError):
        keypoint_array = None

    poses: list[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if (
            len(row) < 6
            or not bool(np.all(np.isfinite(row[:6])))
            or int(row[5]) != person_class_id
        ):
            continue

        rect = clip_xyxy_to_rect(row[:4], image_shape)
        if rect is None:
            continue

        pose_keypoints = None
        if keypoint_array is not None and index < len(keypoint_array):
            pose_keypoints = np.asarray(keypoint_array[index], dtype=np.float32)

        poses.append(
            {
                "xyxy": [float(value) for value in row[:4]],
                "rect": rect,
                "conf": float(row[4]),
                "cls": int(row[5]),
                "keypoints": pose_keypoints,
            }
        )
    return poses


def rect_iou(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    """Calculate IoU for two ``x/y/w/h`` rectangles."""
    first_x1, first_y1, first_x2, first_y2 = rect_to_xyxy(first)
    second_x1, second_y1, second_x2, second_y2 = rect_to_xyxy(second)

    intersection_w = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_h = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_w * intersection_h
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def person_segments_can_share_reid(
    first: Dict[str, Any],
    second: Dict[str, Any],
    *,
    bbox_iou_threshold: float = algorithm_config.person.assignment.reid_duplicate_bbox_iou,
    mask_iou_threshold: float = algorithm_config.person.assignment.reid_duplicate_mask_iou,
) -> bool:
    """Return whether two detections are likely duplicates of the same person."""
    first_rect = first.get("rect") or {}
    second_rect = second.get("rect") or {}
    if rect_iou(first_rect, second_rect) < float(bbox_iou_threshold):
        return False

    first_mask = as_person_mask(first.get("mask"))
    second_mask = as_person_mask(second.get("mask"))
    if (
        first_mask is None
        or second_mask is None
        or first_mask.image_shape != second_mask.image_shape
    ):
        return False

    intersection = first_mask.intersection_count(second_mask)
    union = first_mask.count() + second_mask.count() - intersection
    if union <= 0:
        return False
    return float(intersection / union) >= float(mask_iou_threshold)


def same_frame_reid_exclusions(
    assignments: Dict[str, Iterable[Dict[str, Any]]],
    current_segment: Dict[str, Any],
) -> set[str]:
    """Return assigned ReIDs that cannot be reused by the current segment."""
    return {
        str(reid)
        for reid, assigned_segments in assignments.items()
        if str(reid)
        and any(
            not person_segments_can_share_reid(
                assigned_segment,
                current_segment,
            )
            for assigned_segment in assigned_segments
        )
    }


def pose_mask_agreement(
    visible_points: np.ndarray,
    mask: PersonMask | np.ndarray,
    rect: Dict[str, Any],
) -> float:
    """Return the share of visible pose points supported by a person mask."""
    if visible_points.size == 0:
        return 0.0
    region = as_person_mask(mask)
    if region is None:
        return 0.0

    try:
        rect_width = int(rect.get("w", 0))
        rect_height = int(rect.get("h", 0))
    except (AttributeError, TypeError, ValueError):
        rect_width = 0
        rect_height = 0
    assignment = algorithm_config.person.assignment
    radius = max(
        assignment.pose_mask_sample_radius_min_px,
        int(
            round(
                min(rect_width, rect_height)
                * assignment.pose_mask_sample_radius_ratio
            )
        ),
    )
    hits = 0
    for x_value, y_value, _ in visible_points:
        x = int(round(float(x_value)))
        y = int(round(float(y_value)))
        if region.count((x - radius, y - radius, x + radius + 1, y + radius + 1)):
            hits += 1
    return float(hits / len(visible_points))


def _pose_visible_points(pose: Dict[str, Any]) -> np.ndarray:
    try:
        points = np.asarray(pose.get("keypoints"), dtype=np.float32)
    except (TypeError, ValueError):
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape != (17, 3):
        return np.empty((0, 3), dtype=np.float32)
    visible = np.all(np.isfinite(points), axis=1) & (
        points[:, 2] >= algorithm_config.person.assignment.pose_keypoint_visible_threshold
    )
    return points[visible]


def _segment_mask(segment: Dict[str, Any]) -> PersonMask | None:
    mask = as_person_mask(segment.get("mask"))
    if mask is None or mask.count() == 0:
        return None
    return mask


def match_person_segments_to_poses(
    segments: list[Dict[str, Any]],
    poses: list[Dict[str, Any]],
    *,
    min_iou: float = algorithm_config.person.assignment.pose_match_min_iou,
) -> list[Dict[str, Any]]:
    """Attach poses to masks using deterministic mask-first one-to-one matching."""
    assignment = algorithm_config.person.assignment
    matched_segments = [
        {
            **segment,
            "pose": None,
            "poseMatchIou": 0.0,
            "poseMaskAgreement": 0.0,
            "poseMatchScore": 0.0,
        }
        for segment in segments
    ]
    candidates: list[tuple[float, float, float, int, int]] = []
    for segment_index, segment in enumerate(matched_segments):
        segment_rect = segment.get("rect") or {}
        mask = _segment_mask(segment)
        if mask is None:
            continue
        for pose_index, pose in enumerate(poses):
            iou = rect_iou(segment_rect, pose.get("rect") or {})
            if iou < min_iou:
                continue
            visible_points = _pose_visible_points(pose)
            mask_agreement = pose_mask_agreement(
                visible_points,
                mask,
                segment_rect,
            )
            if mask_agreement < assignment.pose_match_min_mask_agreement:
                continue
            match_score = (
                assignment.pose_match_mask_weight * mask_agreement
                + assignment.pose_match_iou_weight * iou
            )
            candidates.append(
                (match_score, mask_agreement, iou, segment_index, pose_index)
            )

    used_segments: set[int] = set()
    used_poses: set[int] = set()
    for match_score, mask_agreement, iou, segment_index, pose_index in sorted(
        candidates,
        key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]),
    ):
        if segment_index in used_segments or pose_index in used_poses:
            continue
        matched_segments[segment_index]["pose"] = poses[pose_index]
        matched_segments[segment_index]["poseMatchIou"] = float(iou)
        matched_segments[segment_index]["poseMaskAgreement"] = float(mask_agreement)
        matched_segments[segment_index]["poseMatchScore"] = float(match_score)
        used_segments.add(segment_index)
        used_poses.add(pose_index)

    return matched_segments
