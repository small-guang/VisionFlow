import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.algorithm_config import algorithm_config
from utils.person_mask import as_person_mask

from core.face_observation import normalize_face_id_scalar
from utils.keyframe.detection_helpers import clip_xyxy_to_rect, expand_square_rect, rect_to_xyxy
from utils.keyframe.keyframe_flow_helpers import normalize_numpy_vector
from utils.quality.face_orientation import (
    FaceOrientation,
    estimate_face_orientation,
    normalize_face_orientation,
)
from utils.quality.info_messages import describe_quality_reason

try:
    from loguru import logger as default_logger
except ImportError:
    default_logger = logging.getLogger(__name__)


UNMATCHED_FACE_ID = "0"  # 人型未匹配到人脸时使用的默认 face_id
PERSON_FACE_NORMALIZED_EMBEDDING_NORM_TOLERANCE = 1e-4  # 复用外部归一化人脸特征时，允许其 L2 范数相对 1.0 的最大误差
PERSON_FACE_HEAD_KEYPOINT_INDICES = (0, 1, 2, 3, 4)  # 鼻、双眼、双耳


@dataclass(frozen=True)
class OwnedFaceSelection:
    detection_index: int
    bbox: tuple[int, int, int, int]
    info: str


def meets_pre_extraction_quality_criteria(*args, **kwargs):
    from utils.quality.face_quality import meets_pre_extraction_quality_criteria as impl

    return impl(*args, **kwargs)


def is_high_quality_feature(*args, **kwargs):
    from utils.quality.face_quality import is_high_quality_feature as impl

    return impl(*args, **kwargs)


def _face_detection_rect(
    detection: Any,
    image_shape: tuple[int, ...],
) -> dict[str, int] | None:
    if not isinstance(detection, dict):
        return None
    try:
        return clip_xyxy_to_rect(detection.get("xyxy", []), image_shape)
    except (TypeError, ValueError, OverflowError):
        return None


def _select_owned_face_detection_with_info(
    detections: list[Any],
    image_shape: tuple[int, ...],
    person_mask: Any,
    pose_keypoints: Any = None,
) -> tuple[OwnedFaceSelection | None, str]:
    mask = as_person_mask(person_mask, image_shape)
    if mask is None or mask.count() == 0:
        return None, f"人脸归属失败：人物mask缺失或尺寸无效，检测到 {len(detections)} 张人脸"

    head_anchor: tuple[float, float] | None = None
    try:
        points = np.asarray(pose_keypoints, dtype=np.float32)
    except (TypeError, ValueError):
        points = np.empty((0, 0), dtype=np.float32)
    if points.ndim == 2 and points.shape == (17, 3):
        head_points = points[list(PERSON_FACE_HEAD_KEYPOINT_INDICES)]
        visible = np.all(np.isfinite(head_points), axis=1) & (
            head_points[:, 2] >= algorithm_config.person_face.ownership.pose_visible_threshold
        )
        if bool(np.any(visible)):
            visible_head_points = head_points[visible, :2]
            head_anchor = (
                float(np.median(visible_head_points[:, 0])),
                float(np.median(visible_head_points[:, 1])),
            )

    candidates: list[dict[str, Any]] = []
    valid_detection_count = 0
    for detection_index, detection in enumerate(detections):
        rect = _face_detection_rect(detection, image_shape)
        if rect is None:
            continue
        valid_detection_count += 1
        bbox = rect_to_xyxy(rect)
        x1, y1, x2, y2 = bbox
        face_area = (x2 - x1) * (y2 - y1)
        if face_area <= 0:
            continue
        coverage = float(mask.count(bbox) / face_area)
        center_x = min(mask.image_shape[1] - 1, max(0, int((x1 + x2 - 1) / 2)))
        center_y = min(mask.image_shape[0] - 1, max(0, int((y1 + y2 - 1) / 2)))
        center_inside = mask.count((center_x, center_y, center_x + 1, center_y + 1)) > 0
        if not center_inside or coverage < algorithm_config.person_face.ownership.mask_min_coverage:
            continue

        width = x2 - x1
        height = y2 - y1
        try:
            confidence = float(detection.get("conf", 0.0))
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        head_contains = False
        normalized_head_distance = float("inf")
        if head_anchor is not None:
            head_x, head_y = head_anchor
            margin_x = width * algorithm_config.person_face.ownership.head_bbox_margin_ratio
            margin_y = height * algorithm_config.person_face.ownership.head_bbox_margin_ratio
            head_contains = bool(
                x1 - margin_x <= head_x <= x2 + margin_x
                and y1 - margin_y <= head_y <= y2 + margin_y
            )
            normalized_head_distance = float(
                np.hypot(head_x - (x1 + x2) / 2.0, head_y - (y1 + y2) / 2.0)
                / max(float(np.hypot(width, height)), 1.0)
            )
        candidates.append(
            {
                "detection_index": detection_index,
                "bbox": bbox,
                "coverage": coverage,
                "confidence": confidence,
                "area": width * height,
                "head_contains": head_contains,
                "head_distance": normalized_head_distance,
            }
        )

    if not candidates:
        return (
            None,
            f"人脸归属失败：{valid_detection_count} 张有效检测均不满足人物mask中心点和"
            f"覆盖率>={algorithm_config.person_face.ownership.mask_min_coverage:.2f}",
        )

    if head_anchor is not None:
        candidates = [candidate for candidate in candidates if candidate["head_contains"]]
        if not candidates:
            return None, "人脸归属失败：通过人物mask筛选的候选均不匹配Pose头部"
        candidates.sort(
            key=lambda item: (
                int(item["head_contains"]),
                -float(item["head_distance"]),
                float(item["coverage"]),
                float(item["confidence"]),
                int(item["area"]),
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda item: (
                float(item["coverage"]),
                float(item["confidence"]),
                int(item["area"]),
            ),
            reverse=True,
        )
    selected = candidates[0]
    pose_info = (
        f"，Pose头部匹配={bool(selected['head_contains'])}"
        if head_anchor is not None
        else "，无可用Pose头部点"
    )
    info = (
        f"人脸归属候选 {len(candidates)}/{valid_detection_count}，"
        f"选中mask覆盖率 {float(selected['coverage']):.3f}，"
        f"检测置信度 {float(selected['confidence']):.3f}{pose_info}"
    )
    return (
        OwnedFaceSelection(
            detection_index=int(selected["detection_index"]),
            bbox=selected["bbox"],
            info=info,
        ),
        info,
    )


def select_owned_face_detection(
    detections: list[Any],
    image_shape: tuple[int, ...],
    person_mask: Any,
    pose_keypoints: Any = None,
) -> OwnedFaceSelection | None:
    selection, _ = _select_owned_face_detection_with_info(
        detections, image_shape, person_mask, pose_keypoints
    )
    return selection


def _select_best_face_bbox(
    detections: list[Any],
    image_shape: tuple[int, ...],
    person_mask: Any,
    pose_keypoints: Any = None,
) -> tuple[tuple[int, int, int, int] | None, str]:
    selection, info = _select_owned_face_detection_with_info(
        detections, image_shape, person_mask, pose_keypoints
    )
    if selection is None:
        return None, info
    return selection.bbox, selection.info


def _bbox_to_rect(bbox: tuple[int, int, int, int]) -> dict[str, int]:
    x1, y1, x2, y2 = bbox
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _face_match_result(
    face_id: str,
    info: str,
    score: float = 0.0,
    *,
    face_orientation: Any = FaceOrientation.UNKNOWN,
    pitch: float | None = None,
    yaw: float | None = None,
    roll: float | None = None,
) -> dict[str, Any]:
    return {
        "face_id": normalize_face_id_scalar(face_id),
        "info": info,
        "score": float(score),
        "face_orientation": normalize_face_orientation(face_orientation),
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }


def _has_valid_person_mask(person_mask: Any, image_shape: tuple[int, ...]) -> bool:
    mask = as_person_mask(person_mask, image_shape)
    return mask is not None and mask.count() > 0


async def recognize_person_face_candidate_with_info(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    device_id: str,
    timestamp: int,
    triton_client: Any,
    search_callback: Callable[..., Awaitable[tuple[Any, float, str]]],
    *,
    landmarks_in_image: Any = None,
    quality_score: float | None = None,
    normalized_embedding: Any = None,
    raw_embedding_norm: float | None = None,
    ownership_info: str = "",
    normalize_vector: Callable[[np.ndarray], np.ndarray] = normalize_numpy_vector,
    similarity_threshold: float = algorithm_config.face.matching.general_similarity_threshold,
    logger: Any = default_logger,
    enable_recognition: bool = True,
    image_executor: Any = None,
) -> dict[str, Any]:
    active_logger = logger or default_logger
    face_orientation = FaceOrientation.UNKNOWN.value
    pitch = None
    yaw = None
    roll = None

    def make_result(face_id: str, info: str, score: float = 0.0) -> dict[str, Any]:
        details = "；".join(item for item in (info, ownership_info) if item)
        return _face_match_result(
            face_id,
            details,
            score,
            face_orientation=face_orientation,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
        )

    try:
        if image is None or image.size == 0:
            raise ValueError
        bbox_values = list(bbox)
        if len(bbox_values) != 4:
            raise ValueError
        face_rect = clip_xyxy_to_rect(bbox_values, image.shape)
    except (AttributeError, TypeError, ValueError, OverflowError):
        face_rect = None
    if face_rect is None:
        return make_result(UNMATCHED_FACE_ID, "检测到人脸但裁剪区域无效")

    x1, y1, x2, y2 = rect_to_xyxy(face_rect)
    expanded_bbox = expand_square_rect(
        face_rect,
        image.shape,
        scale_factor=algorithm_config.person_face.binding.crop_scale_factor,
    )
    if expanded_bbox is None:
        return make_result(
            UNMATCHED_FACE_ID,
            "检测到人脸但裁剪区域无效",
        )

    safe_x1, safe_y1, safe_x2, safe_y2 = expanded_bbox
    expanded_crop = image[safe_y1:safe_y2, safe_x1:safe_x2]
    face_crop = image[y1:y2, x1:x2]
    if expanded_crop.size == 0 or face_crop.size == 0:
        return make_result(UNMATCHED_FACE_ID, "检测到人脸但裁剪区域无效")

    bbox_in_expanded = np.array(
        [x1 - safe_x1, y1 - safe_y1, x2 - safe_x1, y2 - safe_y1],
        dtype=np.float32,
    )

    supplied_landmarks = None
    if landmarks_in_image is not None:
        try:
            candidate_landmarks = np.asarray(landmarks_in_image, dtype=np.float32)
            if candidate_landmarks.shape == (5, 2) and bool(
                np.all(np.isfinite(candidate_landmarks))
            ):
                supplied_landmarks = candidate_landmarks
        except (TypeError, ValueError):
            supplied_landmarks = None

    if supplied_landmarks is not None:
        landmarks = supplied_landmarks.copy()
        landmarks[:, 0] -= safe_x1
        landmarks[:, 1] -= safe_y1
    else:
        try:
            inferred_landmarks = await triton_client.extract_face_keypoints(
                expanded_crop, bbox_in_expanded
            )
        except Exception as exc:
            active_logger.warning(
                f"Person face keypoint extraction failed, using faceId=0: device={device_id}, "
                f"timestamp={timestamp}, error={exc}"
            )
            return make_result(UNMATCHED_FACE_ID, f"人脸关键点提取失败: {exc}")

        if inferred_landmarks is None:
            return make_result(UNMATCHED_FACE_ID, "人脸关键点为空，无法识别")
        try:
            inferred_landmarks = np.asarray(inferred_landmarks, dtype=np.float32)
        except (TypeError, ValueError):
            return make_result(UNMATCHED_FACE_ID, "人脸关键点格式无效，无法识别")
        if inferred_landmarks.size < 10:
            return make_result(UNMATCHED_FACE_ID, "人脸关键点少于5个，无法识别")
        try:
            landmarks = inferred_landmarks.reshape(-1, 2)[:5]
        except ValueError:
            return make_result(UNMATCHED_FACE_ID, "人脸关键点格式无效，无法识别")

    try:
        orientation_result = estimate_face_orientation(
            landmarks,
            expanded_crop.shape,
        )
        face_orientation = normalize_face_orientation(
            orientation_result.orientation
        )
        pitch = orientation_result.pitch
        yaw = orientation_result.yaw
        roll = orientation_result.roll
    except Exception as exc:
        active_logger.warning(
            f"Person face orientation estimation failed: device={device_id}, "
            f"timestamp={timestamp}, error={exc}"
        )
        face_orientation = FaceOrientation.UNKNOWN.value
        pitch = None
        yaw = None
        roll = None

    if not enable_recognition:
        return make_result(UNMATCHED_FACE_ID, "已完成人脸朝向分析，未执行人脸识别")

    is_good, reason = meets_pre_extraction_quality_criteria(
        face_crop,
        landmarks,
        bbox_in_expanded.tolist(),
        active_logger,
        pose_image_shape=expanded_crop.shape,
        laplacian_threshold=algorithm_config.person_face.binding.laplacian_threshold,
        min_face_width=algorithm_config.person_face.binding.min_face_width_px,
        max_pose_abs_degrees=algorithm_config.person_face.binding.max_pose_abs_degrees,
        geometry_std_ratio_threshold=algorithm_config.person_face.binding.geometry_std_ratio_threshold,
    )
    if not is_good:
        active_logger.debug(
            f"Person face rejected before feature extraction: device={device_id}, "
            f"reason={reason}"
        )
        return make_result(
            UNMATCHED_FACE_ID,
            f"人脸预检未通过：{describe_quality_reason(reason)}",
        )

    try:
        resolved_quality_score = (
            float(quality_score) if quality_score is not None else None
        )
    except (TypeError, ValueError, OverflowError):
        resolved_quality_score = None
    if resolved_quality_score is None or not np.isfinite(resolved_quality_score):
        try:
            quality_result = await triton_client.extract_face_quality(
                expanded_crop, landmarks
            )
            resolved_quality_score = float(quality_result.score)
        except Exception as exc:
            active_logger.warning(
                f"Person face quality extraction failed, using faceId=0: device={device_id}, "
                f"timestamp={timestamp}, error={exc}"
            )
            return make_result(UNMATCHED_FACE_ID, f"人脸质量评分失败: {exc}")

    if resolved_quality_score < algorithm_config.person_face.binding.quality_min_score:
        active_logger.debug(
            f"Person face quality score {resolved_quality_score:.2f} below threshold "
            f"{algorithm_config.person_face.binding.quality_min_score:.2f}: device={device_id}"
        )
        return make_result(
            UNMATCHED_FACE_ID,
            f"人脸质量评分 {resolved_quality_score:.2f} 低于阈值 "
            f"{algorithm_config.person_face.binding.quality_min_score:.2f}",
        )

    shared_normalized = None
    shared_raw_norm = None
    try:
        candidate_normalized = np.asarray(normalized_embedding, dtype=np.float32)
        candidate_raw_norm = float(raw_embedding_norm)
        candidate_normalized_norm = float(np.linalg.norm(candidate_normalized))
        if (
            candidate_normalized.ndim == 1
            and candidate_normalized.size > 0
            and bool(np.all(np.isfinite(candidate_normalized)))
            and abs(candidate_normalized_norm - 1.0)
            <= PERSON_FACE_NORMALIZED_EMBEDDING_NORM_TOLERANCE
            and np.isfinite(candidate_raw_norm)
            and candidate_raw_norm >= 0.0
        ):
            shared_normalized = candidate_normalized
            shared_raw_norm = candidate_raw_norm
    except (TypeError, ValueError, OverflowError):
        pass

    if shared_normalized is not None and shared_raw_norm is not None:
        embedding = shared_normalized * shared_raw_norm
        normalized_for_search = shared_normalized
    else:
        try:
            features = await triton_client.extract_face_features(expanded_crop, landmarks)
            embedding = features.vector
        except Exception as exc:
            active_logger.warning(
                f"Person face feature extraction failed, using faceId=0: device={device_id}, "
                f"timestamp={timestamp}, error={exc}"
            )
            return make_result(UNMATCHED_FACE_ID, f"人脸特征提取失败: {exc}")
        normalized_for_search = None

    is_good, reason = is_high_quality_feature(
        embedding,
        norm_threshold=algorithm_config.person_face.binding.feature_norm_threshold,
    )
    if not is_good:
        active_logger.debug(
            f"Person face feature rejected: device={device_id}, reason={reason}"
        )
        return make_result(
            UNMATCHED_FACE_ID,
            f"人脸特征质量未通过：{describe_quality_reason(reason)}",
        )

    if normalized_for_search is None:
        if image_executor is not None:
            normalized_for_search = await image_executor.run(normalize_vector, embedding)
        else:
            loop = asyncio.get_running_loop()
            normalized_for_search = await loop.run_in_executor(
                None, normalize_vector, embedding
            )

    try:
        matched_face_id, vector_score, _ = await search_callback(
            normalized_for_search,
            similarity_threshold,
            None,
        )
    except Exception as exc:
        active_logger.warning(
            f"Person face gallery search failed, using faceId=0: device={device_id}, "
            f"timestamp={timestamp}, error={exc}"
        )
        return make_result(UNMATCHED_FACE_ID, f"人脸向量库检索失败: {exc}")

    normalized_face_id = normalize_face_id_scalar(matched_face_id)
    if normalized_face_id == UNMATCHED_FACE_ID:
        return make_result(
            UNMATCHED_FACE_ID,
            f"人脸向量未匹配，最高相似度 {float(vector_score):.4f}",
            vector_score,
        )
    return make_result(
        normalized_face_id,
        f"人脸向量匹配face_id={normalized_face_id}，最高相似度 {float(vector_score):.4f}",
        vector_score,
    )


async def match_face_id_from_person_crop_with_info(
    person_crop: np.ndarray,
    device_id: str,
    timestamp: int,
    triton_client: Any,
    search_callback: Callable[
        [np.ndarray, float, list | None], Awaitable[tuple[Any, float, str]]
    ],
    normalize_vector: Callable[[np.ndarray], np.ndarray] = normalize_numpy_vector,
    similarity_threshold: float = algorithm_config.face.matching.general_similarity_threshold,
    logger: Any = default_logger,
    person_mask: Any = None,
    pose_keypoints: Any = None,
    enable_recognition: bool = True,
    *,
    image_executor: Any = None,
) -> dict[str, Any]:
    active_logger = logger or default_logger
    ownership_info = ""
    face_orientation = FaceOrientation.UNKNOWN.value

    def make_result(face_id: str, info: str, score: float = 0.0) -> dict[str, Any]:
        details = "；".join(item for item in (info, ownership_info) if item)
        return _face_match_result(
            face_id,
            details,
            score,
            face_orientation=face_orientation,
        )

    if person_crop is None or person_crop.size == 0:
        return make_result(UNMATCHED_FACE_ID, "人脸检测失败: 人型裁剪图为空")

    try:
        detections = await triton_client.extract_face_detections(person_crop)
    except Exception as exc:
        active_logger.warning(
            f"Person face detection failed, using faceId=0: device={device_id}, "
            f"timestamp={timestamp}, error={exc}"
        )
        return make_result(UNMATCHED_FACE_ID, f"人脸检测失败：{exc}")

    if not isinstance(detections, list):
        return make_result(UNMATCHED_FACE_ID, "人脸检测响应格式无效，无法判断人脸朝向")

    if not _has_valid_person_mask(person_mask, person_crop.shape):
        ownership_info = (
            f"人脸归属失败：人物mask缺失或尺寸无效，检测到 {len(detections)} 张人脸"
        )
        return make_result(UNMATCHED_FACE_ID, "未找到属于当前人型的人脸")

    if not detections:
        face_orientation = FaceOrientation.NO_FACE.value
        return make_result(UNMATCHED_FACE_ID, "未检测到人脸")

    valid_detection_count = sum(
        _face_detection_rect(detection, person_crop.shape) is not None
        for detection in detections
    )
    if valid_detection_count == 0:
        return make_result(
            UNMATCHED_FACE_ID,
            "人脸检测结果均无有效边界框，无法判断人脸朝向",
        )

    selection, ownership_info = _select_owned_face_detection_with_info(
        detections,
        person_crop.shape,
        person_mask,
        pose_keypoints,
    )
    if selection is None:
        face_orientation = FaceOrientation.NO_FACE.value
        active_logger.debug(
            f"No face detected in person crop: device={device_id}, timestamp={timestamp}"
        )
        return make_result(UNMATCHED_FACE_ID, "未找到属于当前人型的人脸")

    x1, y1, x2, y2 = selection.bbox
    ownership_info = (
        f"{selection.info}，人脸框crop坐标 "
        f"x={x1},y={y1},w={x2 - x1},h={y2 - y1}"
    )
    return await recognize_person_face_candidate_with_info(
        image=person_crop,
        bbox=selection.bbox,
        device_id=device_id,
        timestamp=timestamp,
        triton_client=triton_client,
        search_callback=search_callback,
        ownership_info=ownership_info,
        normalize_vector=normalize_vector,
        similarity_threshold=similarity_threshold,
        logger=active_logger,
        enable_recognition=enable_recognition,
        image_executor=image_executor,
    )
