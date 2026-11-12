"""Shared helpers for keyframe face/person processing flows."""

from __future__ import annotations

import uuid
from typing import Any, Optional

import cv2
import numpy as np

from core.algorithm_config import algorithm_config
from utils.reid.reid_models import REID_SOURCE_BUFFER, REID_SOURCE_MILVUS


QUALITY_LEVEL_DROP = 0  # 质量等级：丢弃
QUALITY_LEVEL_RECOGNITION = 1  # 质量等级：仅识别
QUALITY_LEVEL_ENROLL = 2  # 质量等级：可建库

UNMATCHED_FACE_ID = "0"  # 未匹配到人脸时使用的默认 face_id
UNMATCHED_REID = "0"  # 未匹配到人型时使用的默认 reid


def encode_image_to_bytes(image: np.ndarray) -> Optional[bytes]:
    success, encoded_image = cv2.imencode(".jpg", image)
    if success:
        return encoded_image.tobytes()
    return None


async def upload_crop_bytes(
    oss_client: Any,
    image_bytes: bytes,
    device_sn: str,
    crop_type: str,
) -> str:
    crop_prefix = f"{device_sn}/{crop_type}" if device_sn else crop_type
    object_name = f"{crop_prefix}/{uuid.uuid4()}.jpg"
    uploaded_name = await oss_client.upload_object(image_bytes, object_name)
    return uploaded_name or ""


def normalize_numpy_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm == 0):
        return vector.flatten()
    return (vector / norm).flatten()


def extract_quality_score(quality_result: Any) -> float:
    if isinstance(quality_result, dict):
        value = quality_result.get("quality_score")
    else:
        value = quality_result

    if hasattr(value, "flatten"):
        flattened = value.flatten()
        if len(flattened) == 0:
            raise ValueError("empty quality score")
        value = flattened[0]
    return float(value)


def confidence_to_float(score: Any) -> float:
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def quality_level_from_score(
    score: Any,
    *,
    recognition_min_score: float,
    enroll_min_score: float,
) -> int:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return QUALITY_LEVEL_DROP
    if value >= enroll_min_score:
        return QUALITY_LEVEL_ENROLL
    if value >= recognition_min_score:
        return QUALITY_LEVEL_RECOGNITION
    return QUALITY_LEVEL_DROP


def face_quality_level_from_score(score: Any) -> int:
    return quality_level_from_score(
        score,
        recognition_min_score=algorithm_config.face.quality.recognition_min_score,
        enroll_min_score=algorithm_config.face.quality.enroll_min_score,
    )


def person_quality_level_from_score(score: Any) -> int:
    return quality_level_from_score(
        score,
        recognition_min_score=algorithm_config.person.quality.general.recognition_min_score,
        enroll_min_score=algorithm_config.person.quality.general.enroll_min_score,
    )


def person_quality_score_from_mask(mask_pixel_count: Any) -> float:
    try:
        count = max(0.0, float(mask_pixel_count))
    except (TypeError, ValueError):
        count = 0.0
    return min(
        count / algorithm_config.person.quality.general.max_mask_pixels
        * algorithm_config.person.quality.general.score_max,
        algorithm_config.person.quality.general.score_max,
    )


def valid_landmarks(landmarks: Any) -> Optional[np.ndarray]:
    try:
        points = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim == 1 and points.size % 2 == 0:
        points = points.reshape(-1, 2)
    if points.ndim != 2 or points.shape[0] < 5 or points.shape[1] < 2:
        return None
    return points[:, :2][:5]


def merge_search_results(
    milvus_id: Any,
    milvus_score: float,
    milvus_hit_id: str,
    pending_id: Any,
    pending_score: float,
    pending_hit_id: str,
):
    milvus_score = float(milvus_score or 0.0)
    pending_score = float(pending_score or 0.0)
    if pending_score > milvus_score:
        highest_score = pending_score
        highest_hit_id = str(pending_hit_id or "")
    else:
        highest_score = milvus_score
        highest_hit_id = str(milvus_hit_id or "")

    if milvus_id is not None and (pending_id is None or milvus_score >= pending_score):
        return milvus_id, highest_score, highest_hit_id
    if pending_id is not None:
        return pending_id, highest_score, highest_hit_id
    return None, highest_score, highest_hit_id


def format_reid_source(source: str) -> str:
    source_labels = {
        REID_SOURCE_MILVUS: "向量库",
        REID_SOURCE_BUFFER: "缓存",
    }
    labels = [
        source_labels.get(item, item)
        for item in str(source or "").split("+")
        if item
    ]
    return "+".join(labels)
