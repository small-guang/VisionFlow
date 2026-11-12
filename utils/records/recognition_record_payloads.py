"""Payload builders for ALog Kafka recognition, detection, and vector records."""

import math
import time
from typing import Any, Dict

from utils.quality.quality_reasons import quality_reason_value
from utils.quality.face_orientation import normalize_face_orientation
from utils.quality.person_orientation import normalize_person_orientation
from utils.reid.reid_enrollment_state import normalize_reid_enrollment_state
from utils.reid.reid_state import normalize_reid_state

VECTOR_MODEL_NAMES = {
    "person": "Person ReID Vector | OpenAI CLIP ViT-B/16 (ViT-B-16.pt)",
    "person_sbs": "Person ReID Vector | FastReID SBS R50-IBN (DukeMTMC)",
    "picture": "Picture Keyframe Vector | OFA-Sys Chinese-CLIP ViT-B/16 + RoBERTa-wwm",
    "face": "Face Recognition Vector | InsightFace buffalo_l (w600k_r50.onnx)",
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _to_ratio(value: Any) -> float:
    return max(0.0, min(1.0, _to_float(value)))


def _to_edge_count(value: Any) -> int:
    return max(0, min(4, _to_int(value)))


def _to_nonnegative_int(value: Any) -> int:
    return max(0, _to_int(value))


def _to_nonnegative_float(value: Any) -> float:
    return max(0.0, _to_float(value))


def _to_bounded_nonnegative_int(value: Any, maximum: int) -> int:
    return min(maximum, _to_nonnegative_int(value))


def _to_upper_str(value: Any, default: str = "") -> str:
    return _to_str(value, default).strip().upper()


def _to_foreign_person_evidence_source(value: Any) -> str:
    source = _to_str(value, "none").strip().lower()
    return (
        source
        if source in {"none", "mask", "bbox_fallback", "mixed"}
        else "none"
    )


def _to_reid_source(value: Any) -> str:
    aliases = {
        "milvus": "MILVUS",
        "向量库": "MILVUS",
        "buffer": "BUFFER",
        "缓存": "BUFFER",
        "new": "NEW",
    }
    return "+".join(
        aliases.get(part.strip().lower(), part.strip().upper())
        for part in _to_str(value).split("+")
        if part.strip()
    )


def _created_at_ms() -> int:
    return int(time.time() * 1000)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _to_str_list(value: Any) -> list[str]:
    value = _jsonable(value)
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]

    result: list[str] = []
    for item in items:
        text = quality_reason_value(item)
        if text:
            result.append(text)
    return result


def _vector_dim(embedding: Any) -> int:
    shape = getattr(embedding, "shape", None)
    if shape:
        try:
            return int(shape[-1])
        except (TypeError, ValueError, IndexError):
            pass

    value = _jsonable(embedding)
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        first = value[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(value)
    return 0


def normalize_face_keypoint(keypoint: Any) -> Dict[str, Any]:
    points = _jsonable(keypoint)
    if points is None:
        points = []
    return {"points": points}


def normalize_person_keypoint(keypoint: Any) -> Dict[str, Any]:
    points = _jsonable(keypoint)
    if not isinstance(points, (list, tuple)) or len(points) != 17:
        return {"format": "coco17", "points": []}

    normalized: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            return {"format": "coco17", "points": []}
        try:
            values = [float(point[0]), float(point[1]), float(point[2])]
        except (TypeError, ValueError):
            return {"format": "coco17", "points": []}
        normalized.append([value if math.isfinite(value) else 0.0 for value in values])
    return {"format": "coco17", "points": normalized}


def normalize_face_rect(rect: Any) -> Dict[str, int]:
    if not isinstance(rect, dict):
        rect = {}
    return {
        "x": _to_int(rect.get("x")),
        "y": _to_int(rect.get("y")),
        "w": _to_int(rect.get("w")),
        "h": _to_int(rect.get("h")),
    }


def normalize_people_rect(rect: Any) -> Dict[str, int]:
    return normalize_face_rect(rect)


def build_vector_record(
    *,
    vector_id: Any,
    embedding: Any,
    keyframe_id: Any,
    picture_vector_id: Any,
    timestamp: int,
    keyframe_url: str,
    target_type: str,
    ai_id: str,
    dev_sn: str = "",
    vector_db_name: str = "",
    is_pano_keyframe: Any = False,
) -> Dict[str, Any]:
    target = str(target_type or "")
    return {
        "vectorId": _to_int(vector_id),
        "vectorDim": _vector_dim(embedding),
        "devSn": dev_sn or "",
        "keyframeId": _to_str(keyframe_id),
        "pictureVectorId": _to_int(picture_vector_id),
        "isPanoKeyframe": bool(is_pano_keyframe),
        "timestamp": _to_int(timestamp),
        "createdAt": _created_at_ms(),
        "keyframeUrl": keyframe_url or "",
        "modelName": VECTOR_MODEL_NAMES.get(target, ""),
        "vectorDbName": vector_db_name or "",
        "aiId": ai_id or "",
        "targetType": target,
    }


def build_face_detection_record(
    *,
    keyframe_id: Any,
    picture_vector_id: Any,
    crop_url: str,
    dev_sn: str,
    timestamp: int,
    detect_time: int,
    face_keypoint: Any,
    face_rect: Any,
    confidence: float,
    status: int,
    ai_id: str,
    info: str,
    face_quality_score: Any = 0.0,
    face_quality_level: Any = 0,
    face_orientation: Any = "unknown",
    is_pano_keyframe: Any = False,
) -> Dict[str, Any]:
    return {
        "keyframeId": _to_str(keyframe_id),
        "pictureVectorId": _to_int(picture_vector_id),
        "faceOrientation": normalize_face_orientation(face_orientation),
        "isPanoKeyframe": bool(is_pano_keyframe),
        "crop_url": crop_url or "",
        "devSn": dev_sn or "",
        "timestamp": _to_int(timestamp),
        "createdAt": _created_at_ms(),
        "detectTime": _to_int(detect_time),
        "faceKeypoint": normalize_face_keypoint(face_keypoint),
        "faceRect": normalize_face_rect(face_rect),
        "confidence": _to_float(confidence),
        "faceQualityScore": _to_float(face_quality_score),
        "faceQualityLevel": _to_int(face_quality_level),
        "status": _to_int(status),
        "aiId": ai_id or "",
        "info": info or "",
    }


def build_person_detection_record(
    *,
    keyframe_id: Any,
    picture_vector_id: Any,
    crop_url: str,
    dev_sn: str,
    timestamp: int,
    detect_time: int,
    people_rect: Any,
    confidence: float,
    mask_pixel_count: int,
    status: int,
    ai_id: str,
    info: str,
    person_quality_score: Any = 0.0,
    person_quality_level: Any = 0,
    person_quality_action: Any = "",
    mask_score: Any = 0.0,
    crop_score: Any = 0.0,
    quality_reasons: Any = None,
    person_keypoint: Any = None,
    pose_score: Any = 0.0,
    visible_keypoint_count: Any = 0,
    torso_keypoint_count: Any = 0,
    shoulder_hip_chain_count: Any = 0,
    half_body_pose_eligible: Any = False,
    body_completeness: Any = "unknown",
    is_half_body: Any = False,
    is_truncated: Any = False,
    has_second_person: Any = False,
    foreign_person_ratio: Any = 0.0,
    foreign_person_core_ratio: Any = 0.0,
    foreign_person_pixel_count: Any = 0,
    foreign_person_core_pixel_count: Any = 0,
    foreign_person_bbox_fallback_ratio: Any = 0.0,
    foreign_person_bbox_fallback_pixel_count: Any = 0,
    foreign_person_edge_dominant: Any = False,
    foreign_person_evidence_source: Any = "none",
    padding_ratio: Any = 0.0,
    letterbox_padding_ratio: Any = 0.0,
    source_overflow_ratio: Any = 0.0,
    bbox_overflow_ratio: Any = 0.0,
    near_image_edge_count: Any = 0,
    truncated_edge_count: Any = 0,
    face_orientation: Any = "unknown",
    person_orientation: Any = "unknown",
    is_pano_keyframe: Any = False,
) -> Dict[str, Any]:
    return {
        "keyframeId": _to_str(keyframe_id),
        "pictureVectorId": _to_int(picture_vector_id),
        "faceOrientation": normalize_face_orientation(face_orientation),
        "personOrientation": normalize_person_orientation(person_orientation),
        "isPanoKeyframe": bool(is_pano_keyframe),
        "crop_url": crop_url or "",
        "devSn": dev_sn or "",
        "timestamp": _to_int(timestamp),
        "createdAt": _created_at_ms(),
        "detectTime": _to_int(detect_time),
        "peopleRect": normalize_people_rect(people_rect),
        "personKeypoint": normalize_person_keypoint(person_keypoint),
        "confidence": _to_float(confidence),
        "maskPixelCount": _to_int(mask_pixel_count),
        "personQualityScore": _to_float(person_quality_score),
        "personQualityLevel": _to_int(person_quality_level),
        "personQualityAction": _to_str(person_quality_action),
        "maskScore": _to_float(mask_score),
        "cropScore": _to_float(crop_score),
        "poseScore": _to_float(pose_score),
        "visibleKeypointCount": _to_int(visible_keypoint_count),
        "torsoKeypointCount": _to_bounded_nonnegative_int(
            torso_keypoint_count, 4
        ),
        "shoulderHipChainCount": _to_bounded_nonnegative_int(
            shoulder_hip_chain_count, 2
        ),
        "halfBodyPoseEligible": bool(half_body_pose_eligible),
        "bodyCompleteness": _to_str(body_completeness, "unknown"),
        "isHalfBody": bool(is_half_body),
        "isTruncated": bool(is_truncated),
        "hasSecondPerson": bool(has_second_person),
        "foreignPersonRatio": _to_nonnegative_float(foreign_person_ratio),
        "foreignPersonCoreRatio": _to_nonnegative_float(
            foreign_person_core_ratio
        ),
        "foreignPersonPixelCount": _to_nonnegative_int(
            foreign_person_pixel_count
        ),
        "foreignPersonCorePixelCount": _to_nonnegative_int(
            foreign_person_core_pixel_count
        ),
        "foreignPersonBBoxFallbackRatio": _to_nonnegative_float(
            foreign_person_bbox_fallback_ratio
        ),
        "foreignPersonBBoxFallbackPixelCount": _to_nonnegative_int(
            foreign_person_bbox_fallback_pixel_count
        ),
        "foreignPersonEdgeDominant": bool(foreign_person_edge_dominant),
        "foreignPersonEvidenceSource": _to_foreign_person_evidence_source(
            foreign_person_evidence_source
        ),
        "paddingRatio": _to_ratio(padding_ratio),
        "letterboxPaddingRatio": _to_ratio(letterbox_padding_ratio),
        "sourceOverflowRatio": _to_ratio(source_overflow_ratio),
        "bboxOverflowRatio": _to_ratio(bbox_overflow_ratio),
        "nearImageEdgeCount": _to_edge_count(near_image_edge_count),
        "truncatedEdgeCount": _to_edge_count(truncated_edge_count),
        "qualityReasons": _to_str_list(quality_reasons),
        "status": _to_int(status),
        "aiId": ai_id or "",
        "info": info or "",
    }


def build_face_recognition_record(
    *,
    dev_sn: str,
    keyframe_id: Any,
    picture_vector_id: Any,
    face_url: str,
    timestamp: int,
    face_id: str,
    score: float,
    ai_id: str,
    face_status: int,
    info: str,
    vector_hit_face_id: str = "",
    face_quality_score: Any = 0.0,
    face_quality_level: Any = 0,
    face_orientation: Any = "unknown",
    is_pano_keyframe: Any = False,
) -> Dict[str, Any]:
    return {
        "devSn": dev_sn or "",
        "keyframeId": _to_str(keyframe_id),
        "pictureVectorId": _to_int(picture_vector_id),
        "faceOrientation": normalize_face_orientation(face_orientation),
        "isPanoKeyframe": bool(is_pano_keyframe),
        "faceUrl": face_url or "",
        "timestamp": _to_int(timestamp),
        "createdAt": _created_at_ms(),
        "aiFaceId": str(face_id or ""),
        "score": _to_float(score),
        "vectorHitFaceId": str(vector_hit_face_id or ""),
        "aiId": ai_id or "",
        "faceStatus": _to_int(face_status),
        "faceQualityScore": _to_float(face_quality_score),
        "faceQualityLevel": _to_int(face_quality_level),
        "info": info or "",
    }


def build_person_recognition_record(
    *,
    dev_sn: str,
    keyframe_id: Any,
    picture_vector_id: Any,
    image_url: str,
    face_id: str,
    reid: str,
    score: float,
    detect_time: int,
    timestamp: int,
    people_rect: Any,
    info: str,
    reid_status: int,
    reid_state: Any = "",
    reid_enrollment_state: Any = "",
    reid_enrollment_reason: Any = "",
    ai_id: str,
    vector_hit_reid: str = "",
    reid_decision: Any = "SKIPPED",
    reid_decision_reason: Any = "",
    reid_search_mode: Any = "NOT_EXECUTED",
    reid_source: Any = "",
    is_new_reid: Any = False,
    canonical_face_id: Any = "0",
    reid_top1_score: Any = 0.0,
    reid_mean_score: Any = 0.0,
    reid_class_score: Any = 0.0,
    reid_support_count: Any = 0,
    reid_runner_up_score: Any = 0.0,
    reid_margin: Any = 0.0,
    reid_required_margin: Any = 0.0,
    same_frame_excluded_reid_count: Any = 0,
    person_quality_score: Any = 0.0,
    person_quality_level: Any = 0,
    person_quality_action: Any = "",
    mask_score: Any = 0.0,
    crop_score: Any = 0.0,
    quality_reasons: Any = None,
    pose_score: Any = 0.0,
    visible_keypoint_count: Any = 0,
    torso_keypoint_count: Any = 0,
    shoulder_hip_chain_count: Any = 0,
    half_body_pose_eligible: Any = False,
    body_completeness: Any = "unknown",
    is_half_body: Any = False,
    is_truncated: Any = False,
    has_second_person: Any = False,
    foreign_person_ratio: Any = 0.0,
    foreign_person_core_ratio: Any = 0.0,
    foreign_person_pixel_count: Any = 0,
    foreign_person_core_pixel_count: Any = 0,
    foreign_person_bbox_fallback_ratio: Any = 0.0,
    foreign_person_bbox_fallback_pixel_count: Any = 0,
    foreign_person_edge_dominant: Any = False,
    foreign_person_evidence_source: Any = "none",
    padding_ratio: Any = 0.0,
    letterbox_padding_ratio: Any = 0.0,
    source_overflow_ratio: Any = 0.0,
    bbox_overflow_ratio: Any = 0.0,
    near_image_edge_count: Any = 0,
    truncated_edge_count: Any = 0,
    crop_url: str = "",
    face_orientation: Any = "unknown",
    person_orientation: Any = "unknown",
    is_pano_keyframe: Any = False,
) -> Dict[str, Any]:
    return {
        "devSn": dev_sn or "",
        "keyframeId": _to_str(keyframe_id),
        "pictureVectorId": _to_int(picture_vector_id),
        "faceOrientation": normalize_face_orientation(face_orientation),
        "personOrientation": normalize_person_orientation(person_orientation),
        "isPanoKeyframe": bool(is_pano_keyframe),
        "keyframeUrl": image_url or "",
        "crop_url": crop_url or "",
        "aiFaceId": str(face_id or "0"),
        "reid": str(reid or "0"),
        "vectorHitReid": str(vector_hit_reid or ""),
        "score": _to_float(score),
        "detectTime": _to_int(detect_time),
        "timestamp": _to_int(timestamp),
        "createdAt": _created_at_ms(),
        "peopleRect": normalize_people_rect(people_rect),
        "info": info or "",
        "reidStatus": _to_int(reid_status),
        "reidState": normalize_reid_state(reid_state),
        "reidEnrollmentState": normalize_reid_enrollment_state(
            reid_enrollment_state
        ),
        "reidEnrollmentReason": _to_str(reid_enrollment_reason),
        "reidDecision": _to_upper_str(reid_decision, "SKIPPED"),
        "reidDecisionReason": _to_str(reid_decision_reason),
        "reidSearchMode": _to_upper_str(reid_search_mode, "NOT_EXECUTED"),
        "reidSource": _to_reid_source(reid_source),
        "isNewReid": bool(is_new_reid),
        "canonicalFaceId": _to_str(canonical_face_id, "0"),
        "reidTop1Score": _to_float(reid_top1_score),
        "reidMeanScore": _to_float(reid_mean_score),
        "reidClassScore": _to_float(reid_class_score),
        "reidSupportCount": _to_nonnegative_int(reid_support_count),
        "reidRunnerUpScore": _to_float(reid_runner_up_score),
        "reidMargin": _to_float(reid_margin),
        "reidRequiredMargin": _to_float(reid_required_margin),
        "sameFrameExcludedReidCount": _to_nonnegative_int(
            same_frame_excluded_reid_count
        ),
        "personQualityScore": _to_float(person_quality_score),
        "personQualityLevel": _to_int(person_quality_level),
        "personQualityAction": _to_str(person_quality_action),
        "maskScore": _to_float(mask_score),
        "cropScore": _to_float(crop_score),
        "poseScore": _to_float(pose_score),
        "visibleKeypointCount": _to_int(visible_keypoint_count),
        "torsoKeypointCount": _to_bounded_nonnegative_int(
            torso_keypoint_count, 4
        ),
        "shoulderHipChainCount": _to_bounded_nonnegative_int(
            shoulder_hip_chain_count, 2
        ),
        "halfBodyPoseEligible": bool(half_body_pose_eligible),
        "bodyCompleteness": _to_str(body_completeness, "unknown"),
        "isHalfBody": bool(is_half_body),
        "isTruncated": bool(is_truncated),
        "hasSecondPerson": bool(has_second_person),
        "foreignPersonRatio": _to_nonnegative_float(foreign_person_ratio),
        "foreignPersonCoreRatio": _to_nonnegative_float(
            foreign_person_core_ratio
        ),
        "foreignPersonPixelCount": _to_nonnegative_int(
            foreign_person_pixel_count
        ),
        "foreignPersonCorePixelCount": _to_nonnegative_int(
            foreign_person_core_pixel_count
        ),
        "foreignPersonBBoxFallbackRatio": _to_nonnegative_float(
            foreign_person_bbox_fallback_ratio
        ),
        "foreignPersonBBoxFallbackPixelCount": _to_nonnegative_int(
            foreign_person_bbox_fallback_pixel_count
        ),
        "foreignPersonEdgeDominant": bool(foreign_person_edge_dominant),
        "foreignPersonEvidenceSource": _to_foreign_person_evidence_source(
            foreign_person_evidence_source
        ),
        "paddingRatio": _to_ratio(padding_ratio),
        "letterboxPaddingRatio": _to_ratio(letterbox_padding_ratio),
        "sourceOverflowRatio": _to_ratio(source_overflow_ratio),
        "bboxOverflowRatio": _to_ratio(bbox_overflow_ratio),
        "nearImageEdgeCount": _to_edge_count(near_image_edge_count),
        "truncatedEdgeCount": _to_edge_count(truncated_edge_count),
        "qualityReasons": _to_str_list(quality_reasons),
        "aiId": ai_id or "",
    }
