from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict

import numpy as np

from core.async_tasks import run_owned_parallel
from core.config import settings
from core.face_observation import FaceProcessingStage, FrameFaceObservation
from core.identity import FaceIdentityRequest, FaceIdentityService
from core.identity.ports import FaceFeatureSink, FaceRecordSink
from core.infer.errors import TritonInferenceError
from core.performance import runtime_performance
from core.queue_metrics import pipeline_metrics
from core.ordered_pipeline import run_ordered_pipeline
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_DROP,
    confidence_to_float,
    encode_image_to_bytes,
    face_quality_level_from_score,
    normalize_numpy_vector,
    upload_crop_bytes,
    valid_landmarks,
)
from utils.keyframe.detection_helpers import (
    KeyframeContext,
    clip_xyxy_to_rect,
    expand_square_rect,
    rect_to_xyxy,
)
from utils.quality.face_quality import (
    is_high_quality_feature,
    meets_pre_extraction_quality_criteria,
)
from utils.quality.face_orientation import FaceOrientation, estimate_face_orientation
from utils.quality.info_messages import describe_quality_reason
from utils.records.recognition_record_payloads import (
    build_face_detection_record,
    build_face_recognition_record,
)
from utils.storage.base import ObjectStorageClient


@dataclass
class _PreparedFace:
    ai_id: str
    detect_time: int
    confidence: float
    face_rect: dict[str, int]
    detection_info: str
    detection_status: int = 0
    crop_path: str = ""
    landmarks: list[list[float]] = field(default_factory=list)
    quality_score: float = 0.0
    quality_level: int = QUALITY_LEVEL_DROP
    observation: FrameFaceObservation | None = None
    recognition_embedding: np.ndarray | None = None


class FaceKeyframeProcessor:
    def __init__(
        self,
        *,
        triton_client: Any,
        feature_buffer: Any = None,
        face_identity_lock: asyncio.Lock | None = None,
        search_faces_with_pending: Callable[..., Any] | None = None,
        logger: Any,
        face_frame_concurrency: int | None = None,
        face_global_concurrency: int | None = None,
        face_search_service: Any = None,
        prepare_lookahead: int = 0,
        identity_service: FaceIdentityService | None = None,
        records: FaceRecordSink | None = None,
        face_features: FaceFeatureSink | None = None,
        image_executor: Any = None,
    ):
        if face_frame_concurrency is None:
            face_frame_concurrency = settings.FACE_FRAME_CONCURRENCY
        if face_global_concurrency is None:
            face_global_concurrency = settings.FACE_GLOBAL_CONCURRENCY
        for name, value in (
            ("face_frame_concurrency", face_frame_concurrency),
            ("face_global_concurrency", face_global_concurrency),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.frame_concurrency = face_frame_concurrency
        if type(prepare_lookahead) is not int or prepare_lookahead < 0:
            raise ValueError("prepare_lookahead must be a non-negative integer")
        self.prepare_lookahead = prepare_lookahead
        self.face_search_service = face_search_service
        self._preparation_slots = asyncio.Semaphore(face_global_concurrency)
        self.triton_client = triton_client
        self.feature_buffer = feature_buffer
        self.face_identity_lock = face_identity_lock
        self.search_faces_with_pending = search_faces_with_pending
        self.logger = logger
        self.records = records if records is not None else feature_buffer
        self.face_features = face_features if face_features is not None else feature_buffer
        self.identity_service = identity_service
        self.image_executor = image_executor
        self._identity_service()

    def _identity_service(self) -> FaceIdentityService:
        """兼容旧构造；正式运行由应用装配共享身份服务。"""
        service = getattr(self, "identity_service", None)
        if service is None:
            service = FaceIdentityService(
                face_features=getattr(self, "face_features", self.feature_buffer),
                records=getattr(self, "records", self.feature_buffer),
                logger=self.logger,
                identity_lock=self.face_identity_lock,
                search_service=getattr(self, "face_search_service", None),
                search_faces=self._legacy_face_search,
                metrics=runtime_performance,
                clock=time.perf_counter,
            )
            self.identity_service = service
        return service

    async def _legacy_face_search(self, *args, **kwargs):
        return await self.search_faces_with_pending(*args, **kwargs)

    async def _run_image_operation(self, operation, *args):
        executor = getattr(self, "image_executor", None)
        if executor is not None:
            return await executor.run(operation, *args)
        return await asyncio.get_running_loop().run_in_executor(None, operation, *args)

    async def process_detection(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        oss_client: ObjectStorageClient,
    ) -> list[FrameFaceObservation]:
        detections = await self.triton_client.extract_face_detections(key_frame)
        getattr(self.logger, "debug", self.logger.info)(
            f"ALog face detections: count={len(detections)}, sn={ctx.sn}"
        )
        observations: list[FrameFaceObservation] = []
        identity = self._identity_service()
        remote_identity = getattr(identity, "remote_identity", False) is True
        if remote_identity:
            await identity.begin_stage(picture_vector_id, len(detections))
        ordinal = 0

        async def prepare(detection: dict[str, Any]) -> _PreparedFace:
            # Shared across all frames handled by this processor. Identity waits
            # must not hold preparation slots needed by other faces or frames.
            async with pipeline_metrics.slot("face_prepare", self._preparation_slots):
                prepared = await self._prepare_face(ctx, key_frame, oss_client, detection)
            return prepared

        async def commit(prepared: _PreparedFace) -> None:
            nonlocal ordinal
            with pipeline_metrics.activity("face_commit"):
                observation = await self._commit_face(
                    ctx, key_frame_id, picture_vector_id,
                    key_frame_object_name, prepared,
                    **({"ordinal": ordinal} if remote_identity else {}),
                )
            if observation is not None:
                observations.append(observation)
            if remote_identity:
                await identity.complete_detection(picture_vector_id, ordinal)
            ordinal += 1

        await run_ordered_pipeline(
            detections, prepare, commit,
            concurrency=self.frame_concurrency, lookahead=self.prepare_lookahead,
            stage_name="face_ordered_commit", metrics=pipeline_metrics,
        )
        if remote_identity:
            await identity.end_stage(picture_vector_id)
        return observations

    async def _prepare_face(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        oss_client: ObjectStorageClient,
        detection: dict[str, Any],
    ) -> _PreparedFace:
        face_rect = clip_xyxy_to_rect(detection.get("xyxy", []), key_frame.shape)
        prepared = _PreparedFace(
            ai_id=str(uuid.uuid4()),
            detect_time=int(time.time() * 1000),
            confidence=confidence_to_float(detection.get("conf")),
            face_rect=face_rect or {"x": 0, "y": 0, "w": 0, "h": 0},
            detection_info="ALog人脸检测框无效",
        )
        if face_rect is None:
            return prepared

        source_xyxy = rect_to_xyxy(face_rect)
        prepared.observation = FrameFaceObservation(
            source_rect=source_xyxy, confidence=prepared.confidence,
        )
        expanded_xyxy = expand_square_rect(face_rect, key_frame.shape)
        if expanded_xyxy is None:
            prepared.detection_info = "扩展人脸裁剪区域无效"
            prepared.observation = replace(
                prepared.observation, failure_reason=prepared.detection_info,
            )
            return prepared
        prepared.observation = replace(
            prepared.observation, expanded_source_rect=expanded_xyxy,
        )

        x, y, x2, y2 = source_xyxy
        safe_x1, safe_y1, safe_x2, safe_y2 = expanded_xyxy
        expanded_crop = key_frame[safe_y1:safe_y2, safe_x1:safe_x2]
        face_crop = key_frame[y:y2, x:x2]
        bbox_in_expanded = np.array(
            [x - safe_x1, y - safe_y1, x2 - safe_x1, y2 - safe_y1],
            dtype=np.float32,
        )
        if expanded_crop.size == 0 or face_crop.size == 0:
            prepared.detection_info = "人脸裁剪图为空"
            prepared.observation = replace(
                prepared.observation, failure_reason=prepared.detection_info,
            )
            return prepared

        image_bytes = await self._run_image_operation(encode_image_to_bytes, expanded_crop)

        async def upload() -> None:
            if image_bytes:
                prepared.crop_path = await upload_crop_bytes(
                    oss_client, image_bytes, ctx.sn, "face_crop",
                )
            else:
                self.logger.warning("Failed to encode face crop")
            if not prepared.crop_path:
                self.logger.warning(f"Face crop upload failed: sn={ctx.sn}")

        # 关键点/质量只依赖像素，可与上传重叠；特征提取仍等待上传成功。
        _, landmarks = await run_owned_parallel(
            upload(),
            self._prepare_face_quality(
                prepared, key_frame.shape, expanded_crop, face_crop,
                bbox_in_expanded, (safe_x1, safe_y1),
            ),
        )
        if landmarks is None:
            return prepared
        if not prepared.crop_path:
            prepared.observation = replace(
                prepared.observation, failure_reason="Face crop upload failed",
            )
            return prepared

        try:
            features = await self.triton_client.extract_face_features(expanded_crop, landmarks)
            embedding = features.vector
        except Exception as error:
            self.logger.error(f"Failed to extract face features: {error}")
            prepared.observation = replace(
                prepared.observation, failure_reason=f"特征提取失败：{error}",
            )
            return prepared

        raw_embedding = np.asarray(embedding, dtype=np.float32).flatten()
        raw_embedding_norm = float(np.linalg.norm(raw_embedding))
        shared_normalized = (
            await self._run_image_operation(normalize_numpy_vector, raw_embedding)
            if np.all(np.isfinite(raw_embedding)) and raw_embedding_norm > 0.0
            else None
        )
        prepared.observation = replace(
            prepared.observation, normalized_embedding=shared_normalized,
            raw_embedding_norm=(raw_embedding_norm if np.isfinite(raw_embedding_norm) else None),
            completed_stage=FaceProcessingStage.FEATURE,
        )
        is_good, reason = is_high_quality_feature(embedding)
        if not is_good:
            self.logger.debug(f"Face feature rejected: {reason}")
            prepared.observation = replace(
                prepared.observation, failure_reason=f"Face feature rejected: {reason}",
            )
            return prepared
        prepared.recognition_embedding = (
            shared_normalized
            if shared_normalized is not None
            else await self._run_image_operation(normalize_numpy_vector, embedding)
        )
        return prepared

    async def _prepare_face_quality(
        self,
        prepared: _PreparedFace,
        frame_shape: tuple[int, ...],
        expanded_crop: np.ndarray,
        face_crop: np.ndarray,
        bbox_in_expanded: np.ndarray,
        expanded_origin: tuple[int, int],
    ) -> np.ndarray | None:
        """只准备像素派生结果，不访问存储或身份状态。"""
        x, y, x2, y2 = rect_to_xyxy(prepared.face_rect)
        assert prepared.observation is not None
        try:
            landmarks = await self.triton_client.extract_face_keypoints(
                expanded_crop, bbox_in_expanded,
            )
        except TritonInferenceError as error:
            self.logger.warning(f"Face keypoint inference failed: {error}")
            landmarks = None
        landmarks = valid_landmarks(landmarks)
        if landmarks is None:
            prepared.detection_info = "人脸关键点不可用"
            prepared.observation = replace(
                prepared.observation, failure_reason=prepared.detection_info,
            )
            return None

        landmarks_full = landmarks.copy()
        landmarks_full[:, 0] += expanded_origin[0]
        landmarks_full[:, 1] += expanded_origin[1]
        face_orientation = estimate_face_orientation(landmarks_full, frame_shape).label
        prepared.landmarks = landmarks_full.tolist()
        prepared.observation = replace(
            prepared.observation, source_landmarks=landmarks_full,
            orientation=face_orientation, completed_stage=FaceProcessingStage.LANDMARKS,
        )
        is_good, reason = meets_pre_extraction_quality_criteria(
            face_crop, landmarks_full, [x, y, x2, y2], self.logger,
            pose_image_shape=frame_shape,
        )
        if not is_good:
            prepared.detection_info = f"人脸预检未通过：{describe_quality_reason(reason)}"
            prepared.observation = replace(
                prepared.observation, failure_reason=prepared.detection_info,
            )
            return None

        try:
            quality_result = await self.triton_client.extract_face_quality(expanded_crop, landmarks)
            quality_score = quality_result.score
        except Exception as error:
            prepared.detection_info = f"人脸质量评分失败：{error}"
            prepared.observation = replace(
                prepared.observation, failure_reason=prepared.detection_info,
            )
            return None

        prepared.quality_score = quality_score
        prepared.quality_level = face_quality_level_from_score(quality_score)
        prepared.observation = replace(
            prepared.observation, quality_score=quality_score,
            completed_stage=FaceProcessingStage.QUALITY,
        )
        if prepared.quality_level == QUALITY_LEVEL_DROP:
            prepared.detection_info = (
                f"人脸质量分 {quality_score:.2f}，"
                f"质量等级 {prepared.quality_level}，已丢弃"
            )
            prepared.observation = replace(
                prepared.observation, failure_reason=prepared.detection_info,
            )
            return None

        prepared.detection_status = 1
        prepared.detection_info = (
            f"人脸质量分 {quality_score:.2f}，"
            f"质量等级 {prepared.quality_level}，已通过"
        )
        return landmarks

    async def _commit_face(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        prepared: _PreparedFace,
        *,
        ordinal: int | None = None,
    ) -> FrameFaceObservation | None:
        observation = prepared.observation
        face_orientation = (
            observation.orientation if observation is not None else FaceOrientation.UNKNOWN.value
        )
        await self.queue_face_detection_record(
            ctx, key_frame_id, picture_vector_id, prepared.crop_path,
            prepared.landmarks, prepared.face_rect, prepared.ai_id, prepared.detect_time,
            status=prepared.detection_status, confidence=prepared.confidence,
            info=prepared.detection_info, face_quality_score=prepared.quality_score,
            face_quality_level=prepared.quality_level, face_orientation=face_orientation,
        )
        if prepared.recognition_embedding is None:
            return observation

        assert observation is not None
        request = FaceIdentityRequest(
            embedding=prepared.recognition_embedding,
            quality_level=prepared.quality_level,
            feature_metadata={
                "device_id": ctx.sn,
                "keyframe_id": key_frame_id,
                "picture_vector_id": picture_vector_id,
                "keyframe_url": key_frame_object_name,
                "face_rect": prepared.face_rect,
                "face_key_points": prepared.landmarks,
                "face_crop_path": prepared.crop_path,
                "timestamp": ctx.timestamp,
                "ai_id": prepared.ai_id,
                "quality_score": prepared.quality_score,
                "is_pano_keyframe": ctx.is_pano_keyframe,
            },
        )

        def build_record(outcome):
            return self._build_face_recognition_record(
                ctx, key_frame_id, picture_vector_id, prepared.crop_path,
                outcome.face_id, outcome.score, outcome.vector_hit_face_id,
                prepared.ai_id, outcome.status, outcome.info,
                face_quality_score=prepared.quality_score,
                face_quality_level=prepared.quality_level,
                face_orientation=face_orientation,
            )

        identity = self._identity_service()
        if getattr(identity, "remote_identity", False) is True:
            outcome = await identity.commit_record(request, {
                "dev_sn": ctx.sn, "keyframe_id": key_frame_id,
                "picture_vector_id": picture_vector_id,
                "face_url": prepared.crop_path, "timestamp": ctx.timestamp,
                "ai_id": prepared.ai_id, "face_quality_score": prepared.quality_score,
                "face_quality_level": prepared.quality_level,
                "face_orientation": face_orientation,
                "is_pano_keyframe": ctx.is_pano_keyframe,
            }, ordinal=ordinal)
        else:
            outcome = await identity.commit(request, build_record)
        return replace(
            observation, assigned_face_id=outcome.face_id,
            recognition_score=outcome.score,
            completed_stage=(
                FaceProcessingStage.FEATURE if outcome.search_failure_reason
                else FaceProcessingStage.IDENTITY
            ),
            failure_reason=outcome.search_failure_reason,
        )

    async def queue_face_detection_record(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        crop_image_url: str,
        face_keypoint: Any,
        face_rect: Dict[str, int],
        ai_id: str,
        detect_time: int,
        status: int,
        confidence: float,
        info: str,
        face_quality_score: float = 0.0,
        face_quality_level: int = QUALITY_LEVEL_DROP,
        face_orientation: str = FaceOrientation.UNKNOWN.value,
    ):
        record = build_face_detection_record(
            keyframe_id=key_frame_id,
            picture_vector_id=picture_vector_id,
            crop_url=crop_image_url,
            dev_sn=ctx.sn,
            timestamp=ctx.timestamp,
            detect_time=detect_time,
            face_keypoint=face_keypoint,
            face_rect=face_rect,
            confidence=confidence,
            status=status,
            ai_id=ai_id,
            info=info,
            face_quality_score=face_quality_score,
            face_quality_level=face_quality_level,
            face_orientation=face_orientation,
            is_pano_keyframe=ctx.is_pano_keyframe,
        )
        await self.records.add_face_detection_record(record)

    def _build_face_recognition_record(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        face_crop_path: str,
        assigned_face_id: str,
        score: float,
        vector_hit_face_id: str,
        ai_id: str,
        face_status: int,
        info: str,
        face_quality_score: float = 0.0,
        face_quality_level: int = QUALITY_LEVEL_DROP,
        face_orientation: str = FaceOrientation.UNKNOWN.value,
    ) -> Dict[str, Any]:
        return build_face_recognition_record(
            dev_sn=ctx.sn,
            keyframe_id=key_frame_id,
            picture_vector_id=picture_vector_id,
            face_url=face_crop_path,
            timestamp=ctx.timestamp,
            face_id=assigned_face_id,
            score=score,
            vector_hit_face_id=vector_hit_face_id,
            ai_id=ai_id,
            face_status=face_status,
            info=info,
            face_quality_score=face_quality_score,
            face_quality_level=face_quality_level,
            face_orientation=face_orientation,
            is_pano_keyframe=ctx.is_pano_keyframe,
        )

    async def queue_face_recognition_record(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        face_crop_path: str,
        assigned_face_id: str,
        score: float,
        vector_hit_face_id: str,
        ai_id: str,
        face_status: int,
        info: str,
        face_quality_score: float = 0.0,
        face_quality_level: int = QUALITY_LEVEL_DROP,
        face_orientation: str = FaceOrientation.UNKNOWN.value,
    ):
        record = self._build_face_recognition_record(
            ctx,
            key_frame_id,
            picture_vector_id,
            face_crop_path,
            assigned_face_id,
            score,
            vector_hit_face_id,
            ai_id,
            face_status,
            info,
            face_quality_score=face_quality_score,
            face_quality_level=face_quality_level,
            face_orientation=face_orientation,
        )
        await self.records.add_face_recognition_record(record)
