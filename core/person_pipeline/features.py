"""SBS/CLIP feature extraction, face matching, and CLIP storage."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from core.algorithm_config import algorithm_config
from core.infer.results import FeatureResult
from core.identity.ports import PersonClipSink
from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_ENROLL,
    normalize_numpy_vector,
)
from utils.quality.face_orientation import normalize_face_orientation
from utils.reid.person_identity import IdentityOutcome

from .face_reuse import FrameFaceBinding, PersonFaceReuseService
from .models import (
    FaceMatchOutcome,
    PersonEmbeddingBundle,
    PersonPipelineRuntime,
    PreparedPerson,
)


class PersonFeatureService:
    def __init__(
        self,
        *,
        triton_client: Any,
        feature_buffer: Any = None,
        search_faces_with_pending_locked: Callable[..., Any],
        logger: Any,
        runtime: PersonPipelineRuntime,
        reuse_mode: str,
        clip_features: PersonClipSink | None = None,
        image_executor: Any = None,
    ) -> None:
        if reuse_mode not in {"off", "on"}:
            raise ValueError(f"invalid person face reuse mode: {reuse_mode}")
        self.triton_client = triton_client
        self.clip_features = clip_features if clip_features is not None else getattr(feature_buffer, "tbps", None)
        self.search_faces_with_pending_locked = search_faces_with_pending_locked
        self.logger = logger
        self.runtime = runtime
        self.image_executor = image_executor
        self.reuse_mode = reuse_mode
        self.reuse_service = PersonFaceReuseService(
            triton_client=triton_client,
            search_faces_with_pending_locked=search_faces_with_pending_locked,
            logger=logger,
            image_executor=image_executor,
        )

    async def extract_embeddings(
        self,
        prepared: PreparedPerson,
    ) -> PersonEmbeddingBundle:
        image_data = prepared.image_bytes
        calls = [self.triton_client.extract_sbs_features(image_data)]
        if prepared.quality_decision.person_quality_level == QUALITY_LEVEL_ENROLL:
            calls.append(self.triton_client.extract_tbps_image_features(image_data))
        results = await asyncio.gather(*calls, return_exceptions=True)
        sbs, sbs_failure, sbs_reason = self._parse_sbs(results[0])
        clip = None
        clip_failure = ""
        if prepared.quality_decision.person_quality_level == QUALITY_LEVEL_ENROLL:
            clip, clip_failure = self._parse_clip(results[1])
        return PersonEmbeddingBundle(
            sbs=sbs,
            clip=clip,
            sbs_failure=sbs_failure,
            sbs_failure_reason=sbs_reason,
            clip_failure=clip_failure,
        )

    def _parse_sbs(self, result: Any) -> tuple[np.ndarray | None, str, str]:
        if isinstance(result, BaseException):
            failure = f"SBS 推理失败：{result}"
            self.logger.error(failure)
            return None, failure, "sbs_inference_failed"
        if isinstance(result, FeatureResult):
            candidate = result.vector
        else:
            failure = "SBS 推理响应格式无效"
            self.logger.warning(failure)
            return None, failure, "sbs_embedding_invalid"
        try:
            candidate = np.asarray(candidate, dtype=np.float32).flatten()
        except (TypeError, ValueError, OverflowError):
            failure = "SBS embedding 无效，无法转换为 FP32 向量"
            self.logger.warning(failure)
            return None, failure, "sbs_embedding_invalid"
        if candidate.size != 2048 or not np.all(np.isfinite(candidate)):
            failure = f"SBS embedding 无效，维度={candidate.size}"
            self.logger.warning(failure)
            return None, failure, "sbs_embedding_invalid"
        if float(np.linalg.norm(candidate)) <= 0.0:
            failure = "SBS embedding 为零向量"
            self.logger.warning(failure)
            return None, failure, "sbs_embedding_zero"
        return normalize_numpy_vector(candidate), "", ""

    def _parse_clip(self, result: Any) -> tuple[np.ndarray | None, str]:
        if isinstance(result, BaseException):
            failure = f"CLIP 推理失败：{result}"
            self.logger.error(failure)
            return None, failure
        if isinstance(result, FeatureResult):
            candidate = result.vector
        else:
            failure = "CLIP 推理响应格式无效"
            self.logger.warning(failure)
            return None, failure
        try:
            candidate = np.asarray(candidate, dtype=np.float32).flatten()
        except (TypeError, ValueError, OverflowError):
            failure = "CLIP embedding 无效，无法转换为 FP32 向量"
            self.logger.warning(failure)
            return None, failure
        if candidate.size != 512 or not np.all(np.isfinite(candidate)):
            failure = f"CLIP embedding 无效，维度={candidate.size}"
            self.logger.warning(failure)
            return None, failure
        if float(np.linalg.norm(candidate)) <= 0.0:
            failure = "CLIP embedding 为零向量"
            self.logger.warning(failure)
            return None, failure
        return normalize_numpy_vector(candidate), ""

    async def match_face(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        prepared: PreparedPerson,
        binding: FrameFaceBinding,
        *,
        enable_recognition: bool,
    ) -> FaceMatchOutcome:
        if self.reuse_mode == "off":
            return await self._match_local_face(
                ctx,
                prepared,
                enable_recognition=enable_recognition,
            )

        if bool(getattr(prepared.crop_result, "rotation_applied", False)):
            local = await self._match_local_face(
                ctx,
                prepared,
                enable_recognition=enable_recognition,
            )
            return replace(
                local,
                fallback_reason="person_rotated",
                shared_fallback_reason="person_rotated",
            )

        shared = await self.reuse_service.attempt(
            ctx=ctx,
            key_frame=key_frame,
            binding=binding,
            enable_recognition=enable_recognition,
        )
        if shared.outcome is not None:
            if shared.outcome.face_id != "0" or not enable_recognition:
                return shared.outcome

        local = await self._match_local_face(
            ctx,
            prepared,
            enable_recognition=enable_recognition,
        )
        shared_diagnostics = {
            "fallback_reason": shared.fallback_reason,
            "shared_face_id": (
                shared.outcome.face_id if shared.outcome is not None else "0"
            ),
            "shared_orientation": (
                shared.outcome.orientation
                if shared.outcome is not None
                else "unknown"
            ),
            "shared_source": shared.source,
            "shared_fallback_reason": shared.fallback_reason,
        }
        return replace(local, **shared_diagnostics)

    async def _match_local_face(
        self,
        ctx: KeyframeContext,
        prepared: PreparedPerson,
        *,
        enable_recognition: bool,
    ) -> FaceMatchOutcome:
        callback = self.runtime.match_face
        try:
            keypoints = self.runtime.pose_keypoints_to_crop(
                prepared.observation.person_keypoints,
                prepared.crop_result,
            )
            callback_kwargs = {
                "person_crop": prepared.cropped,
                "person_mask": prepared.mask_crop,
                "pose_keypoints": keypoints,
                "device_id": ctx.sn,
                "timestamp": ctx.timestamp,
                "triton_client": self.triton_client,
                "search_callback": self.search_faces_with_pending_locked,
                "normalize_vector": normalize_numpy_vector,
                "similarity_threshold": algorithm_config.face.matching.general_similarity_threshold,
                "logger": self.logger,
                "enable_recognition": enable_recognition,
            }
            if self.image_executor is not None:
                callback_kwargs["image_executor"] = self.image_executor
            result = await callback(
                **callback_kwargs,
            )
            return FaceMatchOutcome(
                result["face_id"],
                result["info"],
                normalize_face_orientation(result.get("face_orientation")),
            )
        except Exception as error:
            self.logger.exception(
                f"Unexpected person face recognition error, using faceId=0: {error}"
            )
            return FaceMatchOutcome(
                "0",
                f"人型裁剪图人脸识别流程失败：{error}",
            )

    async def store_clip(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        prepared: PreparedPerson,
        embeddings: PersonEmbeddingBundle,
        face: FaceMatchOutcome,
        identity: IdentityOutcome,
    ) -> None:
        quality = prepared.quality_decision
        if (
            quality.person_quality_level != QUALITY_LEVEL_ENROLL
            or embeddings.clip is None
        ):
            return
        observation = prepared.observation
        await self.clip_features.add_feature(
            face_id=face.face_id,
            re_id=identity.reid,
            embedding=embeddings.clip,
            device_id=ctx.sn,
            keyframe_id=key_frame_id,
            picture_vector_id=picture_vector_id,
            keyframe_url=key_frame_object_name,
            person_rect=observation.people_rect,
            timestamp=ctx.timestamp,
            ai_id=observation.ai_id,
            is_pano_keyframe=ctx.is_pano_keyframe,
        )
