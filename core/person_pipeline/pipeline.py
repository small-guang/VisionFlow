"""Thin orchestration across person-pipeline stages."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

from core.async_tasks import run_owned_parallel
from core.face_observation import FrameFaceObservation
from core.queue_metrics import pipeline_metrics
from core.ordered_pipeline import run_ordered_pipeline
from utils.keyframe.detection_helpers import (
    KeyframeContext,
    person_segments_can_share_reid,
    same_frame_reid_exclusions,
)
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_DROP
from utils.quality.info_messages import append_quality_reasons_info
from utils.reid.person_identity import (
    IdentityOutcome,
    IdentityRequest,
    PersonIdentityService,
)

from .features import PersonFeatureService
from .face_reuse import (
    FaceReuseFrameStats,
    FrameFaceBinding,
    bind_frame_faces_to_people,
)
from .models import (
    FaceMatchOutcome,
    PersonAnalysis,
    PersonEmbeddingBundle,
    PersonObservation,
    PersonPipelineRuntime,
    PersonPreparationFailure,
    PreparedPerson,
)
from .observation import PersonObservationService
from .preparation import PersonPreparationService


def duplicate_person_ordinals(segments):
    """保留精确的两两 bbox/mask 关系；相似关系不可传递合并。"""
    return [
        tuple(previous for previous in range(ordinal)
              if person_segments_can_share_reid(segments[previous], current))
        for ordinal, current in enumerate(segments)
    ]


class PersonDetectionPipeline:
    def __init__(
        self,
        *,
        triton_client: Any,
        feature_buffer: Any = None,
        search_faces_with_pending_locked: Callable[..., Any],
        identity_service: PersonIdentityService,
        detection_emitter: Callable[..., Any],
        recognition_emitter: Callable[..., Any],
        logger: Any,
        runtime: PersonPipelineRuntime,
        upright_device_sns: frozenset[str],
        person_face_reuse_mode: str,
        person_frame_concurrency: int = 4,
        person_global_concurrency: int = 16,
        prepare_lookahead: int = 0,
        person_face_concurrency: int | None = None,
        clip_features: Any = None,
        image_executor: Any,
    ) -> None:
        if person_face_concurrency is None:
            person_face_concurrency = person_global_concurrency
        for name, value in (
            ("person_frame_concurrency", person_frame_concurrency),
            ("person_global_concurrency", person_global_concurrency),
            ("person_face_concurrency", person_face_concurrency),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.frame_concurrency = person_frame_concurrency
        if type(prepare_lookahead) is not int or prepare_lookahead < 0:
            raise ValueError("prepare_lookahead must be a non-negative integer")
        self.prepare_lookahead = prepare_lookahead
        self._preparation_slots = asyncio.Semaphore(person_global_concurrency)
        self._face_resolution_slots = asyncio.Semaphore(person_face_concurrency)
        self.identity_service = identity_service
        self.detection_emitter = detection_emitter
        self.recognition_emitter = recognition_emitter
        self.logger = logger
        self.reuse_mode = person_face_reuse_mode
        self.observations = PersonObservationService(
            triton_client=triton_client,
            logger=logger,
            image_executor=image_executor,
        )
        self.preparation = PersonPreparationService(
            logger=logger,
            runtime=runtime,
            upright_device_sns=upright_device_sns,
            image_executor=image_executor,
        )
        self.features = PersonFeatureService(
            triton_client=triton_client,
            feature_buffer=feature_buffer,
            search_faces_with_pending_locked=search_faces_with_pending_locked,
            logger=logger,
            runtime=runtime,
            reuse_mode=person_face_reuse_mode,
            clip_features=clip_features,
            image_executor=image_executor,
        )

    async def process(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        oss_client: Any,
        *,
        frame_faces: Sequence[FrameFaceObservation] = (),
    ) -> None:
        observations = await self.observations.detect(ctx, key_frame)
        if self.reuse_mode == "off":
            bindings = [
                FrameFaceBinding(reason="reuse_off")
                for _ in observations
            ]
        else:
            bindings = bind_frame_faces_to_people(
                frame_faces,
                observations,
                key_frame.shape,
            )
        stats = FaceReuseFrameStats(
            person_count=len(observations),
            frame_face_count=len(frame_faces),
            bound_count=sum(binding.reason == "bound" for binding in bindings),
        )
        all_segments = [observation.segment for observation in observations]
        frame_assignments: dict[str, list[dict[str, Any]]] = {}
        remote_identity = getattr(getattr(self, "identity_service", None), "remote_identity", False) is True
        duplicates = None
        if remote_identity:
            await self.identity_service.begin_stage(picture_vector_id, len(observations))
            duplicates = await self.preparation._run_image_operation(
                duplicate_person_ordinals, all_segments,
            )
        ordinal = 0

        async def prepare(
            item: tuple[PersonObservation, FrameFaceBinding],
        ) -> PersonAnalysis | PersonPreparationFailure:
            observation, binding = item
            # 本阶段的任务寿命仍有帧窗口限制；视觉计算与 Face 解析分别接纳。
            # Face 查询等待不会占用其他帧的视觉准备额度。
            with pipeline_metrics.activity("person_prepare"):
                result = await self._prepare_observation(
                    ctx,
                    key_frame,
                    oss_client,
                    observation,
                    all_segments,
                    binding,
                )
            return result

        async def commit(analysis: PersonAnalysis | PersonPreparationFailure) -> None:
            nonlocal ordinal
            with pipeline_metrics.activity("person_commit"):
                await self._commit_observation(
                    ctx,
                    key_frame_id,
                    picture_vector_id,
                    key_frame_object_name,
                    analysis,
                    frame_assignments,
                    stats,
                    **({"ordinal": ordinal, "duplicate_with_ordinals": duplicates[ordinal]}
                       if remote_identity else {}),
                )
            if remote_identity:
                await self.identity_service.complete_detection(picture_vector_id, ordinal)
            ordinal += 1
        await run_ordered_pipeline(
            zip(observations, bindings), prepare, commit,
            concurrency=self.frame_concurrency, lookahead=self.prepare_lookahead,
            stage_name="person_ordered_commit", metrics=pipeline_metrics,
        )
        if remote_identity:
            await self.identity_service.end_stage(picture_vector_id)
        getattr(self.logger, "debug", self.logger.info)(
            "Person face reuse frame summary: "
            f"mode={self.reuse_mode}, people={stats.person_count}, "
            f"frame_faces={stats.frame_face_count}, bound={stats.bound_count}, "
            f"direct={stats.direct_id_reuse_count}, "
            f"partial={stats.partial_reuse_count}, "
            f"orientation={stats.shared_orientation_count}, "
            f"fallback={stats.local_fallback_count}, "
            f"fallback_recovered={stats.fallback_recovered_count}, "
            f"id_disagreement={stats.id_disagreement_count}, "
            f"id_shared_only={stats.shared_nonzero_local_zero_count}, "
            f"id_local_only={stats.local_nonzero_shared_zero_count}, "
            f"id_different={stats.different_nonzero_id_count}, "
            f"orientation_disagreement={stats.orientation_disagreement_count}, "
            f"fallback_reasons={stats.format_fallback_reasons()}, "
            f"rotated={stats.rotated_person_count}, "
            f"shared_success_rotated={stats.shared_success_rotated_count}, "
            f"shared_success_unrotated={stats.shared_success_unrotated_count}, "
            f"fallback_rotated={stats.local_fallback_rotated_count}, "
            f"fallback_unrotated={stats.local_fallback_unrotated_count}, "
            f"recovery_rotated={stats.local_nonzero_recovery_rotated_count}, "
            f"recovery_unrotated={stats.local_nonzero_recovery_unrotated_count}"
        )

    async def _prepare_observation(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        oss_client: Any,
        observation: PersonObservation,
        all_segments: list[dict[str, Any]],
        frame_face_binding: FrameFaceBinding,
    ) -> PersonAnalysis | PersonPreparationFailure:
        if observation.people_rect["w"] <= 0 or observation.people_rect["h"] <= 0:
            return PersonPreparationFailure(
                observation, "ALog人型分割检测框无效",
            )
        async with pipeline_metrics.slot("person_visual_prepare", self._preparation_slots):
            prepared = await self.preparation.prepare_visual(
                ctx,
                key_frame,
                observation,
                all_segments,
            )
        if prepared is None:
            return PersonPreparationFailure(
                self.preparation.drop_empty_crop(observation),
                "人型裁剪图为空",
            )
        enable_face_recognition = bool(
            prepared.quality_decision.person_quality_level != QUALITY_LEVEL_DROP
            and prepared.image_bytes
        )

        async def upload_and_match_face() -> tuple[PreparedPerson, FaceMatchOutcome]:
            uploaded = await self.preparation.upload_prepared(ctx, oss_client, prepared)
            # 人脸匹配会访问身份候选，仍等上传结束后才发起。
            async with pipeline_metrics.slot("person_face_match", self._face_resolution_slots):
                face = await self.features.match_face(
                    ctx,
                    key_frame,
                    uploaded,
                    frame_face_binding,
                    enable_recognition=enable_face_recognition,
                )
            return uploaded, face

        async def extract_embeddings() -> PersonEmbeddingBundle:
            async with pipeline_metrics.slot("person_visual_prepare", self._preparation_slots):
                with pipeline_metrics.activity("person_embeddings"):
                    return await self.features.extract_embeddings(prepared)

        if not enable_face_recognition:
            uploaded, face = await upload_and_match_face()
            return PersonAnalysis(uploaded, face, None)

        # 编码后的纯特征不依赖 crop_path。两个子任务仍属于当前帧的
        # N + M 窗口，完成上传、匹配与推理后才交付按序身份提交。
        (uploaded, face), embeddings = await run_owned_parallel(
            upload_and_match_face(), extract_embeddings(),
        )
        return PersonAnalysis(uploaded, face, embeddings)

    async def _commit_observation(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        analysis: PersonAnalysis | PersonPreparationFailure,
        frame_assignments: dict[str, list[dict[str, Any]]],
        stats: FaceReuseFrameStats,
        *,
        ordinal: int | None = None,
        duplicate_with_ordinals: tuple[int, ...] = (),
    ) -> None:
        if isinstance(analysis, PersonPreparationFailure):
            await self._emit_preparation_failure(
                ctx,
                key_frame_id,
                picture_vector_id,
                analysis.observation,
                analysis.info,
            )
            return
        prepared = analysis.prepared
        face = analysis.face
        embeddings = analysis.embeddings
        stats.record(face, prepared.crop_result.rotation_applied)
        await self._emit_detection(
            ctx,
            key_frame_id,
            picture_vector_id,
            prepared,
            face.orientation,
        )
        if embeddings is None:
            return
        identity = await self._resolve_identity(
            ctx,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            prepared,
            embeddings,
            face,
            frame_assignments,
            **({"ordinal": ordinal, "duplicate_with_ordinals": duplicate_with_ordinals}
               if ordinal is not None else {}),
        )
        await self.features.store_clip(
            ctx,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            prepared,
            embeddings,
            face,
            identity,
        )
        await self._emit_recognition(
            ctx,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            prepared,
            embeddings,
            face,
            identity,
        )

    async def _emit_preparation_failure(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        observation: PersonObservation,
        message: str,
    ) -> None:
        quality = observation.quality_decision
        await self.detection_emitter(
            ctx,
            key_frame_id,
            picture_vector_id,
            "",
            observation.people_rect,
            observation.ai_id,
            observation.detect_time,
            status=0,
            confidence=observation.confidence,
            mask_pixel_count=observation.mask_pixel_count,
            info=append_quality_reasons_info(message, quality.quality_reasons),
            person_keypoint=observation.person_keypoints,
            quality_decision=quality,
            face_orientation="unknown",
            person_orientation=observation.person_orientation,
        )

    async def _emit_detection(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        prepared: PreparedPerson,
        face_orientation: str,
    ) -> None:
        observation = prepared.observation
        quality = prepared.quality_decision
        dropped = quality.person_quality_level == QUALITY_LEVEL_DROP
        pose_info = (
            f"，pose分 {quality.pose_score:.2f}，"
            f"可见关键点 {quality.visible_keypoint_count}，"
            f"完整度 {quality.body_completeness}"
        )
        info = append_quality_reasons_info(
            (
                f"人型质量分 {quality.person_quality_score:.2f}，"
                f"质量等级 {quality.person_quality_level}，"
                f"{'已丢弃' if dropped else '已通过'}，"
                f"mask像素数 {observation.mask_pixel_count}"
                f"{pose_info}{prepared.crop_info}{prepared.visual_info}"
            ),
            quality.quality_reasons,
        )
        await self.detection_emitter(
            ctx,
            key_frame_id,
            picture_vector_id,
            prepared.crop_path,
            observation.people_rect,
            observation.ai_id,
            observation.detect_time,
            status=0 if dropped else 1,
            confidence=observation.confidence,
            mask_pixel_count=observation.mask_pixel_count,
            info=info,
            person_keypoint=observation.person_keypoints,
            quality_decision=quality,
            face_orientation=face_orientation,
            person_orientation=observation.person_orientation,
        )

    async def _resolve_identity(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        prepared: PreparedPerson,
        embeddings: PersonEmbeddingBundle,
        face: FaceMatchOutcome,
        frame_assignments: dict[str, list[dict[str, Any]]],
        *,
        ordinal: int | None = None,
        duplicate_with_ordinals: tuple[int, ...] = (),
    ) -> IdentityOutcome:
        observation = prepared.observation
        quality = prepared.quality_decision
        if embeddings.sbs is None:
            return IdentityOutcome(
                info=(
                    f"质量等级 {quality.person_quality_level}，"
                    f"{embeddings.sbs_failure}，未执行 ID 分配"
                ),
                decision_reason=embeddings.sbs_failure_reason,
            )
        request = IdentityRequest(
            embedding=embeddings.sbs,
            observed_face_id=face.face_id,
            person_quality_level=quality.person_quality_level,
            person_quality_score=quality.person_quality_score,
            body_completeness=quality.body_completeness,
            device_id=ctx.sn,
            timestamp=ctx.timestamp,
            keyframe_id=key_frame_id,
            picture_vector_id=picture_vector_id,
            keyframe_url=key_frame_object_name,
            crop_url=prepared.crop_path,
            person_rect=observation.people_rect,
            ai_id=observation.ai_id,
            segment_index=observation.segment_index,
            sbs_failure=embeddings.sbs_failure,
            sbs_failure_reason=embeddings.sbs_failure_reason,
            is_pano_keyframe=ctx.is_pano_keyframe,
        )

        def register(reid: str) -> None:
            frame_assignments.setdefault(reid, []).append(observation.segment)

        if getattr(self.identity_service, "remote_identity", False) is True:
            return await self.identity_service.assign_remote(
                request, ordinal=ordinal, duplicate_with_ordinals=duplicate_with_ordinals,
            )
        return await self.identity_service.assign(
            request,
            register_assignment=register,
            excluded_reids_factory=lambda: same_frame_reid_exclusions(
                frame_assignments,
                observation.segment,
            ),
        )

    async def _emit_recognition(
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
        observation = prepared.observation
        quality = prepared.quality_decision
        items = [identity.info, face.info, identity.exclusion_info]
        if identity.state:
            items.append(f"ReID状态={identity.state}")
        if embeddings.clip_failure:
            items.append(embeddings.clip_failure)
        info = "；".join(item for item in items if item)
        diagnostics = identity.diagnostics
        release_kwargs = {}
        if identity.is_new:
            release_kwargs["new_reid_recognition_release"] = (
                identity.new_reid_recognition_release
            )
        await self.recognition_emitter(
            ctx,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            prepared.crop_path,
            face.face_id,
            identity.reid,
            identity.score,
            observation.detect_time,
            observation.people_rect,
            info,
            identity.status,
            observation.ai_id,
            identity.vector_hit_reid,
            reid_state=identity.state,
            reid_enrollment_state=identity.enrollment_state,
            reid_enrollment_reason=identity.enrollment_reason,
            reid_decision=identity.decision,
            reid_decision_reason=identity.decision_reason,
            reid_search_mode=identity.search_mode,
            reid_source=identity.source,
            is_new_reid=identity.is_new,
            canonical_face_id=identity.canonical_face_id,
            reid_top1_score=diagnostics.top1_score,
            reid_mean_score=diagnostics.mean_score,
            reid_class_score=diagnostics.class_score,
            reid_support_count=diagnostics.support_count,
            reid_runner_up_score=diagnostics.runner_up_score,
            reid_margin=diagnostics.margin,
            reid_required_margin=diagnostics.required_margin,
            same_frame_excluded_reid_count=diagnostics.same_frame_excluded_reid_count,
            quality_decision=quality,
            face_orientation=face.orientation,
            person_orientation=observation.person_orientation,
            **release_kwargs,
        )
