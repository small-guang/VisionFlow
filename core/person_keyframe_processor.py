from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional

import numpy as np

from core.config import settings
from core.face_observation import FrameFaceObservation
from core.identity.ports import PersonRecordSink
from core.person_pipeline import (
    PersonDetectionPipeline,
    PersonPipelineRuntime,
)
from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.person_crop import (
    PersonCropResult,
    transform_keypoints_to_person_crop,
)
from utils.keyframe.keyframe_flow_helpers import encode_image_to_bytes
from utils.quality.person.models import PersonQualityDecision
from utils.quality.person.visual import evaluate_person_visual_quality
from utils.records.recognition_record_payloads import (
    build_person_detection_record,
    build_person_recognition_record,
)
from utils.reid.person_face_recognition import match_face_id_from_person_crop_with_info
from utils.reid.person_identity import (
    REID_DECISION_CREATE_NEW,
    REID_DECISION_MATCH_EXISTING,
    REID_DECISION_SKIPPED,
    REID_DECISION_UNASSIGNED,
    PersonIdentityService,
)
from utils.reid.reid_enrollment_state import REID_ENROLLMENT_STATE_NONE
from utils.reid.reid_models import REID_SEARCH_MODE_NOT_EXECUTED
from utils.reid.reid_state import REID_STATE_UNASSIGNED
from utils.storage.base import ObjectStorageClient


def _pose_keypoints_to_crop(
    keypoints: Any,
    crop_result: PersonCropResult,
) -> Optional[np.ndarray]:
    return transform_keypoints_to_person_crop(keypoints, crop_result)


class PersonKeyframeProcessor:
    def __init__(
        self,
        *,
        triton_client: Any,
        feature_buffer: Any = None,
        identity_service: PersonIdentityService,
        search_faces_with_pending_locked: Callable[..., Any],
        logger: Any,
        runtime: PersonPipelineRuntime | None = None,
        upright_device_sns: frozenset[str] | None = None,
        person_face_reuse_mode: str | None = None,
        person_frame_concurrency: int | None = None,
        person_global_concurrency: int | None = None,
        prepare_lookahead: int = 0,
        person_face_concurrency: int | None = None,
        records: PersonRecordSink | None = None,
        sbs_features: Any = None,
        clip_features: Any = None,
        image_executor: Any,
    ):
        self.feature_buffer = feature_buffer
        self.records = records if records is not None else feature_buffer
        self.sbs_features = sbs_features if sbs_features is not None else getattr(feature_buffer, "sbs", None)
        if upright_device_sns is None:
            upright_device_sns = settings.person_upright_device_sns
        upright_device_sns = frozenset(upright_device_sns)
        if person_face_reuse_mode is None:
            person_face_reuse_mode = settings.person_face_reuse_mode
        runtime = runtime or PersonPipelineRuntime(
            encode_image=encode_image_to_bytes,
            evaluate_visual_quality=evaluate_person_visual_quality,
            match_face=match_face_id_from_person_crop_with_info,
            pose_keypoints_to_crop=_pose_keypoints_to_crop,
        )
        if person_frame_concurrency is None:
            person_frame_concurrency = settings.PERSON_FRAME_CONCURRENCY
        if person_global_concurrency is None:
            person_global_concurrency = settings.PERSON_GLOBAL_CONCURRENCY
        self._pipeline = PersonDetectionPipeline(
            triton_client=triton_client,
            feature_buffer=feature_buffer,
            search_faces_with_pending_locked=search_faces_with_pending_locked,
            identity_service=identity_service,
            detection_emitter=self.queue_person_detection_record,
            recognition_emitter=self.queue_person_recognition_record,
            logger=logger,
            runtime=runtime,
            upright_device_sns=upright_device_sns,
            person_face_reuse_mode=person_face_reuse_mode,
            person_frame_concurrency=person_frame_concurrency,
            person_global_concurrency=person_global_concurrency,
            prepare_lookahead=prepare_lookahead,
            person_face_concurrency=person_face_concurrency,
            clip_features=clip_features,
            image_executor=image_executor,
        )

    async def process_detection(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        oss_client: ObjectStorageClient,
        *,
        frame_faces: Sequence[FrameFaceObservation] = (),
    ) -> None:
        await self._pipeline.process(
            ctx,
            key_frame,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            oss_client,
            frame_faces=frame_faces,
        )

    async def queue_person_detection_record(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        crop_image_url: str,
        people_rect: Dict[str, int],
        ai_id: str,
        detect_time: int,
        status: int,
        confidence: float,
        mask_pixel_count: int,
        info: str,
        *,
        quality_decision: PersonQualityDecision,
        person_keypoint: Any = None,
        face_orientation: str = "unknown",
        person_orientation: str = "unknown",
    ):
        record = build_person_detection_record(
            keyframe_id=key_frame_id,
            picture_vector_id=picture_vector_id,
            crop_url=crop_image_url,
            dev_sn=ctx.sn,
            timestamp=ctx.timestamp,
            detect_time=detect_time,
            people_rect=people_rect,
            confidence=confidence,
            mask_pixel_count=mask_pixel_count,
            status=status,
            ai_id=ai_id,
            info=info,
            person_quality_score=quality_decision.person_quality_score,
            person_quality_level=quality_decision.person_quality_level,
            person_quality_action=quality_decision.person_quality_action,
            mask_score=quality_decision.mask_score,
            crop_score=quality_decision.crop_score,
            quality_reasons=quality_decision.quality_reasons,
            person_keypoint=person_keypoint,
            pose_score=quality_decision.pose_score,
            visible_keypoint_count=quality_decision.visible_keypoint_count,
            torso_keypoint_count=quality_decision.torso_keypoint_count,
            shoulder_hip_chain_count=quality_decision.shoulder_hip_chain_count,
            half_body_pose_eligible=quality_decision.half_body_pose_eligible,
            body_completeness=quality_decision.body_completeness,
            is_half_body=quality_decision.is_half_body,
            is_truncated=quality_decision.is_truncated,
            has_second_person=quality_decision.has_second_person,
            foreign_person_ratio=quality_decision.foreign_person_ratio,
            foreign_person_core_ratio=quality_decision.foreign_person_core_ratio,
            foreign_person_pixel_count=quality_decision.foreign_person_pixel_count,
            foreign_person_core_pixel_count=(
                quality_decision.foreign_person_core_pixel_count
            ),
            foreign_person_bbox_fallback_ratio=(
                quality_decision.foreign_person_bbox_fallback_ratio
            ),
            foreign_person_bbox_fallback_pixel_count=(
                quality_decision.foreign_person_bbox_fallback_pixel_count
            ),
            foreign_person_edge_dominant=quality_decision.foreign_person_edge_dominant,
            foreign_person_evidence_source=quality_decision.foreign_person_evidence_source,
            padding_ratio=quality_decision.padding_ratio,
            letterbox_padding_ratio=quality_decision.letterbox_padding_ratio,
            source_overflow_ratio=quality_decision.source_overflow_ratio,
            bbox_overflow_ratio=quality_decision.bbox_overflow_ratio,
            near_image_edge_count=quality_decision.near_image_edge_count,
            truncated_edge_count=quality_decision.truncated_edge_count,
            is_pano_keyframe=ctx.is_pano_keyframe,
            face_orientation=face_orientation,
            person_orientation=person_orientation,
        )
        await self.records.add_person_detection_record(record)

    async def queue_person_recognition_record(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        person_crop_path: str,
        face_id: str,
        reid: str,
        score: float,
        detect_time: int,
        people_rect: Dict[str, int],
        info: str,
        reid_status: int,
        ai_id: str,
        vector_hit_reid: str,
        *,
        quality_decision: PersonQualityDecision,
        reid_state: str = REID_STATE_UNASSIGNED,
        reid_enrollment_state: str = REID_ENROLLMENT_STATE_NONE,
        reid_enrollment_reason: str = "",
        reid_decision: str = REID_DECISION_SKIPPED,
        reid_decision_reason: str = "",
        reid_search_mode: str = REID_SEARCH_MODE_NOT_EXECUTED,
        reid_source: str = "",
        is_new_reid: bool = False,
        canonical_face_id: str = "0",
        reid_top1_score: float = 0.0,
        reid_mean_score: float = 0.0,
        reid_class_score: float = 0.0,
        reid_support_count: int = 0,
        reid_runner_up_score: float = 0.0,
        reid_margin: float = 0.0,
        reid_required_margin: float = 0.0,
        same_frame_excluded_reid_count: int = 0,
        face_orientation: str = "unknown",
        person_orientation: str = "unknown",
        new_reid_recognition_release: Any = None,
    ):
        record = build_person_recognition_record(
            dev_sn=ctx.sn,
            keyframe_id=key_frame_id,
            picture_vector_id=picture_vector_id,
            image_url=key_frame_object_name,
            crop_url=person_crop_path,
            face_id=face_id,
            reid=reid,
            score=score,
            detect_time=detect_time,
            timestamp=ctx.timestamp,
            people_rect=people_rect,
            info=info,
            reid_status=reid_status,
            reid_state=reid_state,
            reid_enrollment_state=reid_enrollment_state,
            reid_enrollment_reason=reid_enrollment_reason,
            ai_id=ai_id,
            vector_hit_reid=vector_hit_reid,
            reid_decision=reid_decision,
            reid_decision_reason=reid_decision_reason,
            reid_search_mode=reid_search_mode,
            reid_source=reid_source,
            is_new_reid=is_new_reid,
            canonical_face_id=canonical_face_id,
            reid_top1_score=reid_top1_score,
            reid_mean_score=reid_mean_score,
            reid_class_score=reid_class_score,
            reid_support_count=reid_support_count,
            reid_runner_up_score=reid_runner_up_score,
            reid_margin=reid_margin,
            reid_required_margin=reid_required_margin,
            same_frame_excluded_reid_count=same_frame_excluded_reid_count,
            person_quality_score=quality_decision.person_quality_score,
            person_quality_level=quality_decision.person_quality_level,
            person_quality_action=quality_decision.person_quality_action,
            mask_score=quality_decision.mask_score,
            crop_score=quality_decision.crop_score,
            quality_reasons=quality_decision.quality_reasons,
            pose_score=quality_decision.pose_score,
            visible_keypoint_count=quality_decision.visible_keypoint_count,
            torso_keypoint_count=quality_decision.torso_keypoint_count,
            shoulder_hip_chain_count=quality_decision.shoulder_hip_chain_count,
            half_body_pose_eligible=quality_decision.half_body_pose_eligible,
            body_completeness=quality_decision.body_completeness,
            is_half_body=quality_decision.is_half_body,
            is_truncated=quality_decision.is_truncated,
            has_second_person=quality_decision.has_second_person,
            foreign_person_ratio=quality_decision.foreign_person_ratio,
            foreign_person_core_ratio=quality_decision.foreign_person_core_ratio,
            foreign_person_pixel_count=quality_decision.foreign_person_pixel_count,
            foreign_person_core_pixel_count=(
                quality_decision.foreign_person_core_pixel_count
            ),
            foreign_person_bbox_fallback_ratio=(
                quality_decision.foreign_person_bbox_fallback_ratio
            ),
            foreign_person_bbox_fallback_pixel_count=(
                quality_decision.foreign_person_bbox_fallback_pixel_count
            ),
            foreign_person_edge_dominant=quality_decision.foreign_person_edge_dominant,
            foreign_person_evidence_source=quality_decision.foreign_person_evidence_source,
            padding_ratio=quality_decision.padding_ratio,
            letterbox_padding_ratio=quality_decision.letterbox_padding_ratio,
            source_overflow_ratio=quality_decision.source_overflow_ratio,
            bbox_overflow_ratio=quality_decision.bbox_overflow_ratio,
            near_image_edge_count=quality_decision.near_image_edge_count,
            truncated_edge_count=quality_decision.truncated_edge_count,
            is_pano_keyframe=ctx.is_pano_keyframe,
            face_orientation=face_orientation,
            person_orientation=person_orientation,
        )
        if is_new_reid:
            if new_reid_recognition_release is None:
                raise RuntimeError(
                    "New ReID recognition requires an SBS persistence release"
                )
            await self.sbs_features.attach_new_reid_recognition(
                new_reid_recognition_release,
                record,
                self.records.add_person_recognition_record,
            )
            return
        await self.records.add_person_recognition_record(record)
