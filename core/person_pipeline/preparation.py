"""独立的人型视觉准备与裁图持久化阶段。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.person_crop import build_person_crop
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_DROP, upload_crop_bytes
from utils.quality.person.decision import (
    drop_person_quality_decision,
    merge_person_crop_quality,
    merge_person_visual_quality,
)
from utils.quality.person.crop import evaluate_person_crop_quality
from utils.quality.person.visual import (
    format_person_visual_metrics,
    format_person_visual_thresholds,
)
from utils.quality.quality_reasons import QualityReason

from .models import PersonObservation, PersonPipelineRuntime, PreparedPerson


class PersonPreparationService:
    def __init__(
        self,
        *,
        logger: Any,
        runtime: PersonPipelineRuntime,
        upright_device_sns: frozenset[str],
        image_executor: Any,
    ) -> None:
        if image_executor is None:
            raise ValueError("image_executor is required")
        self.logger = logger
        self.runtime = runtime
        self.upright_device_sns = frozenset(upright_device_sns)
        self.image_executor = image_executor

    async def prepare(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        oss_client: Any,
        observation: PersonObservation,
        all_segments: list[dict[str, Any]],
    ) -> PreparedPerson | None:
        """为独立评估器保留完成上传的顺序准备入口。"""
        prepared = await self.prepare_visual(ctx, key_frame, observation, all_segments)
        if prepared is None:
            return None
        return await self.upload_prepared(ctx, oss_client, prepared)

    async def prepare_visual(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        observation: PersonObservation,
        all_segments: list[dict[str, Any]],
    ) -> PreparedPerson | None:
        """只产出裁图、质量和编码；不依赖存储完成或身份查询。"""
        timed = getattr(self.image_executor, "run_timed", None)
        if callable(timed):
            prepared = await timed(
                "person_crop_prepare", self._prepare_crop,
                ctx, key_frame, observation, all_segments,
                queue_operation_name="person_crop_prepare_queue_wait",
            )
        else:
            prepared = await self.image_executor.run(
                self._prepare_crop, ctx, key_frame, observation, all_segments,
            )
        if prepared is None:
            return None
        prepared = await self._apply_visual_quality(ctx, prepared)
        image_bytes = await self._run_image_operation(
            self.runtime.encode_image,
            prepared.cropped,
        )
        if not image_bytes:
            self.logger.error("Failed to encode person crop")
        return replace(prepared, image_bytes=image_bytes)

    async def _run_image_operation(self, operation, *args):
        return await self.image_executor.run(operation, *args)

    @staticmethod
    def drop_empty_crop(observation: PersonObservation) -> PersonObservation:
        return replace(
            observation,
            quality_decision=drop_person_quality_decision(
                QualityReason.EMPTY_CROP,
                mask_pixel_count=observation.mask_pixel_count,
                base_decision=observation.quality_decision,
            ),
        )

    def _prepare_crop(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        observation: PersonObservation,
        all_segments: list[dict[str, Any]],
    ) -> PreparedPerson | None:
        rotation_enabled = (
            ctx.is_pano_keyframe
            or "EPS" in ctx.sn.upper()
            or ctx.sn in self.upright_device_sns
        )
        crop_result = build_person_crop(
            key_frame,
            observation.people_rect,
            observation.segment.get("mask"),
            keypoints=observation.person_keypoints,
            rotation_enabled=rotation_enabled,
        )
        if crop_result is None:
            return None
        crop_info = (
            f"，旋转角度 {crop_result.rotation_angle:.2f}°，"
            f"旋转状态 {crop_result.rotation_reason}"
        )
        if crop_result.rotation_reason == "rotation_failed":
            self.logger.warning(
                "Person upright rotation failed; used unrotated natural crop: "
                f"sn={ctx.sn}"
            )
        crop_quality = evaluate_person_crop_quality(
            crop_result,
            observation.segment,
            all_segments,
            key_frame.shape,
            segment_index=observation.segment_index,
        )
        quality = merge_person_crop_quality(
            observation.quality_decision,
            crop_quality,
        )
        metrics = crop_quality.metrics
        if crop_quality.quality_reasons:
            crop_info += (
                f"，总补边占比 {metrics.padding_ratio:.3f}，"
                f"letterbox占比 {metrics.letterbox_padding_ratio:.3f}，"
                f"自然窗口越界占比 {metrics.source_overflow_ratio:.3f}，"
                f"外来人物/目标mask占比 {metrics.foreign_person_ratio:.3f}，"
                f"核心区占比 {metrics.foreign_person_core_ratio:.3f}，"
                f"bbox回退占比 {metrics.foreign_person_bbox_fallback_ratio:.3f}，"
                f"证据来源 {metrics.foreign_person_evidence_source}"
            )
        self._log_crop_geometry(ctx, observation, crop_result, quality, metrics)
        return PreparedPerson(observation, crop_result, quality, crop_info)

    def _log_crop_geometry(
        self,
        ctx: KeyframeContext,
        observation: PersonObservation,
        crop_result: Any,
        quality: Any,
        metrics: Any,
    ) -> None:
        self.logger.debug(
            "Person crop geometry: "
            f"original_rect={observation.people_rect}, "
            f"requested_xyxy={crop_result.requested_xyxy}, "
            f"source_xyxy={crop_result.source_xyxy}, "
            f"content_xyxy={crop_result.content_xyxy}, "
            f"padding={crop_result.padding}, "
            f"rotation_angle={crop_result.rotation_angle:.2f}, "
            f"rotation_applied={crop_result.rotation_applied}, "
            f"rotation_reason={crop_result.rotation_reason}, "
            f"padding_ratio={metrics.padding_ratio:.4f}, "
            f"letterbox_padding_ratio={metrics.letterbox_padding_ratio:.4f}, "
            f"source_overflow_ratio={metrics.source_overflow_ratio:.4f}, "
            f"bbox_overflow_ratio={quality.bbox_overflow_ratio:.4f}, "
            f"near_image_edge_count={quality.near_image_edge_count}, "
            f"truncated_edge_count={quality.truncated_edge_count}, "
            f"target_mask_pixels={metrics.target_mask_pixel_count}, "
            f"foreign_person_pixels={metrics.foreign_person_pixel_count}, "
            f"foreign_person_ratio={metrics.foreign_person_ratio:.4f}, "
            f"foreign_person_core_pixels={metrics.foreign_person_core_pixel_count}, "
            f"foreign_person_core_ratio={metrics.foreign_person_core_ratio:.4f}, "
            f"foreign_person_bbox_fallback_pixels={metrics.foreign_person_bbox_fallback_pixel_count}, "
            f"foreign_person_bbox_fallback_ratio={metrics.foreign_person_bbox_fallback_ratio:.4f}, "
            f"foreign_person_edge_dominant={metrics.foreign_person_edge_dominant}, "
            f"foreign_person_evidence_source={metrics.foreign_person_evidence_source}, "
            f"crop_shape={crop_result.image.shape}, sn={ctx.sn}"
        )

    async def _apply_visual_quality(
        self,
        ctx: KeyframeContext,
        prepared: PreparedPerson,
    ) -> PreparedPerson:
        if prepared.quality_decision.person_quality_level == QUALITY_LEVEL_DROP:
            return prepared
        try:
            visual = await self._run_image_operation(
                self.runtime.evaluate_visual_quality,
                prepared.cropped,
                prepared.mask_crop,
            )
            quality = merge_person_visual_quality(prepared.quality_decision, visual)
            metrics_text = format_person_visual_metrics(visual.metrics)
            self.logger.debug(
                "Person visual precheck: "
                f"{metrics_text}, "
                f"reasons={[reason.value for reason in visual.quality_reasons]}, "
                f"sn={ctx.sn}"
            )
            info = ""
            if visual.quality_reasons:
                info = (
                    f"，图像预检指标：{metrics_text}；"
                    f"{format_person_visual_thresholds(visual.quality_reasons)}"
                )
            return replace(prepared, quality_decision=quality, visual_info=info)
        except Exception as error:
            self.logger.exception(
                f"Person visual precheck failed, keeping structural quality: {error}"
            )
            return prepared

    async def upload_prepared(
        self,
        ctx: KeyframeContext,
        oss_client: Any,
        prepared: PreparedPerson,
    ) -> PreparedPerson:
        """上传已编码裁图；调用方拥有任务寿命和最终结果的汇合。"""
        image_bytes = prepared.image_bytes
        crop_path = ""
        if image_bytes:
            crop_path = await upload_crop_bytes(
                oss_client,
                image_bytes,
                ctx.sn,
                "person_crop",
            )
        if not crop_path:
            self.logger.warning(f"Person crop upload failed: sn={ctx.sn}")
        return replace(prepared, crop_path=crop_path)
