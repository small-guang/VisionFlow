"""Person segmentation, pose inference, binding, and observation creation."""

from __future__ import annotations

import asyncio
import time
from typing import Any
import uuid

import numpy as np

from utils.keyframe.detection_helpers import (
    KeyframeContext,
    match_person_segments_to_poses,
)
from utils.keyframe.keyframe_flow_helpers import confidence_to_float
from utils.quality.person_orientation import estimate_person_orientation
from utils.quality.person.evaluator import evaluate_person_quality

from .models import PersonObservation


class PersonObservationService:
    def __init__(
        self, *, triton_client: Any, logger: Any, image_executor: Any = None,
    ) -> None:
        self.triton_client = triton_client
        self.logger = logger
        self.image_executor = image_executor

    async def detect(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
    ) -> list[PersonObservation]:
        pose_extractor = getattr(self.triton_client, "extract_person_poses", None)
        if not callable(pose_extractor):
            raise RuntimeError(
                "Person detection requires a callable Triton "
                "extract_person_poses interface"
            )
        segment_result, pose_result = await asyncio.gather(
            self.triton_client.extract_person_segments(key_frame),
            pose_extractor(key_frame),
            return_exceptions=True,
        )
        if isinstance(segment_result, BaseException):
            raise segment_result
        pose_failed = isinstance(pose_result, BaseException) or not isinstance(
            pose_result,
            list,
        )
        if pose_failed:
            self.logger.error(
                "ALog person pose inference failed or returned invalid data: "
                f"sn={ctx.sn}, result={pose_result}"
            )
            person_poses = []
        else:
            person_poses = pose_result

        args = (segment_result, person_poses, pose_failed, key_frame)
        if self.image_executor is None:
            # 独立评估器和旧调用保留原执行路径；应用装配注入有界资源。
            observations = self._build_frame(*args)
        else:
            timed = getattr(self.image_executor, "run_timed", None)
            if callable(timed):
                observations = await timed(
                    "person_observation_build", self._build_frame, *args,
                    queue_operation_name="person_observation_build_queue_wait",
                )
            else:
                observations = await self.image_executor.run(self._build_frame, *args)

        # 日志与执行器计时仍在事件循环更新；线程只处理本帧局部观察对象。
        for observation in observations:
            segment = observation.segment
            self.logger.debug(
                "Person pose binding: "
                f"segment_index={observation.segment_index}, matched={segment.get('pose') is not None}, "
                f"iou={float(segment.get('poseMatchIou') or 0.0):.4f}, "
                f"mask_agreement={float(segment.get('poseMaskAgreement') or 0.0):.4f}, "
                f"match_score={float(segment.get('poseMatchScore') or 0.0):.4f}, "
                f"sn={ctx.sn}"
            )
        getattr(self.logger, "debug", self.logger.info)(
            f"ALog person detections: segments={len(observations)}, "
            f"poses={len(person_poses)}, pose_failed={pose_failed}, sn={ctx.sn}"
        )
        return observations

    def _build_frame(
        self,
        segment_result: list[dict[str, Any]],
        person_poses: list[dict[str, Any]],
        pose_failed: bool,
        key_frame: np.ndarray,
    ) -> list[PersonObservation]:
        """每帧一次绑定并按原检测顺序构建；输入像素、mask 和 Pose 只读。"""
        segments = match_person_segments_to_poses(segment_result, person_poses)
        for segment in segments:
            segment["poseInferenceFailed"] = pose_failed
        return [
            self._build(index, segment, segments, key_frame)
            for index, segment in enumerate(segments)
        ]

    @staticmethod
    def _build(
        segment_index: int,
        segment: dict[str, Any],
        all_segments: list[dict[str, Any]],
        key_frame: np.ndarray,
    ) -> PersonObservation:
        people_rect = segment.get("rect") or {"x": 0, "y": 0, "w": 0, "h": 0}
        person_keypoints = (segment.get("pose") or {}).get("keypoints")
        return PersonObservation(
            segment_index=segment_index,
            segment=segment,
            people_rect=people_rect,
            mask_pixel_count=int(segment.get("maskPixelCount") or 0),
            confidence=confidence_to_float(segment.get("conf")),
            person_keypoints=person_keypoints,
            person_orientation_result=estimate_person_orientation(
                person_keypoints,
                people_rect,
                key_frame.shape,
            ),
            quality_decision=evaluate_person_quality(
                segment,
                all_segments,
                key_frame.shape,
                segment_index=segment_index,
            ),
            ai_id=str(uuid.uuid4()),
            detect_time=int(time.time() * 1000),
        )
