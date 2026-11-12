"""人物 Face 的本地回退与部分复用都借用应用 CPU 资源。"""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from core.execution import AsyncWorkExecutor
from core.face_observation import FaceProcessingStage, FrameFaceObservation
from core.infer.results import FaceQualityResult, FeatureResult
from core.keyframe_processor import KeyframeProcessor
from core.person_pipeline.face_reuse import FrameFaceBinding
from utils.reid import person_face_recognition as person_face


class RecordingMetrics:
    def __init__(self):
        self.names = []

    def record(self, name, elapsed, *, failed=False):
        self.names.append(name)


class PersonFaceExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.image = np.ones((120, 120, 3), dtype=np.uint8)
        self.landmarks = np.array(
            [[35, 40], [65, 40], [50, 55], [40, 70], [60, 70]], dtype=np.float32,
        )
        self.triton = SimpleNamespace(
            extract_face_detections=AsyncMock(return_value=[{"xyxy": [20, 20, 80, 80], "confidence": 0.9}]),
            extract_face_keypoints=AsyncMock(return_value=self.landmarks),
            extract_face_quality=AsyncMock(return_value=FaceQualityResult(5.5)),
            extract_face_features=AsyncMock(return_value=FeatureResult(np.ones(512, dtype=np.float32))),
        )
        self.search = AsyncMock(return_value=("face-existing", 0.99, "face-existing"))
        for name, value in (
            ("meets_pre_extraction_quality_criteria", (True, None)),
            ("is_high_quality_feature", (True, None)),
            ("estimate_face_orientation", SimpleNamespace(orientation="front", pitch=0, yaw=0, roll=0)),
        ):
            patched = patch.object(person_face, name, return_value=value)
            patched.start()
            self.addCleanup(patched.stop)

    def features(self, executor, reuse_mode):
        face_identity = SimpleNamespace(search=self.search)
        processor = KeyframeProcessor(
            triton_client=self.triton, face_identity_service=face_identity,
            person_identity_service=SimpleNamespace(), picture_features=object(),
            records=object(), sbs_features=object(), person_clip_features=object(),
            person_face_reuse_mode=reuse_mode, image_executor=executor,
        )
        return processor.person_processor._pipeline.features

    async def match(self, executor, *, reuse_mode, binding):
        prepared = SimpleNamespace(
            crop_result=SimpleNamespace(rotation_applied=False, source_to_crop=np.eye(2, 3)),
            observation=SimpleNamespace(person_keypoints=None),
            cropped=self.image, mask_crop=np.ones((120, 120), dtype=np.uint8),
        )
        features = self.features(executor, reuse_mode)
        return await features.match_face(
            SimpleNamespace(sn="camera", timestamp=1), self.image, prepared,
            binding, enable_recognition=True,
        )

    async def test_local_face_normalization_cannot_fall_back_to_default_executor(self):
        metrics = RecordingMetrics()
        executor = AsyncWorkExecutor(1, name="image_cpu", metrics=metrics)
        self.addAsyncCleanup(executor.close)
        with patch.object(asyncio.get_running_loop(), "run_in_executor", side_effect=AssertionError("default executor used")):
            result = await self.match(executor, reuse_mode="off", binding=FrameFaceBinding())
        self.assertEqual(result.face_id, "face-existing")
        self.assertEqual(metrics.names.count("image_cpu_run"), 1)
        self.assertAlmostEqual(float(np.linalg.norm(self.search.await_args.args[0])), 1.0, places=6)

    async def test_partial_face_reuse_normalizes_new_feature_in_same_cpu_resource(self):
        metrics = RecordingMetrics()
        executor = AsyncWorkExecutor(1, name="image_cpu", metrics=metrics)
        self.addAsyncCleanup(executor.close)
        observation = FrameFaceObservation(
            source_rect=(20, 20, 80, 80), confidence=0.9,
            expanded_source_rect=(14, 14, 86, 86), source_landmarks=self.landmarks,
            orientation="front", quality_score=5.5, normalized_embedding=None,
            raw_embedding_norm=None, assigned_face_id="0",
            completed_stage=FaceProcessingStage.QUALITY,
        )
        with patch.object(asyncio.get_running_loop(), "run_in_executor", side_effect=AssertionError("default executor used")):
            result = await self.match(
                executor, reuse_mode="on",
                binding=FrameFaceBinding(frame_face=observation, reason="bound"),
            )
        self.assertEqual(result.face_id, "face-existing")
        self.assertEqual(result.source, "partial_reuse")
        self.triton.extract_face_detections.assert_not_awaited()
        self.triton.extract_face_keypoints.assert_not_awaited()
        self.assertEqual(metrics.names.count("image_cpu_run"), 1)

    async def test_standalone_helper_retains_optional_executor_compatibility(self):
        loop = asyncio.get_running_loop()
        with patch.object(loop, "run_in_executor", wraps=loop.run_in_executor) as submit:
            result = await person_face.recognize_person_face_candidate_with_info(
                self.image, (20, 20, 80, 80), "camera", 1, self.triton, self.search,
                landmarks_in_image=self.landmarks, quality_score=5.5,
            )
        self.assertEqual(result["face_id"], "face-existing")
        self.assertIsNone(submit.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
