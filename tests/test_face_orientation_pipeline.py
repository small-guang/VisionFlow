import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from core.face_keyframe_processor import FaceKeyframeProcessor
from core.face_observation import FaceProcessingStage
from core.infer.errors import InferenceUnavailable
from core.infer.results import FaceQualityResult, FeatureResult
from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.keyframe_flow_helpers import (
    normalize_numpy_vector as normalize_vector,
)
from utils.quality.face_orientation import FaceOrientation, FaceOrientationResult


class DummyLogger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class CapturingBuffer:
    def __init__(self):
        self.face_detection_records = []
        self.face_recognition_records = []
        self.face_features = []

    async def add_face_detection_record(self, record):
        self.face_detection_records.append(record)

    async def add_face_recognition_record(self, record):
        self.face_recognition_records.append(record)

    async def add_face_feature(self, **kwargs):
        self.face_features.append(kwargs)


class FaceTriton:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.keypoint_calls = 0
        self.quality_inputs = []

    async def extract_face_detections(self, _image):
        return [{"xyxy": [50, 40, 90, 80], "conf": 0.9}]

    async def extract_face_keypoints(self, _crop, _bbox):
        self.keypoint_calls += 1
        if isinstance(self.landmarks, BaseException):
            raise self.landmarks
        return self.landmarks

    async def extract_face_quality(self, crop, landmarks):
        self.quality_inputs.append(
            (np.asarray(crop).copy(), np.asarray(landmarks).copy())
        )
        return FaceQualityResult(8.0)

    async def extract_face_features(self, _crop, _landmarks):
        return FeatureResult(np.ones(512, dtype=np.float32))


class FaceOrientationPipelineTests(unittest.IsolatedAsyncioTestCase):
    def _processor(self, landmarks):
        buffer = CapturingBuffer()
        triton = FaceTriton(landmarks)

        async def search_existing(*_args, **_kwargs):
            return "known-face", 0.99, "known-face"

        return (
            FaceKeyframeProcessor(
                triton_client=triton,
                feature_buffer=buffer,
                face_identity_lock=asyncio.Lock(),
                search_faces_with_pending=search_existing,
                logger=DummyLogger(),
            ),
            buffer,
            triton,
        )

    async def _process(self, processor):
        return await processor.process_detection(
            KeyframeContext(
                sn="CAM001",
                timestamp=1710000000000,
                object_name="CAM001/picture/k.jpg",
                keyframe_id="keyframe-1",
            ),
            np.zeros((120, 120, 3), dtype=np.uint8),
            "keyframe-1",
            42,
            "CAM001/picture/k.jpg",
            object(),
        )

    async def test_valid_landmarks_propagate_one_orientation_to_detection_and_recognition(self):
        processor, buffer, triton = self._processor(
            np.array(
                [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                dtype=np.float32,
            )
        )
        orientation = FaceOrientationResult(FaceOrientation.SIDE, 0, 35, 0)
        with (
            patch(
                "core.face_keyframe_processor.upload_crop_bytes",
                new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
            ),
            patch(
                "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
                return_value=(True, ""),
            ),
            patch(
                "core.face_keyframe_processor.is_high_quality_feature",
                return_value=(True, ""),
            ),
            patch(
                "core.face_keyframe_processor.estimate_face_orientation",
                return_value=orientation,
                create=True,
            ) as estimate,
        ):
            observations = await self._process(processor)

        self.assertEqual(triton.keypoint_calls, 1)
        self.assertEqual(len(triton.quality_inputs), 1)
        quality_image, quality_landmarks = triton.quality_inputs[0]
        self.assertEqual(quality_image.shape, (80, 80, 3))
        np.testing.assert_array_equal(
            quality_landmarks,
            np.array(
                [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                dtype=np.float32,
            ),
        )
        estimate.assert_called_once()
        estimated_landmarks, estimated_shape = estimate.call_args.args
        np.testing.assert_array_equal(
            estimated_landmarks,
            np.array(
                [[50, 40], [90, 40], [70, 60], [55, 80], [85, 80]],
                dtype=np.float32,
            ),
        )
        self.assertEqual(estimated_shape, (120, 120, 3))
        self.assertEqual(len(buffer.face_detection_records), 1)
        self.assertEqual(len(buffer.face_recognition_records), 1)
        self.assertEqual(buffer.face_detection_records[0]["faceOrientation"], "side")
        self.assertEqual(buffer.face_recognition_records[0]["faceOrientation"], "side")
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source_rect, (50, 40, 90, 80))
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.IDENTITY)
        self.assertEqual(observations[0].assigned_face_id, "known-face")
        self.assertEqual(observations[0].recognition_score, 0.99)

    async def test_finite_feature_normalization_runs_in_executor(self):
        processor, _buffer, _triton = self._processor(
            np.array(
                [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                dtype=np.float32,
            )
        )
        event_loop_thread = threading.get_ident()
        normalization_thread_ids = []

        def record_normalization_thread(vector):
            normalization_thread_ids.append(threading.get_ident())
            return normalize_vector(vector)

        with (
            patch(
                "core.face_keyframe_processor.upload_crop_bytes",
                new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
            ),
            patch(
                "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
                return_value=(True, ""),
            ),
            patch(
                "core.face_keyframe_processor.normalize_numpy_vector",
                side_effect=record_normalization_thread,
            ),
        ):
            observations = await self._process(processor)

        self.assertEqual(len(observations), 1)
        self.assertEqual(len(normalization_thread_ids), 1)
        self.assertNotEqual(normalization_thread_ids[0], event_loop_thread)

    async def test_missing_keypoints_records_unknown_without_recognition_or_estimation(self):
        processor, buffer, triton = self._processor(
            InferenceUnavailable("keypoints unavailable")
        )
        with (
            patch(
                "core.face_keyframe_processor.upload_crop_bytes",
                new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
            ),
            patch(
                "core.face_keyframe_processor.estimate_face_orientation",
                create=True,
            ) as estimate,
        ):
            observations = await self._process(processor)

        self.assertEqual(triton.keypoint_calls, 1)
        estimate.assert_not_called()
        self.assertEqual(len(buffer.face_detection_records), 1)
        self.assertEqual(buffer.face_detection_records[0]["faceOrientation"], "unknown")
        self.assertEqual(buffer.face_recognition_records, [])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source_rect, (50, 40, 90, 80))
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.DETECTION)

    async def test_precheck_rejection_returns_landmarks_observation(self):
        processor, buffer, _triton = self._processor(
            np.array(
                [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                dtype=np.float32,
            )
        )
        orientation = FaceOrientationResult(FaceOrientation.FRONT, 0, 0, 0)
        with (
            patch(
                "core.face_keyframe_processor.upload_crop_bytes",
                new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
            ),
            patch(
                "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
                return_value=(False, "blur"),
            ),
            patch(
                "core.face_keyframe_processor.estimate_face_orientation",
                return_value=orientation,
            ),
        ):
            observations = await self._process(processor)

        self.assertEqual(len(buffer.face_detection_records), 1)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source_rect, (50, 40, 90, 80))
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.LANDMARKS)
        self.assertIn("预检", observations[0].failure_reason)

    async def test_invalid_bbox_is_omitted_from_observations(self):
        processor, _buffer, triton = self._processor(np.ones((5, 2), dtype=np.float32))
        triton.extract_face_detections = AsyncMock(
            return_value=[{"xyxy": [10, 10, 10, 60], "conf": 0.9}]
        )

        observations = await self._process(processor)

        self.assertEqual(observations, [])

    async def test_invalid_expanded_crop_returns_detection_observation(self):
        processor, _buffer, _triton = self._processor(np.ones((5, 2), dtype=np.float32))
        with patch(
            "core.face_keyframe_processor.expand_square_rect", return_value=None
        ):
            observations = await self._process(processor)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source_rect, (50, 40, 90, 80))
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.DETECTION)
        self.assertIn("扩展", observations[0].failure_reason)

    async def test_empty_expanded_crop_returns_detection_observation(self):
        processor, _buffer, _triton = self._processor(np.ones((5, 2), dtype=np.float32))
        with patch(
            "core.face_keyframe_processor.expand_square_rect", return_value=(0, 0, 0, 0)
        ):
            observations = await self._process(processor)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].expanded_source_rect, (0, 0, 0, 0))
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.DETECTION)
        self.assertIn("为空", observations[0].failure_reason)

    async def test_quality_inference_failure_returns_landmarks_observation(self):
        processor, _buffer, triton = self._processor(
            np.array(
                [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                dtype=np.float32,
            )
        )
        triton.extract_face_quality = AsyncMock(side_effect=RuntimeError("quality down"))
        with (
            patch(
                "core.face_keyframe_processor.upload_crop_bytes",
                new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
            ),
            patch(
                "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
                return_value=(True, ""),
            ),
        ):
            observations = await self._process(processor)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.LANDMARKS)
        self.assertEqual(observations[0].source_landmarks.shape, (5, 2))
        self.assertIn("评分失败", observations[0].failure_reason)

    async def test_missing_upload_after_accepted_quality_returns_quality_observation(self):
        processor, _buffer, _triton = self._processor(
            np.array(
                [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                dtype=np.float32,
            )
        )
        with (
            patch(
                "core.face_keyframe_processor.upload_crop_bytes",
                new=AsyncMock(return_value=""),
            ),
            patch(
                "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
                return_value=(True, ""),
            ),
        ):
            observations = await self._process(processor)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.QUALITY)
        self.assertEqual(observations[0].quality_score, 8.0)
        self.assertIn("upload", observations[0].failure_reason)

    async def test_rejected_finite_feature_returns_feature_observation(self):
        processor, _buffer, triton = self._processor(
            np.array(
                [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                dtype=np.float32,
            )
        )
        triton.extract_face_features = AsyncMock(
            return_value=FeatureResult(np.full(512, 0.01, dtype=np.float32))
        )
        with (
            patch(
                "core.face_keyframe_processor.upload_crop_bytes",
                new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
            ),
            patch(
                "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
                return_value=(True, ""),
            ),
        ):
            observations = await self._process(processor)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.FEATURE)
        self.assertAlmostEqual(observations[0].raw_embedding_norm, 0.226274, places=5)
        self.assertIsNotNone(observations[0].normalized_embedding)

    async def test_feature_inference_failure_returns_quality_observation(self):
        processor, _buffer, triton = self._processor(
            np.array(
                [[20, 20], [60, 20], [40, 40], [25, 60], [55, 60]],
                dtype=np.float32,
            )
        )
        triton.extract_face_features = AsyncMock(side_effect=RuntimeError("feature down"))
        with (
            patch(
                "core.face_keyframe_processor.upload_crop_bytes",
                new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
            ),
            patch(
                "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
                return_value=(True, ""),
            ),
        ):
            observations = await self._process(processor)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].completed_stage, FaceProcessingStage.QUALITY)
        self.assertIn("特征提取失败", observations[0].failure_reason)

    async def test_untyped_keypoint_failure_is_not_silently_downgraded(self):
        processor, _buffer, _triton = self._processor(
            RuntimeError("unexpected keypoint bug")
        )
        with patch(
            "core.face_keyframe_processor.upload_crop_bytes",
            new=AsyncMock(return_value="CAM001/face_crop/f.jpg"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "unexpected keypoint bug"
            ):
                await self._process(processor)


if __name__ == "__main__":
    unittest.main()
