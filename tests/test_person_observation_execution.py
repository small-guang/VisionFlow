"""人物观察的整帧 CPU 边界保持结果、顺序与事件循环所有权。"""

import asyncio
from copy import deepcopy
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from core.execution import AsyncWorkExecutor
from core.person_pipeline.observation import PersonObservationService
from core.person_pipeline.pipeline import PersonDetectionPipeline
from tests.person_quality_test_support import _person_test_pose
from utils.keyframe.detection_helpers import match_person_segments_to_poses
from utils.person_mask import PersonMask
from utils.quality.person.evaluator import evaluate_person_quality


class RecordingTiming:
    def __init__(self):
        self.records = []

    def record(self, operation, duration_ms, *, failed=False):
        self.records.append((operation, duration_ms, failed, threading.get_ident()))


class PersonObservationExecutionTests(unittest.IsolatedAsyncioTestCase):
    def make_inputs(self):
        image = np.arange(200 * 240 * 3, dtype=np.uint8).reshape(200, 240, 3)
        segments, poses = [], []
        # 检测顺序与位置顺序不同，Pose 再反序，避免仅验证长度而漏掉绑定变化。
        for index, x in enumerate((150, 40)):
            rect = {"x": x, "y": 40, "w": 40, "h": 100}
            pixels = np.ones((100, 40), dtype=np.uint8)
            pixels[30:35, 10:15] = 0
            if index:
                pixels[60:70, :] = 0
            mask = PersonMask(pixels, (x, 40), image.shape[:2])
            segments.append({
                "rect": rect, "conf": 0.91 - index * 0.1,
                "mask": mask, "maskPixelCount": mask.count(),
            })
            poses.append({**_person_test_pose(rect), "rect": rect})
        poses.reverse()
        client = SimpleNamespace(
            extract_person_segments=AsyncMock(return_value=segments),
            extract_person_poses=AsyncMock(return_value=poses),
        )
        return image, segments, poses, client

    def make_executor(self, *, metrics=None):
        executor = AsyncWorkExecutor(1, name="image_cpu", metrics=metrics)
        self.addAsyncCleanup(executor.close)
        return executor

    def assert_observations_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            for name in (
                "segment_index", "people_rect", "mask_pixel_count", "confidence",
                "person_orientation_result", "quality_decision",
            ):
                self.assertEqual(getattr(got, name), getattr(want, name), name)
            np.testing.assert_array_equal(got.person_keypoints, want.person_keypoints)
            self.assertEqual(got.segment.keys(), want.segment.keys())
            for name in got.segment:
                if name in ("mask", "pose"):
                    self.assertIs(got.segment[name], want.segment[name])
                else:
                    self.assertEqual(got.segment[name], want.segment[name], name)
            self.assertTrue(got.ai_id)
            self.assertGreater(got.detect_time, 0)

    async def test_binding_and_quality_run_off_loop_once_with_loop_owned_metrics(self):
        image, segments, poses, client = self.make_inputs()
        loop = asyncio.get_running_loop()
        owner = threading.get_ident()
        entered, release = asyncio.Event(), threading.Event()
        stages, log_threads = [], []
        metrics = RecordingTiming()
        executor = self.make_executor(metrics=metrics)
        logger = Mock()
        logger.debug.side_effect = lambda *args: log_threads.append(threading.get_ident())
        service = PersonObservationService(
            triton_client=client, logger=logger, image_executor=executor,
        )

        def binding(*args):
            stages.append(("binding", threading.get_ident()))
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release observation binding")
            return match_person_segments_to_poses(*args)

        def quality(*args, **kwargs):
            stages.append(("quality", threading.get_ident()))
            return evaluate_person_quality(*args, **kwargs)

        with patch("core.person_pipeline.observation.match_person_segments_to_poses", binding), patch(
            "core.person_pipeline.observation.evaluate_person_quality", quality,
        ):
            task = asyncio.create_task(service.detect(SimpleNamespace(sn="test"), image))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.wait_for(asyncio.sleep(0), 0.2)
                self.assertFalse(task.done())
                self.assertEqual([stage for stage, _ in stages], ["binding"])
                self.assertEqual(executor.queue_snapshot()["inflight_jobs"], 1)
                release.set()
                observations = await asyncio.wait_for(task, 2)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

        self.assertEqual([item.segment_index for item in observations], [0, 1])
        self.assertEqual([stage for stage, _ in stages], ["binding", "quality", "quality"])
        self.assertTrue(all(thread != owner for _, thread in stages))
        self.assertTrue(log_threads)
        self.assertTrue(all(thread == owner for thread in log_threads))
        names = [name for name, _, _, _ in metrics.records]
        self.assertCountEqual(names, [
            "image_cpu_run", "image_cpu_queue_wait",
            "person_observation_build", "person_observation_build_queue_wait",
        ])
        self.assertTrue(all(thread == owner for _, _, _, thread in metrics.records))

    async def test_threaded_result_matches_legacy_and_preserves_input_coordinates(self):
        image, segments, poses, client = self.make_inputs()
        original_image = image.copy()
        original_segments, original_poses = deepcopy(segments), deepcopy(poses)
        legacy = PersonObservationService(triton_client=client, logger=Mock())
        threaded = PersonObservationService(
            triton_client=client, logger=Mock(), image_executor=self.make_executor(),
        )
        expected = await legacy.detect(SimpleNamespace(sn="test"), image)
        actual = await threaded.detect(SimpleNamespace(sn="test"), image)
        self.assert_observations_equal(actual, expected)
        self.assertEqual([item.people_rect["x"] for item in actual], [150, 40])
        self.assertIs(actual[0].segment["pose"], poses[1])
        self.assertIs(actual[1].segment["pose"], poses[0])
        np.testing.assert_array_equal(image, original_image)
        for current, original in zip(segments, original_segments):
            self.assertEqual(current.keys(), original.keys())
            self.assertEqual(current["rect"], original["rect"])
            self.assertEqual(current["mask"].origin, original["mask"].origin)
            self.assertEqual(current["mask"].image_shape, original["mask"].image_shape)
            np.testing.assert_array_equal(current["mask"].data, original["mask"].data)
        for current, original in zip(poses, original_poses):
            self.assertEqual(current.keys(), original.keys())
            self.assertEqual(current["rect"], original["rect"])
            np.testing.assert_array_equal(current["keypoints"], original["keypoints"])

    async def test_pose_failure_and_empty_detection_keep_legacy_results(self):
        for failure in (RuntimeError("pose unavailable"), None):
            with self.subTest(failure=type(failure).__name__):
                image, segments, poses, client = self.make_inputs()
                if isinstance(failure, Exception):
                    client.extract_person_poses.side_effect = failure
                else:
                    client.extract_person_poses.return_value = failure
                legacy = PersonObservationService(triton_client=client, logger=Mock())
                threaded = PersonObservationService(
                    triton_client=client, logger=Mock(), image_executor=self.make_executor(),
                )
                expected = await legacy.detect(SimpleNamespace(sn="test"), image)
                actual = await threaded.detect(SimpleNamespace(sn="test"), image)
                self.assert_observations_equal(actual, expected)
                self.assertTrue(all(item.segment["poseInferenceFailed"] for item in actual))
                self.assertTrue(all(item.segment["pose"] is None for item in actual))
                client.extract_person_segments.return_value = []
                self.assertEqual(await threaded.detect(SimpleNamespace(sn="test"), image), [])

    async def test_run_only_executor_and_pipeline_injection(self):
        image, segments, poses, client = self.make_inputs()
        underlying = self.make_executor()
        executor = SimpleNamespace(run=AsyncMock(side_effect=underlying.run))
        pipeline = PersonDetectionPipeline(
            triton_client=client, search_faces_with_pending_locked=AsyncMock(),
            identity_service=Mock(), detection_emitter=AsyncMock(),
            recognition_emitter=AsyncMock(), logger=Mock(), runtime=SimpleNamespace(),
            upright_device_sns=frozenset(), person_face_reuse_mode="off",
            image_executor=executor,
        )
        self.assertIs(pipeline.observations.image_executor, executor)
        observations = await pipeline.observations.detect(SimpleNamespace(sn="test"), image)
        self.assertEqual(len(observations), 2)
        executor.run.assert_awaited_once()

    async def test_cancellation_keeps_frame_capacity_until_actual_work_finishes(self):
        image, segments, poses, client = self.make_inputs()
        executor = self.make_executor()
        service = PersonObservationService(
            triton_client=client, logger=Mock(), image_executor=executor,
        )
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        original = service._build_frame

        def blocking(*args):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release observation work")
            return original(*args)

        with patch.object(service, "_build_frame", blocking):
            first = asyncio.create_task(service.detect(SimpleNamespace(sn="first"), image))
            second = None
            try:
                await asyncio.wait_for(entered.wait(), 1)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertEqual(executor.queue_snapshot()["inflight_jobs"], 1)
                second = asyncio.create_task(service.detect(SimpleNamespace(sn="second"), image))

                async def queued():
                    while executor.queue_snapshot()["queued_jobs"] != 1:
                        await asyncio.sleep(0)

                await asyncio.wait_for(queued(), 1)
                self.assertFalse(second.done())
                release.set()
                self.assertEqual(len(await asyncio.wait_for(second, 2)), 2)
            finally:
                release.set()
                await asyncio.gather(*(task for task in (first, second) if task), return_exceptions=True)
        self.assertEqual(executor.queue_snapshot()["inflight_jobs"], 0)

    async def test_segmentation_and_build_errors_propagate_without_partial_results(self):
        image, segments, poses, client = self.make_inputs()
        metrics = RecordingTiming()
        executor = self.make_executor(metrics=metrics)
        service = PersonObservationService(
            triton_client=client, logger=Mock(), image_executor=executor,
        )
        failure = RuntimeError("segmentation failed")
        client.extract_person_segments.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            await service.detect(SimpleNamespace(sn="test"), image)
        self.assertIs(raised.exception, failure)
        self.assertEqual(metrics.records, [])
        client.extract_person_segments.side_effect = None
        failure = ValueError("quality failed")
        with patch("core.person_pipeline.observation.evaluate_person_quality", side_effect=failure):
            with self.assertRaises(ValueError) as raised:
                await service.detect(SimpleNamespace(sn="test"), image)
        self.assertIs(raised.exception, failure)
        runs = [record for record in metrics.records if record[0] == "person_observation_build"]
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0][2])
        self.assertEqual(executor.queue_snapshot()["inflight_jobs"], 0)


if __name__ == "__main__":
    unittest.main()
