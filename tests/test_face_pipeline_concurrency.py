"""Face inference may overlap; identity and record submission stay ordered."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.face_keyframe_processor import FaceKeyframeProcessor
from core.face_observation import FaceProcessingStage
from core.queue_metrics import PipelineMetrics
from tests.test_keyframe_flow import DummyLogger, FaceOnlyTriton, FakeBuffer, FakeOSS
from utils.keyframe.detection_helpers import KeyframeContext


class MultiFaceTriton(FaceOnlyTriton):
    def __init__(self, count):
        super().__init__(8.0)
        self.detections = [
            {"xyxy": [30 + 120 * i, 30, 90 + 120 * i, 90], "conf": 0.75}
            for i in range(count)
        ]

    async def extract_face_detections(self, image):
        return self.detections


class FacePipelineConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Geometry/quality rules have their own suites; these fixtures encode the
        # face index in image pixels so the remote inference double can delay it.
        quality = patch(
            "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
            return_value=(True, None),
        )
        quality.start()
        self.addCleanup(quality.stop)

    def make_flow(self, count=2, *, search=None, **limits):
        frame = np.zeros((120, 120 * max(1, count), 3), dtype=np.uint8)
        for i in range(count):
            frame[:, i * 120:(i + 1) * 120] = i + 1
        buffer = FakeBuffer()
        triton = MultiFaceTriton(count)
        lock = asyncio.Lock()

        async def match_pending(_embedding, **kwargs):
            self.assertTrue(lock.locked())
            if buffer.face_features:
                face_id = buffer.face_features[0]["face_id"]
                return face_id, 0.9, face_id
            return None, 0.0, ""

        processor = FaceKeyframeProcessor(
            triton_client=triton, feature_buffer=buffer,
            face_identity_lock=lock,
            search_faces_with_pending=search or match_pending,
            logger=DummyLogger(), **limits,
        )
        return SimpleNamespace(
            processor=processor, buffer=buffer, triton=triton,
            frame=frame, lock=lock,
        )

    async def run_flow(self, flow, frame_id="frame-1"):
        ctx = KeyframeContext(
            sn="camera", timestamp=1, object_name="camera/frame.jpg",
            keyframe_id=frame_id,
        )
        return await flow.processor.process_detection(
            ctx, flow.frame, frame_id, 1, ctx.object_name, FakeOSS(),
        )

    def start_flow(self, flow, frame_id="frame-1"):
        task = asyncio.create_task(self.run_flow(flow, frame_id))
        self.addAsyncCleanup(self.cancel_tasks, [task])
        return task

    @staticmethod
    async def cancel_tasks(tasks):
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def reached(self, event):
        try:
            await asyncio.wait_for(event.wait(), timeout=2)
        except asyncio.TimeoutError:
            self.fail("the expected concurrent face stage did not start")

    async def test_later_face_finishes_inference_first_but_identity_and_records_stay_ordered(self):
        flow = self.make_flow()
        release_first, second_finished = asyncio.Event(), asyncio.Event()
        original = flow.triton.extract_face_features

        async def delayed_features(crop, landmarks):
            if crop[0, 0, 0] == 1:
                await release_first.wait()
            result = await original(crop, landmarks)
            if crop[0, 0, 0] == 2:
                second_finished.set()
            return result

        flow.triton.extract_face_features = delayed_features
        task = self.start_flow(flow)
        await self.reached(second_finished)
        self.assertEqual(flow.buffer.face_detection_records, [])
        self.assertEqual(flow.buffer.face_features, [])
        self.assertEqual(flow.buffer.face_recognition_records, [])
        release_first.set()
        observations = await task
        self.assertEqual([o.source_rect[0] for o in observations], [30, 150])
        self.assertEqual(
            [r["face_rect"]["x"] for r in flow.buffer.face_features], [30, 150],
        )
        self.assertEqual(
            [r["faceStatus"] for r in flow.buffer.face_recognition_records], [1, 2],
        )
        self.assertEqual(observations[0].assigned_face_id, observations[1].assigned_face_id)
        self.assertEqual(
            [r["aiId"] for r in flow.buffer.face_recognition_records],
            [r["ai_id"] for r in flow.buffer.face_features],
        )
        self.assertEqual(
            [r["faceRect"]["x"] for r in flow.buffer.face_detection_records], [30, 150],
        )

    async def test_frame_window_includes_results_waiting_for_ordered_commit(self):
        flow = self.make_flow(5, face_frame_concurrency=2, face_global_concurrency=4)
        release_first, second_finished = asyncio.Event(), asyncio.Event()
        started = []
        original = flow.triton.extract_face_features

        async def delayed_features(crop, landmarks):
            index = int(crop[0, 0, 0])
            started.append(index)
            if index == 1:
                await release_first.wait()
            result = await original(crop, landmarks)
            if index == 2:
                second_finished.set()
            return result

        flow.triton.extract_face_features = delayed_features
        task = self.start_flow(flow)
        await self.reached(second_finished)
        await asyncio.sleep(0)
        self.assertCountEqual(started, [1, 2])
        release_first.set()
        await task
        self.assertCountEqual(started, [1, 2, 3, 4, 5])

    async def test_global_limit_is_shared_across_frames_and_covers_feature_inference(self):
        flow = self.make_flow(3, face_frame_concurrency=3, face_global_concurrency=2)
        full, release = asyncio.Event(), asyncio.Event()
        active, peak = 0, 0
        original = flow.triton.extract_face_features

        async def blocked_features(*args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                full.set()
            try:
                await release.wait()
                return await original(*args)
            finally:
                active -= 1

        flow.triton.extract_face_features = blocked_features
        first = self.start_flow(flow)
        second = self.start_flow(flow, "frame-2")
        await self.reached(full)
        await asyncio.sleep(0)
        self.assertEqual(peak, 2)
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(peak, 2)
        self.assertEqual(len(flow.buffer.face_recognition_records), 6)
        self.assertEqual(
            sum(r["faceStatus"] == 1 for r in flow.buffer.face_recognition_records), 1,
        )

    async def test_identity_lock_covers_pending_insert_across_frames(self):
        flow = self.make_flow(1, face_frame_concurrency=1, face_global_concurrency=1)
        inserting, release_insert, second_prepared = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(),
        )
        add_feature = flow.buffer.add_face_feature
        original = flow.triton.extract_face_features
        search = flow.processor.search_faces_with_pending
        search_calls = 0

        async def observe_search(*args, **kwargs):
            nonlocal search_calls
            search_calls += 1
            return await search(*args, **kwargs)

        async def delayed_insert(**kwargs):
            inserting.set()
            await release_insert.wait()
            self.assertTrue(flow.lock.locked())
            await add_feature(**kwargs)

        async def observe_features(*args):
            result = await original(*args)
            if flow.triton.feature_calls == 2:
                second_prepared.set()
            return result

        flow.processor.search_faces_with_pending = observe_search
        flow.buffer.add_face_feature = delayed_insert
        flow.triton.extract_face_features = observe_features
        first = self.start_flow(flow)
        await self.reached(inserting)
        second = self.start_flow(flow, "frame-2")
        await self.reached(second_prepared)
        await asyncio.sleep(0)
        self.assertEqual(search_calls, 1)
        release_insert.set()
        await asyncio.gather(first, second)
        self.assertEqual(search_calls, 2)
        self.assertEqual(
            [r["faceStatus"] for r in flow.buffer.face_recognition_records], [1, 2],
        )

    async def test_first_face_commits_before_slow_later_face_finishes(self):
        flow = self.make_flow()
        second_started, release_second, first_committed = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(),
        )
        original = flow.triton.extract_face_features
        add_feature = flow.buffer.add_face_feature

        async def delayed_features(crop, landmarks):
            if crop[0, 0, 0] == 2:
                second_started.set()
                await release_second.wait()
            return await original(crop, landmarks)

        async def observe_commit(**kwargs):
            await add_feature(**kwargs)
            first_committed.set()

        flow.triton.extract_face_features = delayed_features
        flow.buffer.add_face_feature = observe_commit
        task = self.start_flow(flow)
        await self.reached(second_started)
        await self.reached(first_committed)
        self.assertFalse(task.done())
        self.assertEqual(len(flow.buffer.face_recognition_records), 1)
        release_second.set()
        await task

    async def test_identity_wait_releases_preparation_slot(self):
        second_finished = asyncio.Event()

        async def waiting_search(*args, **kwargs):
            await second_finished.wait()
            return "existing-face", 0.9, "existing-face"

        flow = self.make_flow(
            search=waiting_search, face_frame_concurrency=2, face_global_concurrency=1,
        )
        original = flow.triton.extract_face_features

        async def observe_features(crop, landmarks):
            result = await original(crop, landmarks)
            if crop[0, 0, 0] == 2:
                second_finished.set()
            return result

        flow.triton.extract_face_features = observe_features
        await asyncio.wait_for(self.run_flow(flow), timeout=2)
        self.assertEqual(len(flow.buffer.face_recognition_records), 2)

    async def test_cancellation_drains_active_and_waiting_faces_and_releases_slots(self):
        flow = self.make_flow(4, face_frame_concurrency=4, face_global_concurrency=2)
        full, release = asyncio.Event(), asyncio.Event()
        active = set()
        original = flow.triton.extract_face_keypoints

        async def blocked_keypoints(crop, bbox):
            index = int(crop[0, 0, 0])
            active.add(index)
            if len(active) == 2:
                full.set()
            try:
                await release.wait()
                return await original(crop, bbox)
            finally:
                active.remove(index)

        flow.triton.extract_face_keypoints = blocked_keypoints
        task = self.start_flow(flow)
        await self.reached(full)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(active, set())
        self.assertEqual(flow.buffer.face_detection_records, [])
        release.set()
        observations = await asyncio.wait_for(self.run_flow(flow, "next-frame"), timeout=2)
        self.assertEqual(len(observations), 4)

    async def test_preparation_failure_cancels_later_faces_before_returning(self):
        flow = self.make_flow()
        second_started, second_stopped = asyncio.Event(), asyncio.Event()

        async def failed_keypoints(crop, bbox):
            if crop[0, 0, 0] == 1:
                await second_started.wait()
                raise RuntimeError("preparation unavailable")
            second_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                second_stopped.set()

        flow.triton.extract_face_keypoints = failed_keypoints
        task = self.start_flow(flow)
        with self.assertRaisesRegex(RuntimeError, "preparation unavailable"):
            await asyncio.wait_for(task, timeout=2)
        self.assertTrue(second_stopped.is_set())
        self.assertEqual(flow.buffer.face_features, [])

    async def test_commit_failure_cancels_later_faces_and_releases_identity_lock(self):
        flow = self.make_flow()
        second_started, second_stopped = asyncio.Event(), asyncio.Event()
        original = flow.triton.extract_face_features

        async def blocked_features(crop, landmarks):
            if crop[0, 0, 0] == 2:
                second_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    second_stopped.set()
            return await original(crop, landmarks)

        async def failed_commit(**kwargs):
            await second_started.wait()
            raise RuntimeError("buffer unavailable")

        flow.triton.extract_face_features = blocked_features
        flow.buffer.add_face_feature = failed_commit
        task = self.start_flow(flow)
        with self.assertRaisesRegex(RuntimeError, "buffer unavailable"):
            await asyncio.wait_for(task, timeout=2)
        self.assertTrue(second_stopped.is_set())
        self.assertFalse(flow.lock.locked())
        self.assertEqual(flow.buffer.face_features, [])

    async def test_rejected_face_does_not_prevent_later_face_from_committing(self):
        flow = self.make_flow()
        original = flow.triton.extract_face_keypoints

        async def invalid_first(crop, bbox):
            if crop[0, 0, 0] == 1:
                return None
            return await original(crop, bbox)

        flow.triton.extract_face_keypoints = invalid_first
        observations = await self.run_flow(flow)
        self.assertEqual([o.source_rect[0] for o in observations], [30, 150])
        self.assertEqual(
            [o.completed_stage for o in observations],
            [FaceProcessingStage.DETECTION, FaceProcessingStage.IDENTITY],
        )
        self.assertEqual([r["status"] for r in flow.buffer.face_detection_records], [0, 1])
        self.assertEqual(len(flow.buffer.face_features), 1)

    async def test_configured_one_slot_restores_serial_processing(self):
        with patch(
            "core.face_keyframe_processor.settings",
            SimpleNamespace(FACE_FRAME_CONCURRENCY=1, FACE_GLOBAL_CONCURRENCY=1),
        ):
            flow = self.make_flow()
        committed_counts = []
        original = flow.triton.extract_face_keypoints

        async def observe_preparation(*args):
            committed_counts.append(len(flow.buffer.face_recognition_records))
            return await original(*args)

        flow.triton.extract_face_keypoints = observe_preparation
        await self.run_flow(flow)
        self.assertEqual(committed_counts, [0, 1])

    async def test_missing_configured_global_limit_fails_fast(self):
        with patch(
            "core.face_keyframe_processor.settings",
            SimpleNamespace(FACE_FRAME_CONCURRENCY=1),
        ):
            with self.assertRaisesRegex(AttributeError, "FACE_GLOBAL_CONCURRENCY"):
                self.make_flow()

    async def test_empty_frame_produces_no_face_work(self):
        flow = self.make_flow(0)
        self.assertEqual(await self.run_flow(flow), [])
        self.assertEqual(flow.buffer.face_detection_records, [])
        self.assertEqual(flow.triton.feature_calls, 0)

    async def test_explicit_limits_reject_invalid_values(self):
        for name in ("face_frame_concurrency", "face_global_concurrency"):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.make_flow(**{name: value})

    async def test_ready_metrics_exclude_running_inference_and_drain_on_every_exit(self):
        for outcome in ("success", "cancel", "commit_failure"):
            with self.subTest(outcome=outcome):
                flow = self.make_flow(2, face_global_concurrency=2)
                metrics = PipelineMetrics()
                release_first, second_ready = asyncio.Event(), asyncio.Event()
                committed = []

                async def prepare(ctx, frame, oss, detection):
                    index = 0 if detection["xyxy"][0] == 30 else 1
                    if index == 0:
                        await release_first.wait()
                    else:
                        second_ready.set()
                    return SimpleNamespace(index=index)

                async def commit(ctx, frame_id, picture_id, object_name, prepared):
                    if outcome == "commit_failure":
                        raise RuntimeError("commit unavailable")
                    committed.append(prepared.index)

                flow.processor._prepare_face = prepare
                flow.processor._commit_face = commit
                with patch("core.face_keyframe_processor.pipeline_metrics", metrics):
                    task = self.start_flow(flow)
                    await self.reached(second_ready)
                    stages = metrics.snapshot()["stages"]
                    self.assertEqual(stages["face_prepare"]["active"], 1)
                    self.assertEqual(stages["face_ordered_commit"]["waiting"], 1)
                    if outcome == "cancel":
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        release_first.set()
                        if outcome == "commit_failure":
                            with self.assertRaisesRegex(RuntimeError, "commit unavailable"):
                                await task
                        else:
                            await task
                            self.assertEqual(committed, [0, 1])
                    stages = metrics.snapshot()["stages"]
                    for stage in stages.values():
                        self.assertEqual(stage["waiting"], 0)
                        self.assertEqual(stage["active"], 0)


if __name__ == "__main__":
    unittest.main()
