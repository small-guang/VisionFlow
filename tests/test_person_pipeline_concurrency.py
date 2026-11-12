"""Person work may finish out of order; identity decisions must not."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.person_keyframe_processor import PersonKeyframeProcessor
from core.execution import AsyncWorkExecutor
from core.person_pipeline.models import PersonPipelineRuntime
from core.queue_metrics import PipelineMetrics
from tests.person_pose_test_support import pose_result, segment_result, textured_image
from tests.test_person_pose_integration import (
    DummyLogger,
    FakeBuffer,
    FakeOSS,
    FakeTriton,
    no_face_match,
    observed_face_match,
    person_result,
)
from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_DROP, encode_image_to_bytes
from utils.keyframe.person_crop import transform_keypoints_to_person_crop
from utils.quality.person.visual import evaluate_person_visual_quality
from utils.reid.person_identity import PersonIdentityService


class PersonPipelineConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def make_flow(self, count=2, *, search=None, **limits):
        frame = np.tile(textured_image(), (1, max(1, count), 1))
        segments, poses = [], []
        for index in range(count):
            rect = {"x": 40 + 120 * index, "y": 20, "w": 40, "h": 120}
            segment = segment_result(rect)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[20:140, rect["x"]:rect["x"] + 40] = 1
            segment["mask"] = mask
            segments.append(segment)
            poses.append(pose_result(rect))
        triton = FakeTriton(segments=segments, poses=poses)
        buffer = FakeBuffer()

        async def choose_identity(_embedding, **kwargs):
            excluded = kwargs.get("excluded_reids", ())
            for candidate in ("first-id", "second-id", "third-id"):
                if candidate not in excluded:
                    return person_result(candidate, 0.95, candidate, "milvus")
            return person_result()

        executor = AsyncWorkExecutor(2, name="image_cpu")
        self.addAsyncCleanup(executor.close)
        processor = PersonKeyframeProcessor(
            image_executor=executor,
            triton_client=triton,
            feature_buffer=buffer,
            identity_service=PersonIdentityService(
                identity_lock=asyncio.Lock(),
                search_persons=search or choose_identity,
                sbs_buffer=buffer.sbs,
                logger=DummyLogger(),
            ),
            search_faces_with_pending_locked=no_face_match,
            logger=DummyLogger(),
            upright_device_sns=frozenset(),
            person_face_reuse_mode="off",
            runtime=PersonPipelineRuntime(
                encode_image=encode_image_to_bytes,
                evaluate_visual_quality=evaluate_person_visual_quality,
                match_face=observed_face_match,
                pose_keypoints_to_crop=transform_keypoints_to_person_crop,
            ),
            **limits,
        )
        return SimpleNamespace(
            processor=processor, pipeline=processor._pipeline,
            buffer=buffer, triton=triton, frame=frame,
        )

    async def run_flow(self, flow, frame_id="frame-1"):
        ctx = KeyframeContext(
            sn="camera", timestamp=1, object_name="camera/frame.jpg",
            keyframe_id=frame_id,
        )
        await flow.processor.process_detection(
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
            self.fail("the expected concurrent pipeline stage did not start")

    async def test_later_person_finishes_inference_before_first_but_ids_commit_in_order(self):
        flow = self.make_flow()
        first_started, release_first, second_finished = (
            asyncio.Event(), asyncio.Event(), asyncio.Event()
        )
        original = flow.pipeline.features.extract_embeddings

        async def delayed_inference(prepared):
            if prepared.observation.segment_index == 0:
                first_started.set()
                await release_first.wait()
            result = await original(prepared)
            if prepared.observation.segment_index == 1:
                second_finished.set()
            return result

        flow.pipeline.features.extract_embeddings = delayed_inference
        task = self.start_flow(flow)
        await self.reached(first_started)
        await self.reached(second_finished)
        self.assertEqual(flow.buffer.recognitions, [])
        self.assertEqual(flow.buffer.features, [])
        release_first.set()
        await task
        self.assertEqual(
            [record["reid"] for record in flow.buffer.recognitions],
            ["first-id", "second-id"],
        )
        self.assertEqual(
            [record["sameFrameExcludedReidCount"] for record in flow.buffer.recognitions],
            [0, 1],
        )
        self.assertEqual(
            [feature["person_rect"]["x"] for feature in flow.buffer.features],
            [40, 160],
        )

    async def test_window_bounds_prepared_results_while_first_person_is_slow(self):
        flow = self.make_flow(6, person_frame_concurrency=2, person_global_concurrency=4)
        release_first, second_finished = asyncio.Event(), asyncio.Event()
        prepared_indices = []
        original = flow.pipeline.preparation.prepare_visual

        async def delayed_prepare(ctx, frame, observation, segments):
            prepared_indices.append(observation.segment_index)
            if observation.segment_index == 0:
                await release_first.wait()
            result = await original(ctx, frame, observation, segments)
            if observation.segment_index == 1:
                second_finished.set()
            return result

        flow.pipeline.preparation.prepare_visual = delayed_prepare
        task = self.start_flow(flow)
        await self.reached(second_finished)
        await asyncio.sleep(0)
        self.assertEqual(prepared_indices, [0, 1])
        release_first.set()
        await task
        self.assertEqual(prepared_indices, list(range(6)))
        self.assertEqual(len(flow.buffer.detections), 6)

    async def test_global_limit_is_shared_by_frames_and_covers_inference(self):
        flow = self.make_flow(4, person_frame_concurrency=4, person_global_concurrency=2)
        full, release = asyncio.Event(), asyncio.Event()
        active, peak = 0, 0
        original = flow.pipeline.features.extract_embeddings

        async def blocked_inference(prepared):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                full.set()
            try:
                await release.wait()
                return await original(prepared)
            finally:
                active -= 1

        flow.pipeline.features.extract_embeddings = blocked_inference
        first = self.start_flow(flow, "frame-1")
        second = self.start_flow(flow, "frame-2")
        await self.reached(full)
        await asyncio.sleep(0)
        self.assertEqual(peak, 2)
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(peak, 2)
        self.assertEqual(len(flow.buffer.recognitions), 8)

    async def test_first_person_commits_without_waiting_for_the_whole_window(self):
        flow = self.make_flow()
        second_started, release_second, first_committed = (
            asyncio.Event(), asyncio.Event(), asyncio.Event()
        )
        original = flow.pipeline.features.extract_embeddings
        emit = flow.pipeline.recognition_emitter

        async def delayed_inference(prepared):
            if prepared.observation.segment_index == 1:
                second_started.set()
                await release_second.wait()
            return await original(prepared)

        async def record_commit(*args, **kwargs):
            await emit(*args, **kwargs)
            first_committed.set()

        flow.pipeline.features.extract_embeddings = delayed_inference
        flow.pipeline.recognition_emitter = record_commit
        task = self.start_flow(flow)
        await self.reached(second_started)
        await self.reached(first_committed)
        self.assertFalse(task.done())
        self.assertEqual(len(flow.buffer.recognitions), 1)
        release_second.set()
        await task

    async def test_cancellation_drains_active_work_and_releases_global_slots(self):
        flow = self.make_flow(4, person_frame_concurrency=4, person_global_concurrency=2)
        full, release = asyncio.Event(), asyncio.Event()
        active = set()
        original = flow.pipeline.preparation.prepare_visual

        async def blocked_prepare(ctx, frame, observation, segments):
            index = observation.segment_index
            active.add(index)
            if len(active) == 2:
                full.set()
            try:
                await release.wait()
                return await original(ctx, frame, observation, segments)
            finally:
                active.remove(index)

        flow.pipeline.preparation.prepare_visual = blocked_prepare
        task = self.start_flow(flow)
        await self.reached(full)
        self.assertEqual(len(active), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(active, set())
        self.assertEqual(flow.buffer.recognitions, [])
        release.set()
        await asyncio.wait_for(self.run_flow(flow, "next-frame"), timeout=2)
        self.assertEqual(len(flow.buffer.detections), 4)

    async def test_identity_failure_cancels_later_work_before_returning(self):
        second_started, second_stopped = asyncio.Event(), asyncio.Event()

        async def failed_search(*args, **kwargs):
            await second_started.wait()
            raise RuntimeError("identity unavailable")

        flow = self.make_flow(search=failed_search)
        original = flow.pipeline.features.extract_embeddings

        async def blocked_inference(prepared):
            if prepared.observation.segment_index == 1:
                second_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    second_stopped.set()
            return await original(prepared)

        flow.pipeline.features.extract_embeddings = blocked_inference
        task = self.start_flow(flow)
        await self.reached(second_started)
        with self.assertRaisesRegex(RuntimeError, "identity unavailable"):
            await task
        self.assertTrue(second_stopped.is_set())
        self.assertEqual(flow.buffer.recognitions, [])

    async def test_identity_wait_does_not_hold_the_only_preparation_slot(self):
        second_finished = asyncio.Event()

        async def waiting_search(*args, **kwargs):
            await second_finished.wait()
            return person_result()

        flow = self.make_flow(
            search=waiting_search, person_frame_concurrency=2,
            person_global_concurrency=1,
        )
        original = flow.pipeline.features.extract_embeddings

        async def observe_inference(prepared):
            result = await original(prepared)
            if prepared.observation.segment_index == 1:
                second_finished.set()
            return result

        flow.pipeline.features.extract_embeddings = observe_inference
        await asyncio.wait_for(self.run_flow(flow), timeout=2)
        self.assertEqual(len(flow.buffer.recognitions), 2)

    async def test_preparation_failure_drains_later_work_without_identity_submission(self):
        flow = self.make_flow()
        second_started, second_stopped = asyncio.Event(), asyncio.Event()

        async def failed_prepare(ctx, frame, observation, segments):
            if observation.segment_index == 0:
                await second_started.wait()
                raise RuntimeError("preparation unavailable")
            second_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                second_stopped.set()

        flow.pipeline.preparation.prepare_visual = failed_prepare
        task = self.start_flow(flow)
        with self.assertRaisesRegex(RuntimeError, "preparation unavailable"):
            await asyncio.wait_for(task, timeout=2)
        self.assertTrue(second_stopped.is_set())
        self.assertEqual(flow.buffer.recognitions, [])
        self.assertEqual(flow.buffer.features, [])

    async def test_configured_single_slot_reaches_the_running_person_pipeline(self):
        with patch(
            "core.person_keyframe_processor.settings",
            SimpleNamespace(PERSON_FRAME_CONCURRENCY=1, PERSON_GLOBAL_CONCURRENCY=1),
        ):
            flow = self.make_flow()
        committed_counts = []
        original = flow.pipeline.preparation.prepare_visual

        async def observe_prepare(*args):
            committed_counts.append(len(flow.buffer.recognitions))
            return await original(*args)

        flow.pipeline.preparation.prepare_visual = observe_prepare
        await self.run_flow(flow)
        self.assertEqual(committed_counts, [0, 1])

    async def test_missing_configured_global_limit_fails_fast(self):
        with patch(
            "core.person_keyframe_processor.settings",
            SimpleNamespace(PERSON_FRAME_CONCURRENCY=1),
        ):
            with self.assertRaisesRegex(AttributeError, "PERSON_GLOBAL_CONCURRENCY"):
                self.make_flow()

    async def test_one_slot_preserves_serial_processing(self):
        flow = self.make_flow(person_frame_concurrency=1, person_global_concurrency=2)
        committed_counts = []
        original = flow.pipeline.preparation.prepare_visual

        async def observe_prepare(*args):
            committed_counts.append(len(flow.buffer.recognitions))
            return await original(*args)

        flow.pipeline.preparation.prepare_visual = observe_prepare
        await self.run_flow(flow)
        self.assertEqual(committed_counts, [0, 1])

    async def test_empty_frame_produces_no_person_work(self):
        flow = self.make_flow(0)
        await self.run_flow(flow)
        self.assertEqual(flow.buffer.detections, [])
        self.assertEqual(flow.triton.sbs_calls, 0)

    async def test_embeddings_finish_while_face_match_waits_without_early_commit(self):
        flow = self.make_flow(1)
        face_started, release_face, embeddings_finished = (
            asyncio.Event(), asyncio.Event(), asyncio.Event()
        )
        match_face = flow.pipeline.features.match_face
        extract_embeddings = flow.pipeline.features.extract_embeddings

        async def blocked_face(*args, **kwargs):
            face_started.set()
            await release_face.wait()
            return await match_face(*args, **kwargs)

        async def observe_embeddings(prepared):
            await face_started.wait()
            result = await extract_embeddings(prepared)
            embeddings_finished.set()
            return result

        flow.pipeline.features.match_face = blocked_face
        flow.pipeline.features.extract_embeddings = observe_embeddings
        task = self.start_flow(flow)
        await self.reached(embeddings_finished)
        self.assertFalse(task.done())
        self.assertEqual(flow.buffer.recognitions, [])
        self.assertEqual(flow.buffer.features, [])
        release_face.set()
        await task
        self.assertEqual(len(flow.buffer.recognitions), 1)

    async def test_branch_failure_cancels_and_drains_the_other_branch(self):
        for failing_branch in ("match_face", "extract_embeddings"):
            with self.subTest(failing_branch=failing_branch):
                flow = self.make_flow(1)
                sibling_started, sibling_stopped = asyncio.Event(), asyncio.Event()

                async def fail(*args, **kwargs):
                    await sibling_started.wait()
                    raise RuntimeError("inference branch unavailable")

                async def wait_forever(*args, **kwargs):
                    sibling_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        sibling_stopped.set()

                sibling = (
                    "extract_embeddings" if failing_branch == "match_face"
                    else "match_face"
                )
                setattr(flow.pipeline.features, failing_branch, fail)
                setattr(flow.pipeline.features, sibling, wait_forever)
                task = self.start_flow(flow)
                with self.assertRaisesRegex(RuntimeError, "inference branch unavailable"):
                    await asyncio.wait_for(task, timeout=2)
                self.assertTrue(sibling_stopped.is_set())
                self.assertEqual(flow.buffer.detections, [])
                self.assertEqual(flow.buffer.recognitions, [])
                self.assertFalse(flow.pipeline._preparation_slots.locked())

    async def test_cancellation_drains_both_branches_and_releases_preparation_slot(self):
        flow = self.make_flow(1, person_global_concurrency=1)
        both_started, release = asyncio.Event(), asyncio.Event()
        active = set()
        match_face = flow.pipeline.features.match_face
        extract_embeddings = flow.pipeline.features.extract_embeddings

        async def blocked(name, callback, *args, **kwargs):
            active.add(name)
            if len(active) == 2:
                both_started.set()
            try:
                await release.wait()
                return await callback(*args, **kwargs)
            finally:
                active.remove(name)

        async def blocked_face(*args, **kwargs):
            return await blocked("face", match_face, *args, **kwargs)

        async def blocked_embeddings(*args, **kwargs):
            return await blocked("embeddings", extract_embeddings, *args, **kwargs)

        flow.pipeline.features.match_face = blocked_face
        flow.pipeline.features.extract_embeddings = blocked_embeddings
        task = self.start_flow(flow)
        await self.reached(both_started)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(active, set())
        self.assertEqual(flow.buffer.recognitions, [])
        release.set()
        await asyncio.wait_for(self.run_flow(flow, "next-frame"), timeout=2)
        self.assertEqual(len(flow.buffer.recognitions), 1)

    async def test_upload_overlaps_embeddings_but_face_match_and_commit_wait_for_path(self):
        flow = self.make_flow(1, person_global_concurrency=1)
        upload_started, release_upload, embeddings_finished = (
            asyncio.Event(), asyncio.Event(), asyncio.Event()
        )
        matched_paths = []
        extract = flow.pipeline.features.extract_embeddings
        match = flow.pipeline.features.match_face

        async def blocked_upload(ctx, storage, prepared):
            upload_started.set()
            await release_upload.wait()
            return replace(prepared, crop_path="finished-crop.jpg")

        async def observe_embeddings(prepared):
            await upload_started.wait()
            self.assertEqual(prepared.crop_path, "")
            result = await extract(prepared)
            embeddings_finished.set()
            return result

        async def observe_face(ctx, frame, prepared, *args, **kwargs):
            matched_paths.append(prepared.crop_path)
            return await match(ctx, frame, prepared, *args, **kwargs)

        flow.pipeline.preparation.upload_prepared = blocked_upload
        flow.pipeline.features.extract_embeddings = observe_embeddings
        flow.pipeline.features.match_face = observe_face
        task = self.start_flow(flow)
        await self.reached(embeddings_finished)
        self.assertEqual(matched_paths, [])
        self.assertEqual(flow.buffer.detections, [])
        self.assertEqual(flow.buffer.recognitions, [])
        self.assertEqual(flow.buffer.features, [])
        release_upload.set()
        await task
        self.assertEqual(matched_paths, ["finished-crop.jpg"])
        self.assertEqual(len(flow.buffer.recognitions), 1)

    async def test_upload_waits_do_not_hold_visual_slots_across_frames(self):
        flow = self.make_flow(1, person_frame_concurrency=1, person_global_concurrency=1)
        release, both_inferred = asyncio.Event(), asyncio.Event()
        uploads, inferred = set(), 0
        extract = flow.pipeline.features.extract_embeddings

        async def blocked_upload(ctx, storage, prepared):
            uploads.add(ctx.keyframe_id)
            await release.wait()
            return replace(prepared, crop_path="finished-crop.jpg")

        async def observe_embeddings(prepared):
            nonlocal inferred
            result = await extract(prepared)
            inferred += 1
            if inferred == 2:
                both_inferred.set()
            return result

        flow.pipeline.preparation.upload_prepared = blocked_upload
        flow.pipeline.features.extract_embeddings = observe_embeddings
        tasks = [self.start_flow(flow, name) for name in ("one", "two")]
        await self.reached(both_inferred)
        self.assertEqual(uploads, {"one", "two"})
        self.assertEqual(flow.buffer.recognitions, [])
        release.set()
        await asyncio.gather(*tasks)
        self.assertEqual(len(flow.buffer.recognitions), 2)

    async def test_empty_upload_path_preserves_person_inference_and_identity(self):
        flow = self.make_flow(1)
        matched_paths = []
        match = flow.pipeline.features.match_face

        async def failed_upload(ctx, storage, prepared):
            return replace(prepared, crop_path="")

        async def observe_face(ctx, frame, prepared, *args, **kwargs):
            matched_paths.append(prepared.crop_path)
            self.assertTrue(kwargs["enable_recognition"])
            return await match(ctx, frame, prepared, *args, **kwargs)

        flow.pipeline.preparation.upload_prepared = failed_upload
        flow.pipeline.features.match_face = observe_face
        await self.run_flow(flow)
        self.assertEqual(matched_paths, [""])
        self.assertEqual(flow.triton.sbs_calls, 1)
        self.assertEqual(len(flow.buffer.recognitions), 1)
        self.assertEqual(len(flow.buffer.features), 1)

    async def test_upload_exception_drains_embedding_branch_before_failure(self):
        flow = self.make_flow(1)
        inference_started, inference_stopped = asyncio.Event(), asyncio.Event()
        failure = RuntimeError("upload exploded")

        async def failed_upload(ctx, storage, prepared):
            await inference_started.wait()
            raise failure

        async def blocked_embeddings(prepared):
            inference_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                inference_stopped.set()

        flow.pipeline.preparation.upload_prepared = failed_upload
        flow.pipeline.features.extract_embeddings = blocked_embeddings
        with self.assertRaises(RuntimeError) as caught:
            await self.run_flow(flow)
        self.assertIs(caught.exception, failure)
        self.assertTrue(inference_stopped.is_set())
        self.assertEqual(flow.buffer.detections, [])
        self.assertEqual(flow.buffer.recognitions, [])

    async def test_embedding_exception_drains_upload_before_failure(self):
        flow = self.make_flow(1)
        upload_started, upload_stopped = asyncio.Event(), asyncio.Event()
        failure = RuntimeError("inference exploded")

        async def blocked_upload(ctx, storage, prepared):
            upload_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                upload_stopped.set()

        async def failed_embeddings(prepared):
            await upload_started.wait()
            raise failure

        flow.pipeline.preparation.upload_prepared = blocked_upload
        flow.pipeline.features.extract_embeddings = failed_embeddings
        with self.assertRaises(RuntimeError) as caught:
            await self.run_flow(flow)
        self.assertIs(caught.exception, failure)
        self.assertTrue(upload_stopped.is_set())
        self.assertEqual(flow.buffer.detections, [])
        self.assertEqual(flow.buffer.recognitions, [])

    async def test_repeated_cancellation_waits_for_upload_cleanup(self):
        flow = self.make_flow(1)
        upload_started, cleanup_started, release_cleanup = (
            asyncio.Event(), asyncio.Event(), asyncio.Event()
        )
        cleanup_finished = asyncio.Event()

        async def blocked_upload(ctx, storage, prepared):
            upload_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_finished.set()

        flow.pipeline.preparation.upload_prepared = blocked_upload
        task = self.start_flow(flow)
        await self.reached(upload_started)
        task.cancel()
        await self.reached(cleanup_started)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release_cleanup.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cleanup_finished.is_set())
        self.assertEqual(flow.buffer.detections, [])
        self.assertEqual(flow.buffer.recognitions, [])

    async def test_slow_first_upload_preserves_window_and_identity_order(self):
        flow = self.make_flow(5, person_frame_concurrency=2, person_global_concurrency=1)
        second_ready, release_first = asyncio.Event(), asyncio.Event()
        started = []

        async def delayed_upload(ctx, storage, prepared):
            index = prepared.observation.segment_index
            started.append(index)
            if index == 0:
                await release_first.wait()
            elif index == 1:
                second_ready.set()
            return replace(prepared, crop_path=f"crop-{index}.jpg")

        flow.pipeline.preparation.upload_prepared = delayed_upload
        task = self.start_flow(flow)
        await self.reached(second_ready)
        await asyncio.sleep(0)
        self.assertEqual(started, [0, 1])
        self.assertEqual(flow.buffer.recognitions, [])
        release_first.set()
        await task
        self.assertEqual(
            [record["reid"] for record in flow.buffer.recognitions[:2]],
            ["first-id", "second-id"],
        )
        self.assertEqual(started, list(range(5)))

    async def test_dropped_quality_or_empty_encoding_never_starts_embeddings(self):
        for gate in ("quality", "encoding"):
            with self.subTest(gate=gate):
                flow = self.make_flow(1)
                prepare = flow.pipeline.preparation.prepare_visual
                match_face = flow.pipeline.features.match_face
                recognition_enabled = []

                async def gated_prepare(*args, **kwargs):
                    prepared = await prepare(*args, **kwargs)
                    if gate == "encoding":
                        return replace(prepared, image_bytes=b"")
                    return replace(
                        prepared,
                        quality_decision=replace(
                            prepared.quality_decision,
                            person_quality_level=QUALITY_LEVEL_DROP,
                        ),
                    )

                async def observe_face(*args, **kwargs):
                    recognition_enabled.append(kwargs["enable_recognition"])
                    return await match_face(*args, **kwargs)

                async def unexpected_embeddings(*args, **kwargs):
                    self.fail("a failed quality/encoding gate must not start embeddings")

                flow.pipeline.preparation.prepare_visual = gated_prepare
                flow.pipeline.features.match_face = observe_face
                flow.pipeline.features.extract_embeddings = unexpected_embeddings
                await self.run_flow(flow)
                self.assertEqual(recognition_enabled, [False])
                self.assertEqual(flow.buffer.recognitions, [])
                self.assertEqual(len(flow.buffer.detections), 1)

    async def test_ready_metrics_exclude_running_inference_and_drain_on_every_exit(self):
        for outcome in ("success", "cancel", "commit_failure"):
            with self.subTest(outcome=outcome):
                flow = self.make_flow(2, person_global_concurrency=2)
                metrics = PipelineMetrics()
                release_first, second_ready = asyncio.Event(), asyncio.Event()
                committed = []

                async def prepare(ctx, frame, oss, observation, *args):
                    index = observation.segment_index
                    if index == 0:
                        await release_first.wait()
                    else:
                        second_ready.set()
                    return SimpleNamespace(index=index)

                async def commit(ctx, frame_id, picture_id, object_name, analysis, *args):
                    if outcome == "commit_failure":
                        raise RuntimeError("commit unavailable")
                    committed.append(analysis.index)

                flow.pipeline._prepare_observation = prepare
                flow.pipeline._commit_observation = commit
                with patch("core.person_pipeline.pipeline.pipeline_metrics", metrics):
                    task = self.start_flow(flow)
                    await self.reached(second_ready)
                    stages = metrics.snapshot()["stages"]
                    self.assertEqual(stages["person_prepare"]["active"], 1)
                    self.assertEqual(stages["person_ordered_commit"]["waiting"], 1)
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
