"""有界锁外刷新必须保留身份原子登记、版本复核和取消边界。"""

import asyncio
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import numpy as np

from core.identity import FaceIdentityRequest, FaceIdentityService
from tests.test_face_candidate_prefetch import ControlledGallery, ControlledPending
from tests.test_keyframe_flow import DummyLogger
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_ENROLL
from utils.reid.face_matching import FaceSampleMatch


class RecordingMetrics:
    def __init__(self):
        self.counts = Counter()

    def record(self, operation, *_args, **_kwargs):
        self.counts[operation] += 1

    @contextmanager
    def measure(self, operation):
        try:
            yield
        finally:
            self.record(operation)


class FaceIdentityRefreshTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.gallery = ControlledGallery()
        self.pending = ControlledPending()
        self.lock = asyncio.Lock()
        self.metrics = RecordingMetrics()
        self.features = []
        self.records = []
        self.ids = []
        self.built = []
        self.embedding = np.array([1.0, 0.0], dtype=np.float32)
        self.request = FaceIdentityRequest(
            self.embedding, QUALITY_LEVEL_ENROLL,
            {"device_id": "camera", "keyframe_id": "frame"},
        )

        def create_id():
            self.assertTrue(self.lock.locked())
            face_id = f"created-{len(self.ids) + 1}"
            self.ids.append(face_id)
            return face_id

        async def add_feature(**feature):
            self.assertTrue(self.lock.locked())
            self.features.append(feature)
            self.pending.samples.append(self.sample(feature["face_id"]))

        async def add_record(record):
            self.assertFalse(self.lock.locked())
            self.records.append(record)

        self.service = FaceIdentityService(
            face_gallery=self.gallery, pending_faces=self.pending,
            face_features=SimpleNamespace(add_face_feature=add_feature),
            records=SimpleNamespace(add_face_recognition_record=add_record),
            logger=DummyLogger(), identity_lock=self.lock, metrics=self.metrics,
            id_factory=create_id,
        )

    @staticmethod
    def sample(face_id="existing"):
        return FaceSampleMatch(face_id, .99, "sample", "crop.jpg")

    def build_record(self, outcome):
        self.assertTrue(self.lock.locked())
        record = {"face_id": outcome.face_id, "status": outcome.status}
        self.built.append(record)
        return record

    async def commit(self):
        return await self.service.commit(self.request, self.build_record)

    async def stop_task(self, task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_stable_miss_does_not_refresh_and_newborn_record_stays_with_feature(self):
        result = await self.commit()
        self.assertEqual(result.face_id, "created-1")
        self.assertEqual(len(self.gallery.calls), 1)
        self.assertEqual(self.ids, ["created-1"])
        self.assertEqual(self.records, [])
        self.assertEqual(self.features[0]["new_face_recognition_record"], self.built[0])
        self.assertEqual(self.metrics.counts["face_identity_requests"], 1)
        self.assertEqual(self.metrics.counts["face_identity_refresh"], 0)
        self.assertEqual(self.metrics.counts["face_identity_locked_search"], 0)

    async def test_refresh_releases_lock_and_merges_pending_registered_during_remote_read(self):
        refreshing, release = asyncio.Event(), asyncio.Event()

        async def remote():
            self.assertFalse(self.lock.locked())
            if len(self.gallery.calls) == 1:
                self.gallery.revision += 2
            else:
                refreshing.set()
                await release.wait()

        self.gallery.search_hook = remote
        task = asyncio.create_task(self.commit())
        try:
            await asyncio.wait_for(refreshing.wait(), 1)
            async with asyncio.timeout(1):
                async with self.lock:
                    self.pending.samples.append(self.sample("registered-while-refreshing"))
            self.assertEqual(self.ids, [])
            self.assertEqual(self.features, [])
            self.assertEqual(self.built, [])
            release.set()
            result = await asyncio.wait_for(task, 1)
            self.assertEqual(result.face_id, "registered-while-refreshing")
            self.assertEqual(result.status, 2)
            self.assertEqual(self.ids, [])
            self.assertEqual(len(self.records), 1)
            self.assertEqual(len(self.gallery.calls), 2)
            self.assertEqual(self.metrics.counts["face_identity_refresh"], 1)
            self.assertEqual(self.metrics.counts["face_identity_refresh_used"], 1)
            self.assertEqual(self.metrics.counts["face_identity_locked_search"], 0)
        finally:
            release.set()
            await self.stop_task(task)

    async def test_two_stale_empty_queries_refresh_then_register_only_one_identity(self):
        both_reading, release = asyncio.Event(), asyncio.Event()

        async def remote():
            if len(self.gallery.calls) <= 2:
                self.gallery.revision += 2
                if len(self.gallery.calls) == 2:
                    both_reading.set()
                await release.wait()

        self.gallery.search_hook = remote
        tasks = [asyncio.create_task(self.commit()) for _ in range(2)]
        try:
            await asyncio.wait_for(both_reading.wait(), 1)
            release.set()
            first, second = await asyncio.wait_for(asyncio.gather(*tasks), 1)
            self.assertEqual(first.face_id, second.face_id)
            self.assertEqual(self.ids, ["created-1"])
            self.assertEqual(sorted([first.status, second.status]), [1, 2])
            self.assertEqual(sum("new_face_recognition_record" in f for f in self.features), 1)
            self.assertEqual(len(self.records), 1)
            self.assertEqual(self.metrics.counts["face_identity_requests"], 2)
        finally:
            release.set()
            for task in tasks:
                await self.stop_task(task)

    async def test_invalid_refresh_is_bounded_and_full_search_stays_under_lock(self):
        lock_states = []

        async def remote():
            lock_states.append(self.lock.locked())
            if len(self.gallery.calls) <= 2:
                self.gallery.revision += 2
                if len(self.gallery.calls) == 2:
                    self.gallery.samples = [self.sample("persisted-during-refresh")]
                    self.pending.revision += 1

        self.gallery.search_hook = remote
        result = await self.commit()
        self.assertEqual(result.face_id, "persisted-during-refresh")
        self.assertEqual(self.ids, [])
        self.assertEqual(lock_states, [False, False, True])
        self.assertEqual(self.metrics.counts["face_identity_refresh"], 1)
        self.assertEqual(self.metrics.counts["face_identity_refresh_invalid"], 1)
        self.assertEqual(self.metrics.counts["face_identity_refresh_exhausted"], 1)
        self.assertEqual(self.metrics.counts["face_identity_locked_search"], 1)

    async def test_active_writer_without_stable_state_immediately_uses_original_fallback(self):
        self.gallery.active_writers = 1
        lock_states = []

        async def remote():
            lock_states.append(self.lock.locked())

        self.gallery.search_hook = remote
        await self.commit()
        self.assertEqual(lock_states, [True])
        self.assertEqual(self.metrics.counts["face_identity_refresh"], 0)
        self.assertEqual(self.metrics.counts["face_identity_refresh_unavailable"], 1)
        self.assertEqual(self.metrics.counts["face_identity_locked_search"], 1)

    async def test_initially_active_writer_can_refresh_once_it_finishes_before_commit(self):
        reads = 0

        def revision():
            nonlocal reads
            reads += 1
            return 0, int(reads == 1)

        self.gallery.candidate_revision = revision
        lock_states = []

        async def remote():
            lock_states.append(self.lock.locked())

        self.gallery.search_hook = remote
        await self.commit()
        self.assertEqual(lock_states, [False])
        self.assertEqual(self.metrics.counts["face_identity_refresh"], 1)
        self.assertEqual(self.metrics.counts["face_identity_refresh_used"], 1)
        self.assertEqual(self.metrics.counts["face_identity_locked_search"], 0)

    async def test_refresh_failure_then_final_search_failure_cannot_create_an_identity(self):
        async def remote():
            if len(self.gallery.calls) == 1:
                self.gallery.revision += 2
            else:
                raise OSError("gallery unavailable")

        self.gallery.search_hook = remote
        result = await self.commit()
        self.assertEqual(result.status, 0)
        self.assertTrue(result.search_failure_reason)
        self.assertEqual(self.ids, [])
        self.assertEqual(self.features, [])
        self.assertEqual(len(self.records), 1)
        self.assertEqual(len(self.gallery.calls), 3)
        self.assertEqual(self.metrics.counts["face_identity_refresh_failed"], 1)
        self.assertEqual(self.metrics.counts["face_identity_locked_search"], 1)
        self.assertFalse(self.lock.locked())

    async def test_pending_change_while_awaiting_refresh_validation_is_rechecked(self):
        async def remote():
            if len(self.gallery.calls) == 1:
                self.gallery.revision += 2

        async def pending_read():
            if len(self.pending.calls) == 1:
                self.gallery.samples = [self.sample("flushed")]
                self.gallery.revision += 2
                self.pending.revision += 1
                await asyncio.sleep(0)

        self.gallery.search_hook = remote
        self.pending.search_hook = pending_read
        result = await self.commit()
        self.assertEqual(result.face_id, "flushed")
        self.assertEqual(self.ids, [])
        self.assertEqual(len(self.gallery.calls), 3)
        self.assertEqual(self.metrics.counts["face_identity_refresh_invalid"], 1)

    async def test_cancelling_refresh_has_no_identity_side_effect_and_releases_lock(self):
        refreshing = asyncio.Event()

        async def remote():
            if len(self.gallery.calls) == 1:
                self.gallery.revision += 2
            else:
                refreshing.set()
                await asyncio.Event().wait()

        self.gallery.search_hook = remote
        task = asyncio.create_task(self.commit())
        try:
            await asyncio.wait_for(refreshing.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(self.lock.locked())
            self.assertEqual(self.ids, [])
            self.assertEqual(self.features, [])
            self.assertEqual(self.built, [])
            self.assertEqual(self.records, [])
            self.assertEqual(len(self.gallery.calls), 2)
            self.assertEqual(self.metrics.counts["face_identity_locked_search"], 0)
        finally:
            await self.stop_task(task)

    async def test_business_failure_after_validated_refresh_is_never_retried(self):
        async def remote():
            if len(self.gallery.calls) == 1:
                self.gallery.revision += 2

        self.gallery.search_hook = remote

        def fail_record(_outcome):
            self.assertTrue(self.lock.locked())
            raise ValueError("record construction failed")

        with self.assertRaisesRegex(ValueError, "record construction failed"):
            await self.service.commit(self.request, fail_record)
        self.assertEqual(len(self.gallery.calls), 2)
        self.assertEqual(self.ids, ["created-1"])
        self.assertEqual(self.features, [])
        self.assertEqual(self.records, [])
        self.assertFalse(self.lock.locked())

    async def test_read_only_search_uses_refresh_and_propagates_terminal_error(self):
        async def remote():
            if len(self.gallery.calls) == 1:
                self.gallery.revision += 2
            else:
                raise OSError("gallery unavailable")

        self.gallery.search_hook = remote
        with self.assertRaisesRegex(OSError, "gallery unavailable"):
            await self.service.search(self.embedding, .45, ["camera"])
        self.assertEqual(self.gallery.calls, [["camera"], ["camera"], ["camera"]])
        self.assertEqual(self.ids, [])
        self.assertEqual(self.features, [])
        self.assertEqual(self.records, [])
        self.assertFalse(self.lock.locked())

    async def test_disabled_prefetch_keeps_legacy_callback_under_lock(self):
        self.service.search_service = None
        lock_states = []

        async def fallback(*_args, **_kwargs):
            lock_states.append(self.lock.locked())
            return "legacy", .99, "legacy"

        self.service.search_faces = fallback
        result = await self.service.search(self.embedding, .45)
        self.assertEqual(result, ("legacy", .99, "legacy"))
        self.assertEqual(lock_states, [True])
        self.assertEqual(self.metrics.counts["face_identity_refresh"], 0)
        self.assertEqual(self.metrics.counts["face_identity_locked_search"], 1)


if __name__ == "__main__":
    unittest.main()
