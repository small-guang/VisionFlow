"""身份锁内与锁外 Face 查询的阶段指标必须隔离。"""

import asyncio
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.identity import FaceIdentityService
from core.performance import RuntimePerformance
from tests.test_face_candidate_prefetch import ControlledGallery, ControlledPending
from tests.test_keyframe_flow import DummyLogger


class RecordingMetrics:
    def __init__(self):
        self.counts = Counter()

    def record(self, name, *_args, **_kwargs):
        self.counts[name] += 1

    @contextmanager
    def measure(self, name):
        try:
            yield
        finally:
            self.record(name)


class FaceLockedSearchMetricsTests(unittest.IsolatedAsyncioTestCase):
    METRICS = (
        "face_identity_locked_full_pending",
        "face_identity_locked_full_milvus",
        "face_identity_locked_full_resolve",
        "face_identity_locked_prefetch_pending",
        "face_identity_locked_prefetch_resolve",
    )

    def setUp(self):
        self.gallery = ControlledGallery()
        self.pending = ControlledPending()
        self.lock = asyncio.Lock()
        self.metrics = RecordingMetrics()
        self.embedding = np.array([1., 0.], dtype=np.float32)
        self.service = FaceIdentityService(
            face_gallery=self.gallery,
            pending_faces=self.pending,
            face_features=SimpleNamespace(),
            records=SimpleNamespace(),
            logger=DummyLogger(),
            identity_lock=self.lock,
            metrics=self.metrics,
        )
        patcher = patch("utils.reid.face_search.runtime_performance", self.metrics)
        self.search_patcher = patcher
        patcher.start()
        self.addCleanup(patcher.stop)

    def stage_counts(self):
        return {name: self.metrics.counts[name] for name in self.METRICS}

    def expected(self, *active):
        return {name: int(name in active) for name in self.METRICS}

    async def test_full_search_measures_all_three_stages_under_lock(self):
        self.gallery.active_writers = 1  # 无稳定快照，直接执行锁内完整搜索。

        async def check_locked():
            self.assertTrue(self.lock.locked())

        self.pending.search_hook = check_locked
        self.gallery.search_hook = check_locked
        self.assertEqual(await self.service.search(self.embedding, .45), (None, 0., ""))
        self.assertEqual(self.stage_counts(), self.expected(*self.METRICS[:3]))

    async def test_prefetch_measures_only_locked_pending_and_resolve(self):
        async def check_unlocked():
            self.assertFalse(self.lock.locked())

        async def check_locked():
            self.assertTrue(self.lock.locked())

        self.gallery.search_hook = check_unlocked
        self.pending.search_hook = check_locked
        self.assertEqual(await self.service.search(self.embedding, .45), (None, 0., ""))
        self.assertEqual(self.stage_counts(), self.expected(*self.METRICS[3:]))

    async def test_invalidated_prefetch_counts_pending_but_not_resolve(self):
        async def invalidate_during_pending_read():
            self.pending.search_hook = None
            self.pending.revision += 1
            self.gallery.active_writers = 1

        self.pending.search_hook = invalidate_during_pending_read
        self.assertEqual(await self.service.search(self.embedding, .45), (None, 0., ""))
        self.assertEqual(self.stage_counts(), self.expected(*self.METRICS[:4]))

    async def test_direct_search_outside_lock_does_not_increment_locked_metrics(self):
        self.assertEqual(await self.service.search_service.search(self.embedding, .45),
                         (None, 0., ""))
        self.assertEqual(self.stage_counts(), self.expected())

    async def test_child_task_does_not_inherit_lock_ownership_for_metrics(self):
        self.gallery.active_writers = 1

        async def run_child_search():
            self.assertTrue(self.lock.locked())
            # 防止子任务再次进入本 hook；它未获得身份锁，即使创建时锁已持有。
            self.pending.search_hook = None
            child = asyncio.create_task(
                self.service.search_service.search(self.embedding, .45),
            )
            self.assertEqual(await child, (None, 0., ""))
            self.assertEqual(self.stage_counts(), self.expected())

        self.pending.search_hook = run_child_search
        self.assertEqual(await self.service.search(self.embedding, .45), (None, 0., ""))
        self.assertEqual(self.stage_counts(), self.expected(*self.METRICS[:3]))

    async def test_real_runtime_performance_exports_all_stages(self):
        real_metrics = RuntimePerformance()
        self.service.metrics = real_metrics
        with patch("utils.reid.face_search.runtime_performance", real_metrics):
            self.gallery.active_writers = 1
            await self.service.search(self.embedding, .45)
            self.gallery.active_writers = 0
            await self.service.search(self.embedding, .45)
        operations = real_metrics.snapshot()["operations"]
        for name in self.METRICS:
            with self.subTest(name=name):
                self.assertEqual(operations[name]["count"], 1)
                self.assertEqual(operations[name]["errors"], 0)
                self.assertGreaterEqual(operations[name]["total_ms"], 0)

    async def test_stage_metrics_use_service_injected_collector(self):
        self.search_patcher.stop()
        real_metrics = RuntimePerformance()
        self.service.metrics = real_metrics
        self.gallery.active_writers = 1
        await self.service.search(self.embedding, .45)
        self.gallery.active_writers = 0
        await self.service.search(self.embedding, .45)
        operations = real_metrics.snapshot()["operations"]
        for name in self.METRICS:
            with self.subTest(name=name):
                self.assertEqual(operations[name]["count"], 1)
        self.assertEqual(operations["face_milvus_search"]["count"], 0)

    async def test_real_runtime_performance_records_pending_failure(self):
        self.gallery.active_writers = 1
        real_metrics = RuntimePerformance()
        self.service.metrics = real_metrics

        async def fail():
            raise OSError("pending unavailable")

        self.pending.search_hook = fail
        with patch("utils.reid.face_search.runtime_performance", real_metrics):
            with self.assertRaisesRegex(OSError, "pending unavailable"):
                await self.service.search(self.embedding, .45)
        operations = real_metrics.snapshot()["operations"]
        self.assertEqual(operations[self.METRICS[0]]["count"], 1)
        self.assertEqual(operations[self.METRICS[0]]["errors"], 1)
        self.assertEqual(operations[self.METRICS[1]]["count"], 0)
        self.assertEqual(operations[self.METRICS[2]]["count"], 0)

    async def test_real_runtime_performance_records_pending_cancellation(self):
        self.gallery.active_writers = 1
        real_metrics = RuntimePerformance()
        self.service.metrics = real_metrics
        entered = asyncio.Event()

        async def wait_for_cancel():
            entered.set()
            await asyncio.Event().wait()

        self.pending.search_hook = wait_for_cancel
        with patch("utils.reid.face_search.runtime_performance", real_metrics):
            task = asyncio.create_task(self.service.search(self.embedding, .45))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        operations = real_metrics.snapshot()["operations"]
        self.assertEqual(operations[self.METRICS[0]]["count"], 1)
        self.assertEqual(operations[self.METRICS[0]]["errors"], 1)
        self.assertEqual(operations[self.METRICS[1]]["count"], 0)
        self.assertEqual(operations[self.METRICS[2]]["count"], 0)

    async def test_search_exception_clears_context_for_next_direct_search(self):
        self.gallery.active_writers = 1

        async def fail():
            raise OSError("gallery unavailable")

        self.gallery.search_hook = fail
        with self.assertRaisesRegex(OSError, "gallery unavailable"):
            await self.service.search(self.embedding, .45)
        before = self.stage_counts()
        self.assertEqual(before, self.expected(*self.METRICS[:2]))
        self.assertFalse(self.lock.locked())
        self.gallery.search_hook = None
        await self.service.search_service.search(self.embedding, .45)
        self.assertEqual(self.stage_counts(), before)

    async def test_cancel_clears_context_for_next_search_in_same_task(self):
        self.gallery.active_writers = 1
        entered = asyncio.Event()

        async def wait_for_cancel():
            entered.set()
            await asyncio.Event().wait()

        self.pending.search_hook = wait_for_cancel

        async def cancel_then_search():
            try:
                await self.service.search(self.embedding, .45)
            except asyncio.CancelledError:
                self.assertFalse(self.lock.locked())
                before = self.stage_counts()
                self.pending.search_hook = None
                await self.service.search_service.search(self.embedding, .45)
                self.assertEqual(self.stage_counts(), before)
                return before
            self.fail("应当传播取消")

        task = asyncio.create_task(cancel_then_search())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            self.assertEqual(await asyncio.wait_for(task, 1), self.expected(self.METRICS[0]))
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
