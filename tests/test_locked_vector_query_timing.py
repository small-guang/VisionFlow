"""锁内人脸查询分项计时与真实共享搜索池、微批及取消的契约。"""

import asyncio
import threading
import time
from types import SimpleNamespace
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from core.execution import AsyncWorkExecutor
from core.face_query_observability import (
    face_identity_lock_metrics_scope, face_identity_locked_query_scope,
)
from core.performance import RuntimePerformance
from core.vector_query import VectorQueryCoordinator
from tests.test_execution_resources import wait_thread_event
from tests.test_face_gallery import load_gallery_modules
from core.identity import FaceIdentityService


PARAMETERS = {"collection_name": "face", "anns_field": "embedding", "limit": 10}
NAMES = (
    "face_identity_locked_query_admission",
    "face_identity_locked_query_batch",
    "face_identity_locked_query_executor",
    "face_identity_locked_query_sdk",
    "face_identity_locked_query_resume",
)


class Client:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return [[{"id": vector[0]}] for vector in kwargs["data"]]


class LockedVectorQueryTimingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.metrics = RuntimePerformance()
        self.executor = AsyncWorkExecutor(1, name="vector_search", metrics=self.metrics)
        self.coordinator = VectorQueryCoordinator(
            self.executor, batch_size=2, wait_ms=10, capacity=4, metrics=self.metrics,
        )
        self.addAsyncCleanup(self.executor.close)
        self.addAsyncCleanup(self.coordinator.close)

    async def locked_search(self, client, value):
        with face_identity_lock_metrics_scope(self.metrics):
            with face_identity_locked_query_scope():
                return await self.coordinator.search(client, [value], **PARAMETERS)

    def operations(self):
        return self.metrics.snapshot()["operations"]

    async def test_mixed_batch_records_only_locked_query_and_keeps_results(self):
        client = Client()
        locked = asyncio.create_task(self.locked_search(client, 1))
        unlocked = asyncio.create_task(self.coordinator.search(client, [2], **PARAMETERS))
        self.assertEqual(await asyncio.gather(locked, unlocked), [[{"id": 1}], [{"id": 2}]])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["data"], [[1], [2]])
        self.assertEqual({k: v for k, v in client.calls[0].items() if k != "data"}, PARAMETERS)
        operations = self.operations()
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(operations[name]["count"], 1)
                self.assertGreaterEqual(operations[name]["total_ms"], 0)
        self.assertEqual(operations["vector_query_rpc_run"]["count"], 1)
        self.assertEqual(operations["vector_query_batch_wait"]["count"], 2)

    async def test_blocked_shared_pool_and_sdk_are_measured_separately(self):
        pool_entered, pool_release = threading.Event(), threading.Event()
        sdk_entered, sdk_release = threading.Event(), threading.Event()
        client = Client()
        original = client.search

        def block_pool():
            pool_entered.set()
            if not pool_release.wait(3):
                raise TimeoutError("共享线程池任务未释放")

        def block_sdk(**kwargs):
            sdk_entered.set()
            if not sdk_release.wait(3):
                raise TimeoutError("SDK 测试任务未释放")
            return original(**kwargs)

        client.search = block_sdk
        occupying = asyncio.create_task(self.executor.run(block_pool))
        query = None
        try:
            await wait_thread_event(pool_entered)
            query = asyncio.create_task(self.locked_search(client, 1))
            async with asyncio.timeout(2):
                while self.executor.queue_snapshot()["queued_jobs"] == 0:
                    await asyncio.sleep(.001)
            await asyncio.sleep(.04)
            pool_release.set()
            await wait_thread_event(sdk_entered)
            await asyncio.sleep(.04)
            sdk_release.set()
            self.assertEqual(await query, [{"id": 1}])
            await occupying
            operations = self.operations()
            self.assertGreaterEqual(operations[NAMES[2]]["total_ms"], 25)
            self.assertGreaterEqual(operations[NAMES[3]]["total_ms"], 25)
            self.assertEqual(operations[NAMES[4]]["count"], 1)
            self.assertGreaterEqual(operations[NAMES[4]]["total_ms"], 0)
            # 保留共享池原指标：占位任务和查询任务均计数。
            self.assertEqual(operations["vector_search_queue_wait"]["count"], 2)
        finally:
            pool_release.set()
            sdk_release.set()
            await asyncio.gather(*(t for t in (occupying, query) if t), return_exceptions=True)

    async def test_cancelled_locked_query_does_not_cancel_batch_sibling(self):
        entered, release = threading.Event(), threading.Event()
        client = Client()
        original = client.search

        def block(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("SDK 测试任务未释放")
            return original(**kwargs)

        client.search = block
        locked = asyncio.create_task(self.locked_search(client, 1))
        sibling = asyncio.create_task(self.coordinator.search(client, [2], **PARAMETERS))
        try:
            await wait_thread_event(entered)
            locked.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await locked
            self.assertEqual(self.coordinator.queue_snapshot()["admitted_queries"], 2)
            release.set()
            self.assertEqual(await sibling, [{"id": 2}])
            self.assertEqual(self.coordinator.queue_snapshot()["admitted_queries"], 0)
            self.assertEqual(len(client.calls), 1)
            operations = self.operations()
            for name in NAMES[:2]:
                self.assertEqual(operations[name]["count"], 1)
            for name in NAMES[2:]:
                self.assertEqual(operations[name]["count"], 0)
        finally:
            release.set()
            await asyncio.gather(locked, sibling, return_exceptions=True)

    async def test_resume_includes_delayed_event_loop_notification(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        client = Client()
        original = client.search

        def controlled_search(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("SDK 测试任务未释放")
            result = original(**kwargs)
            finished.set()
            return result

        client.search = controlled_search
        task = asyncio.create_task(self.locked_search(client, 1))
        try:
            await wait_thread_event(entered)
            release.set()
            # 有意阻塞所属 loop：先确认 SDK 已结束，再延迟通知和调用方恢复。
            self.assertTrue(finished.wait(1))
            time.sleep(.06)
            self.assertEqual(await task, [{"id": 1}])
            resume = self.operations()["face_identity_locked_query_resume"]
            self.assertEqual(resume["count"], 1)
            self.assertGreaterEqual(resume["total_ms"], 40)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancel_after_sdk_completion_before_caller_resumes(self):
        client = Client()
        original = self.coordinator._record_locked
        task = None

        def cancel_before_resume(metrics, name, duration_ms, **kwargs):
            original(metrics, name, duration_ms, **kwargs)
            if name == "face_identity_locked_query_sdk":
                # 同步结果分发完成后，取消回调先于 shield 唤醒调用方执行。
                asyncio.get_running_loop().call_soon(task.cancel)

        with patch.object(self.coordinator, "_record_locked", side_effect=cancel_before_resume):
            task = asyncio.create_task(self.locked_search(client, 1))
            with self.assertRaises(asyncio.CancelledError):
                await task
        operations = self.operations()
        for name in NAMES[:4]:
            self.assertEqual(operations[name]["count"], 1)
        self.assertEqual(operations[NAMES[4]]["count"], 0)
        self.assertEqual(self.coordinator.queue_snapshot()["admitted_queries"], 0)

    async def test_sdk_error_reaches_both_queries_and_releases_capacity(self):
        failure = RuntimeError("milvus search failed")

        def fail(**kwargs):
            raise failure

        client = Client()
        client.search = fail
        locked = asyncio.create_task(self.locked_search(client, 1))
        sibling = asyncio.create_task(self.coordinator.search(client, [2], **PARAMETERS))
        results = await asyncio.gather(locked, sibling, return_exceptions=True)
        self.assertTrue(all(result is failure for result in results))
        self.assertEqual(self.coordinator.queue_snapshot()["admitted_queries"], 0)
        self.assertEqual(self.coordinator.queue_snapshot()["failed_queries"], 2)
        operations = self.operations()
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(operations[name]["count"], 1)
        self.assertEqual(operations[NAMES[3]]["errors"], 1)
        self.assertEqual(operations[NAMES[4]]["errors"], 1)

    async def test_child_task_does_not_inherit_lock_ownership(self):
        client = Client()
        with face_identity_lock_metrics_scope(self.metrics):
            with face_identity_locked_query_scope():
                child = asyncio.create_task(self.coordinator.search(client, [1], **PARAMETERS))
                self.assertEqual(await child, [{"id": 1}])
        operations = self.operations()
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(operations[name]["count"], 0)

    async def test_query_scope_without_lock_does_not_record(self):
        client = Client()
        with face_identity_locked_query_scope():
            self.assertEqual(await self.coordinator.search(client, [1], **PARAMETERS), [{"id": 1}])
        operations = self.operations()
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(operations[name]["count"], 0)

    async def test_identity_service_to_real_face_gallery_records_locked_query(self):
        _, module = load_gallery_modules()
        gallery = object.__new__(module.FaceGallery)
        gallery.client = SimpleNamespace(search=lambda **kwargs: [[] for _ in kwargs["data"]])
        gallery.collection_name = "face"
        gallery.query_coordinator = self.coordinator
        # pending 无版本接口时退回锁内完整搜索，覆盖生产链路而非人工设置 scope。
        service = FaceIdentityService(
            face_gallery=gallery, pending_faces=None, face_features=SimpleNamespace(),
            records=SimpleNamespace(), logger=SimpleNamespace(), metrics=self.metrics,
        )
        self.assertEqual(await service.search(np.array([1., 0.], dtype=np.float32), .45),
                         (None, 0., ""))
        operations = self.operations()
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(operations[name]["count"], 1)

    async def test_lock_scope_without_face_query_scope_does_not_record(self):
        client = Client()
        with face_identity_lock_metrics_scope(self.metrics):
            self.assertEqual(await self.coordinator.search(client, [1], **PARAMETERS), [{"id": 1}])
        operations = self.operations()
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(operations[name]["count"], 0)


if __name__ == "__main__":
    unittest.main()
