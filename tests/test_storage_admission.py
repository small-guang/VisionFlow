"""验证慢上传、取消及关闭时，下载保底和真实工作容量仍然有效。"""

import asyncio
from contextvars import ContextVar
import threading
import unittest
from unittest.mock import patch

from core.execution import ExecutionResources
from core.performance import RuntimePerformance
from core.storage_execution import StorageWorkExecutor
from tests.test_execution_resources import wait_thread_event
from utils.storage.execution import run_storage_operation


class StorageAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_notified_waiter_does_not_strand_later_upload(self):
        executor = StorageWorkExecutor(2, download_reserved=1)
        entered, release = threading.Event(), threading.Event()

        def upload():
            entered.set()
            release.wait(3)

        running = asyncio.create_task(executor.run_storage("upload", upload))
        first = second = None
        try:
            await wait_thread_event(entered)
            existing = executor._upload_tasks.copy()
            first = asyncio.create_task(executor.run_storage("upload", lambda: "cancelled"))
            for _ in range(3):
                await asyncio.sleep(0)
            victim, = executor._upload_tasks - existing
            second = asyncio.create_task(executor.run_storage("upload", lambda: "next"))
            for _ in range(3):
                await asyncio.sleep(0)
            self.assertEqual(executor.queue_snapshot()["upload_admission_waiters"], 2)
            original_notify = executor._upload_condition.notify

            def cancel_after_notify(count=1):
                original_notify(count)
                if not victim.done():
                    victim.cancel()

            # 精确构造：最后一个上传归还名额，首等待者已获通知但尚未恢复时取消。
            with patch.object(executor._upload_condition, "notify", side_effect=cancel_after_notify):
                release.set()
                await running
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertEqual(await asyncio.wait_for(second, 1), "next")
            self.assertEqual(executor.queue_snapshot()["queued_jobs"], 0)
            self.assertEqual(executor.queue_snapshot()["admitted_uploads"], 0)
        finally:
            release.set()
            for task in (first, second):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (running, first, second) if task), return_exceptions=True)
            await executor.close()

    async def test_explicit_drain_survives_repeated_cancel_and_honors_custom_metrics(self):
        default_metrics, custom_metrics = RuntimePerformance(), RuntimePerformance()
        executor = StorageWorkExecutor(2, download_reserved=1, metrics=default_metrics)
        entered, release = threading.Event(), threading.Event()

        def upload():
            entered.set()
            release.wait(3)

        task = asyncio.create_task(run_storage_operation(
            executor, "upload", upload, metrics=custom_metrics, drain_on_cancel=True,
        ))
        try:
            await wait_thread_event(entered)
            task.cancel()
            for _ in range(3):
                await asyncio.sleep(0)
            task.cancel()
            for _ in range(3):
                await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(executor.queue_snapshot()["admitted_uploads"], 1)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(executor.queue_snapshot()["admitted_uploads"], 0)
            self.assertEqual(custom_metrics.snapshot()["operations"]["storage_upload_run"]["count"], 1)
            self.assertEqual(default_metrics.snapshot()["operations"]["storage_upload_run"]["count"], 0)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await executor.close()

    async def test_slow_uploads_leave_download_capacity_without_adding_threads(self):
        executor = StorageWorkExecutor(3, download_reserved=1)
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()

        def upload(index):
            entered[index].set()
            if not release.wait(3):
                raise TimeoutError("upload test not released")
            return index

        tasks = [asyncio.create_task(run_storage_operation(executor, "upload", upload, i)) for i in range(2)]
        third = None
        try:
            await asyncio.gather(*(wait_thread_event(event) for event in entered))
            third = asyncio.create_task(run_storage_operation(executor, "upload", lambda: "queued"))
            for _ in range(3):
                await asyncio.sleep(0)
            self.assertEqual(executor.queue_snapshot()["upload_admission_waiters"], 1)
            self.assertEqual(executor.queue_snapshot()["admitted_uploads"], 2)
            self.assertFalse(third.done())
            with patch.object(asyncio.get_running_loop(), "run_in_executor", side_effect=AssertionError("default pool used")):
                self.assertEqual(await asyncio.wait_for(
                    run_storage_operation(executor, "download", lambda: b"image"), 1,
                ), b"image")
            self.assertEqual(executor.queue_snapshot()["capacity_jobs"], 3)
            self.assertEqual(executor._pool._executor._max_workers, 3)
            self.assertEqual(executor.queue_snapshot()["inflight_jobs"], 2)
        finally:
            release.set()
            await asyncio.gather(*tasks, *([third] if third else []), return_exceptions=True)
            await executor.close()

    async def test_downloads_can_use_the_whole_pool(self):
        executor = StorageWorkExecutor(3, download_reserved=1)
        entered = [threading.Event() for _ in range(3)]
        release = threading.Event()

        def download(index):
            entered[index].set()
            release.wait(3)

        tasks = [asyncio.create_task(executor.run_storage("download", download, i)) for i in range(3)]
        try:
            await asyncio.gather(*(wait_thread_event(event) for event in entered))
            self.assertEqual(executor.queue_snapshot()["inflight_jobs"], 3)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await executor.close()

    async def test_cancelled_upload_keeps_capacity_and_close_drains_after_repeated_cancel(self):
        executor = StorageWorkExecutor(2, download_reserved=1)
        entered, release = threading.Event(), threading.Event()
        calls = []

        def upload():
            entered.set()
            release.wait(3)
            calls.append("finished")

        running = asyncio.create_task(executor.run_storage("upload", upload))
        waiting = closing = None
        try:
            await wait_thread_event(entered)
            running.cancel()
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            self.assertEqual(executor.queue_snapshot()["admitted_uploads"], 1)
            self.assertEqual(executor.queue_snapshot()["inflight_jobs"], 1)
            waiting = asyncio.create_task(executor.run_storage("upload", lambda: calls.append("unexpected")))
            for _ in range(3):
                await asyncio.sleep(0)
            self.assertEqual(executor.queue_snapshot()["queued_jobs"], 1)
            self.assertEqual(await executor.run_storage("download", lambda: "download"), "download")
            closing = asyncio.create_task(executor.close())
            for _ in range(3):
                await asyncio.sleep(0)
            closing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await asyncio.wait_for(waiting, 1)
            self.assertFalse(executor._storage_close_task.done())
            release.set()
            await executor.close()
            self.assertEqual(calls, ["finished"])
            snapshot = executor.queue_snapshot()
            self.assertTrue(snapshot["closed"])
            self.assertEqual((snapshot["admitted_uploads"], snapshot["inflight_jobs"], snapshot["queued_jobs"]), (0, 0, 0))
            self.assertFalse(executor._upload_tasks)
        finally:
            release.set()
            await asyncio.gather(*(task for task in (running, waiting, closing) if task), return_exceptions=True)
            await executor.close()

    async def test_cancelled_upload_waiters_never_start_sdk_operation(self):
        for blocked_by_download in (False, True):
            with self.subTest(blocked_by_download=blocked_by_download):
                executor = StorageWorkExecutor(2, download_reserved=1)
                release = threading.Event()
                entered = [threading.Event() for _ in range(2 if blocked_by_download else 1)]
                calls = []

                def block(index):
                    entered[index].set()
                    release.wait(3)

                direction = "download" if blocked_by_download else "upload"
                blockers = [asyncio.create_task(executor.run_storage(direction, block, i)) for i in range(len(entered))]
                waiter = None
                try:
                    await asyncio.gather(*(wait_thread_event(event) for event in entered))
                    waiter = asyncio.create_task(executor.run_storage("upload", lambda: calls.append("unexpected")))
                    for _ in range(3):
                        await asyncio.sleep(0)
                    self.assertEqual(executor.queue_snapshot()["queued_jobs"], 1)
                    waiter.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await waiter
                    release.set()
                    await asyncio.gather(*blockers)
                    await executor.close()
                    self.assertEqual(calls, [])
                    self.assertEqual(executor.queue_snapshot()["admitted_uploads"], 0)
                finally:
                    release.set()
                    await asyncio.gather(*blockers, *([waiter] if waiter else []), return_exceptions=True)
                    await executor.close()

    async def test_queue_timings_include_upload_admission_and_preserve_context_and_errors(self):
        rows = []

        class Metrics:
            def record(self, name, duration_ms, *, failed=False):
                rows.append((name, duration_ms, failed, threading.get_ident()))

        executor = StorageWorkExecutor(2, download_reserved=1, metrics=Metrics())
        entered, release = threading.Event(), threading.Event()
        marker = ContextVar("storage-marker", default="absent")
        token = marker.set("request")
        failure = OSError("original SDK error")

        def block():
            entered.set()
            release.wait(3)

        def fail():
            self.assertEqual(marker.get(), "request")
            raise failure

        first = asyncio.create_task(executor.run_storage("upload", block))
        second = None
        try:
            await wait_thread_event(entered)
            second = asyncio.create_task(executor.run_storage("upload", fail))
            await asyncio.sleep(0.025)
            release.set()
            await first
            with self.assertRaises(OSError) as raised:
                await second
            self.assertIs(raised.exception, failure)
            self.assertEqual(await executor.run_storage("upload", lambda: 42), 42)
            waits = [row[1] for row in rows if row[0] == "storage_upload_queue_wait"]
            totals = [row[1] for row in rows if row[0] == "storage_io_queue_wait"]
            self.assertEqual(waits, totals)
            self.assertGreater(waits[1], 15)
            self.assertEqual(sum(row[2] for row in rows if row[0] == "storage_upload_run"), 1)
            self.assertTrue(all(row[3] == threading.get_ident() for row in rows))
        finally:
            marker.reset(token)
            release.set()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
            await executor.close()

    async def test_zero_reserve_preserves_full_upload_capacity_and_metrics(self):
        metrics = RuntimePerformance()
        executor = StorageWorkExecutor(2, metrics=metrics)
        release = threading.Event()
        entered = [threading.Event(), threading.Event()]

        def upload(index):
            entered[index].set()
            release.wait(3)

        tasks = [asyncio.create_task(executor.run_storage("upload", upload, i)) for i in range(2)]
        try:
            await asyncio.gather(*(wait_thread_event(event) for event in entered))
            self.assertEqual(executor.queue_snapshot()["inflight_jobs"], 2)
            self.assertFalse(executor._upload_tasks)
            release.set()
            await asyncio.gather(*tasks)
            self.assertEqual(metrics.snapshot()["operations"]["storage_upload_run"]["count"], 2)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await executor.close()

    async def test_invalid_reserve_fails_before_any_resource_pool_is_created(self):
        for reserve in (-1, 2, 3, True, 0.5):
            with self.subTest(reserve=reserve), patch("core.execution.AsyncWorkExecutor") as factory:
                with self.assertRaises(ValueError):
                    ExecutionResources(image_workers=1, storage_workers=2, search_workers=1,
                                       write_workers=1, storage_download_reserved=reserve)
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
