import asyncio
from contextvars import ContextVar
import threading
import unittest
from unittest.mock import patch

from core.execution import AsyncWorkExecutor, ExecutionResources, run_blocking
from core.performance import RuntimePerformance


async def wait_thread_event(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 2)


class ExecutionResourcesTests(unittest.IsolatedAsyncioTestCase):
    def resources(self):
        return ExecutionResources(
            image_workers=1, storage_workers=1, search_workers=1, write_workers=1,
        )

    async def test_resource_domains_isolate_blocking_work_and_avoid_default_executor(self):
        resources = self.resources()
        entered, release = threading.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def block_storage():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test failed to release storage")

        task = asyncio.create_task(resources.storage_io.run(block_storage))
        try:
            await wait_thread_event(entered)
            with patch.object(loop, "run_in_executor", side_effect=AssertionError("default executor used")):
                image, search, write = await asyncio.wait_for(asyncio.gather(
                    resources.image_cpu.run(lambda: "image"),
                    resources.vector_search.run(lambda: "search"),
                    resources.vector_write.run(lambda: "write"),
                ), 1)
            self.assertEqual((image, search, write), ("image", "search", "write"))
            self.assertEqual(resources.queue_snapshot()["storage_io"]["inflight_jobs"], 1)
        finally:
            release.set()
            await task
            await resources.close()

    async def test_cancelled_caller_keeps_real_capacity_and_close_waits(self):
        resources = self.resources()
        entered, release = threading.Event(), threading.Event()
        ran_waiter = False

        def block():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test failed to release worker")

        def waiter():
            nonlocal ran_waiter
            ran_waiter = True

        running = asyncio.create_task(resources.image_cpu.run(block))
        waiting = closing = None
        try:
            await wait_thread_event(entered)
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            waiting = asyncio.create_task(resources.image_cpu.run(waiter))
            await asyncio.sleep(0)
            self.assertEqual(resources.image_cpu.queue_snapshot()["queued_jobs"], 1)
            self.assertEqual(resources.image_cpu.queue_snapshot()["inflight_jobs"], 1)
            closing = asyncio.create_task(resources.close())
            for _ in range(3):
                await asyncio.sleep(0)
            closing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            self.assertFalse(resources._close_task.done())
            release.set()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await waiting
            await resources.close()
            self.assertFalse(ran_waiter)
            self.assertTrue(all(item["closed"] for item in resources.queue_snapshot().values()))
        finally:
            release.set()
            await asyncio.gather(*(task for task in (running, waiting, closing) if task), return_exceptions=True)
            await resources.close()

    async def test_context_keywords_and_original_errors_cross_threads(self):
        executor = AsyncWorkExecutor(1, name="test")
        marker = ContextVar("marker", default="absent")
        token = marker.set("request-context")
        failure = ValueError("original failure")

        def operation(value, *, factor):
            return value * factor, marker.get(), threading.get_ident()

        def fail():
            raise failure

        try:
            value, context, thread_id = await executor.run(operation, 3, factor=7)
            self.assertEqual((value, context), (21, "request-context"))
            self.assertNotEqual(thread_id, threading.get_ident())
            with self.assertRaises(ValueError) as raised:
                await executor.run(fail)
            self.assertIs(raised.exception, failure)
            self.assertEqual(await executor.run(lambda: 1), 1)
        finally:
            marker.reset(token)
            await executor.close()

    async def test_group_uses_injected_executor_and_preserves_timing_and_order(self):
        executor = AsyncWorkExecutor(2, name="test")
        metrics = RuntimePerformance()
        loop = asyncio.get_running_loop()
        try:
            with patch("core.thread_timing.runtime_performance", metrics), patch.object(
                loop, "run_in_executor", side_effect=AssertionError("default executor used"),
            ):
                values = await executor.run_group("sbs_search", lambda: "upper", lambda: "lower")
            self.assertEqual(values, ["upper", "lower"])
            operations = metrics.snapshot()["operations"]
            self.assertEqual(operations["sbs_search_thread_run"]["count"], 2)
            self.assertEqual(operations["sbs_search_thread_queue"]["count"], 2)
            self.assertEqual(operations["sbs_search_group_wait"]["count"], 1)
            self.assertEqual(operations["sbs_search_future_group_join"]["count"], 1)
        finally:
            await executor.close()

    async def test_optional_runner_compatibility_preserves_result(self):
        executor = AsyncWorkExecutor(1, name="test")
        try:
            self.assertEqual(await run_blocking(executor, pow, 2, exp=3), 8)
            self.assertEqual(await run_blocking(None, pow, 3, exp=2), 9)
        finally:
            await executor.close()

    async def test_close_does_not_require_default_executor(self):
        resources = self.resources()
        await resources.image_cpu.run(lambda: 1)
        with patch.object(
            asyncio.get_running_loop(), "run_in_executor",
            side_effect=AssertionError("default executor used during close"),
        ):
            await resources.close()
        self.assertTrue(resources.image_cpu._pool._executor._shutdown)

    async def test_timed_operation_records_domain_and_step_on_loop(self):
        rows = []

        class Metrics:
            def record(self, name, duration_ms, *, failed=False):
                rows.append((name, duration_ms, failed, threading.get_ident()))

        executor = AsyncWorkExecutor(1, name="image_cpu", metrics=Metrics())
        try:
            result = await executor.run_timed("image_decode", lambda: "decoded")
            self.assertEqual(result, "decoded")
            self.assertEqual([row[0] for row in rows], [
                "image_cpu_run", "image_decode", "image_cpu_queue_wait", "image_decode_queue_wait",
            ])
            self.assertTrue(all(row[3] == threading.get_ident() for row in rows))
            self.assertEqual(rows[0][1], rows[1][1])
            self.assertEqual(rows[2][1], rows[3][1])
        finally:
            await executor.close()

    async def test_invalid_budget_is_rejected_before_creating_resources(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                ExecutionResources(
                    image_workers=1, storage_workers=1, search_workers=1,
                    write_workers=workers,
                )


if __name__ == "__main__":
    unittest.main()
