"""直接线程 Future 保留三路并行、上下文、异常与取消语义。"""

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import partial
import gc
import threading
import unittest
from unittest.mock import patch

from core.performance import RuntimePerformance
from core.thread_timing import gather_timed_threads


class CaptureMetrics:
    def __init__(self):
        self.rows = []
        self.ready = asyncio.Event()
        self.expected_rows = 0

    def record(self, operation, duration_ms, *, failed=False):
        self.rows.append((operation, duration_ms, failed, threading.get_ident()))
        if len(self.rows) >= self.expected_rows:
            self.ready.set()

    async def wait_for_rows(self, count):
        self.expected_rows = count
        self.ready.clear()
        if len(self.rows) < count:
            await asyncio.wait_for(self.ready.wait(), 3)

    def by_operation(self):
        grouped = defaultdict(list)
        for name, duration, failed, thread_id in self.rows:
            grouped[name].append((duration, failed, thread_id))
        return grouped


class DirectThreadGatherTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.metrics = CaptureMetrics()
        patcher = patch("core.thread_timing.runtime_performance", self.metrics)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_parallel_order_kwargs_context_and_callback_critical_path(self):
        barrier = threading.Barrier(3, timeout=1)
        value_context = ContextVar("gather_context", default="missing")
        token = value_context.set("private-frame")
        self.addCleanup(value_context.reset, token)
        loop_thread = threading.get_ident()

        def worker(value, *, timing):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            barrier.wait()
            observed = value_context.get()
            value_context.set(value)
            return value, timing, observed

        result = await gather_timed_threads(
            "sbs_search",
            *(partial(worker, value, timing="target-kwarg") for value in (3, 1, 2)),
        )
        await self.metrics.wait_for_rows(19)
        self.assertEqual(result, [
            (3, "target-kwarg", "private-frame"),
            (1, "target-kwarg", "private-frame"),
            (2, "target-kwarg", "private-frame"),
        ])
        self.assertEqual(value_context.get(), "private-frame")
        operations = self.metrics.by_operation()
        for suffix in ("thread_queue", "thread_run", "loop_resume", "future_callback"):
            self.assertEqual(len(operations[f"sbs_search_{suffix}"]), 3)
        phases = (
            "future_critical_dispatch", "future_critical_thread_queue",
            "future_critical_thread_run", "future_critical_callback", "future_group_join",
        )
        for suffix in ("group_wait", "future_group_submit_span", *phases):
            self.assertEqual(len(operations[f"sbs_search_{suffix}"]), 1)
        self.assertAlmostEqual(
            sum(operations[f"sbs_search_{suffix}"][0][0] for suffix in phases),
            operations["sbs_search_group_wait"][0][0], places=7,
        )
        for name, duration, failed, thread_id in self.metrics.rows:
            self.assertNotIn("private", name)
            self.assertGreaterEqual(duration, 0)
            self.assertFalse(failed)
            self.assertEqual(thread_id, loop_thread)

    async def test_error_is_original_and_does_not_wait_for_other_branch(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        failure = ValueError("original failure")

        def fail():
            self.assertTrue(entered.wait(timeout=1))
            raise failure

        def slow():
            entered.set()
            release.wait(timeout=1)
            return 42

        with self.assertRaises(ValueError) as caught:
            await gather_timed_threads("sbs_search", fail, slow)
        self.assertIs(caught.exception, failure)
        self.assertFalse(release.is_set())
        release.set()
        await self.metrics.wait_for_rows(9)
        operations = self.metrics.by_operation()
        self.assertEqual(len(operations["sbs_search_future_callback"]), 2)
        self.assertEqual(
            sorted(row[1] for row in operations["sbs_search_future_callback"]),
            [False, True],
        )
        self.assertTrue(operations["sbs_search_group_wait"][0][1])
        self.assertFalse(any("critical" in name for name in operations))
        self.assertNotIn("sbs_search_future_group_join", operations)

    async def test_cancel_running_worker_preserves_background_completion_metrics(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def worker():
            loop.call_soon_threadsafe(entered.set)
            release.wait(timeout=1)
            return 42

        task = asyncio.create_task(gather_timed_threads("sbs_search", worker))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.metrics.rows), 1)
        release.set()
        await self.metrics.wait_for_rows(4)
        operations = self.metrics.by_operation()
        self.assertEqual(len(operations["sbs_search_thread_run"]), 1)
        self.assertFalse(operations["sbs_search_thread_run"][0][1])
        self.assertTrue(operations["sbs_search_group_wait"][0][1])
        self.assertNotIn("sbs_search_future_callback", operations)
        self.assertFalse(any("critical" in name for name in operations))

    async def test_cancel_queued_futures_prevents_worker_execution(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        entered = asyncio.Event()
        release = threading.Event()
        called = []
        self.addCleanup(release.set)

        def blocker():
            loop.call_soon_threadsafe(entered.set)
            release.wait(timeout=1)

        blocker_future = loop.run_in_executor(None, blocker)
        await asyncio.wait_for(entered.wait(), 1)
        task = asyncio.create_task(gather_timed_threads(
            "sbs_search", *(partial(called.append, value) for value in (3, 1, 2)),
        ))
        await asyncio.sleep(0)  # helper 同步提交三路后首次 await，线程仍被 blocker 占用。
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()
        await blocker_future
        await loop.run_in_executor(None, lambda: None)  # 排在已取消三路之后的队尾哨兵。
        self.assertEqual(called, [])
        self.assertEqual(len(self.metrics.rows), 1)
        operations = self.metrics.by_operation()
        self.assertTrue(operations["sbs_search_group_wait"][0][1])
        self.assertNotIn("sbs_search_thread_run", operations)
        self.assertNotIn("sbs_search_future_callback", operations)

    async def test_future_callback_is_independent_of_thread_statistics_callback(self):
        loop = asyncio.get_running_loop()
        original_schedule = loop.call_soon_threadsafe
        delayed = []

        def schedule(callback, *args, context=None):
            if getattr(callback, "__name__", "") == "record_thread":
                delayed.append((callback, args))
                return None
            return original_schedule(callback, *args, context=context)

        with patch.object(loop, "call_soon_threadsafe", side_effect=schedule):
            self.assertEqual(await gather_timed_threads("sbs_search", lambda: 42), [42])
        operations = self.metrics.by_operation()
        self.assertNotIn("sbs_search_loop_resume", operations)
        self.assertEqual(len(operations["sbs_search_future_callback"]), 1)
        self.assertEqual(len(delayed), 1)
        for callback, args in delayed:
            original_schedule(callback, *args)
        await self.metrics.wait_for_rows(11)
        self.assertEqual(len(self.metrics.by_operation()["sbs_search_loop_resume"]), 1)

    async def test_partial_submission_error_preserves_error_and_observes_running_failure(self):
        loop = asyncio.get_running_loop()
        original_submit = loop.run_in_executor
        old_handler = loop.get_exception_handler()
        unhandled = []
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))
        self.addCleanup(loop.set_exception_handler, old_handler)
        release = threading.Event()
        self.addCleanup(release.set)
        submitted = False
        submission_failure = RuntimeError("executor rejected submission")
        worker_failure = ValueError("already submitted worker failed")

        def worker():
            release.wait(timeout=1)
            raise worker_failure

        def submit(*args, **kwargs):
            nonlocal submitted
            if submitted:
                raise submission_failure
            submitted = True
            return original_submit(*args, **kwargs)

        with patch.object(loop, "run_in_executor", side_effect=submit):
            with self.assertRaises(RuntimeError) as caught:
                await gather_timed_threads("sbs_search", worker, lambda: 42)
        self.assertIs(caught.exception, submission_failure)
        release.set()
        await self.metrics.wait_for_rows(5)
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
        self.assertEqual(unhandled, [])
        operations = self.metrics.by_operation()
        self.assertTrue(operations["sbs_search_group_wait"][0][1])
        self.assertTrue(operations["sbs_search_future_callback"][0][1])
        self.assertFalse(any("critical" in name for name in operations))

    async def test_runtime_operation_names_are_static_and_do_not_expose_identifiers(self):
        metrics = RuntimePerformance()
        with patch("core.thread_timing.runtime_performance", metrics):
            self.assertEqual(
                await gather_timed_threads("sbs_search", lambda: "private-reid"),
                ["private-reid"],
            )
            await asyncio.sleep(0)
        snapshot = metrics.snapshot()
        for suffix in (
            "future_callback", "group_wait", "future_group_submit_span",
            "future_critical_dispatch", "future_critical_thread_queue",
            "future_critical_thread_run", "future_critical_callback", "future_group_join",
        ):
            self.assertEqual(snapshot["operations"][f"sbs_search_{suffix}"]["count"], 1)
        self.assertNotIn("private-reid", str(snapshot))


if __name__ == "__main__":
    unittest.main()
