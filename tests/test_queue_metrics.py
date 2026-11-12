import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from core.queue_metrics import PipelineMetrics
from core.bounded_cpu_pool import BoundedCpuPool


class QueueMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_waiters_active_and_interval_counts(self):
        now = [1.0]
        metrics = PipelineMetrics(clock=lambda: now[0], wall_clock=lambda: 1000.0)
        lock = asyncio.Lock()
        started = asyncio.Event()
        release = asyncio.Event()

        async def work():
            async with metrics.slot("identity", lock):
                started.set()
                await release.wait()

        first = asyncio.create_task(work())
        await started.wait()
        second = asyncio.create_task(work())
        await asyncio.sleep(0)
        now[0] = 3.0
        sample = metrics.snapshot()["stages"]["identity"]
        self.assertEqual((sample["waiting"], sample["active"]), (1, 1))
        self.assertEqual(sample["oldest_wait_ms"], 2000)
        second.cancel()
        await asyncio.gather(second, return_exceptions=True)
        self.assertTrue(lock.locked())
        self.assertEqual(metrics.snapshot()["stages"]["identity"]["waiting"], 0)
        release.set()
        await first
        now[0] = 5.0
        ended = metrics.snapshot()["stages"]["identity"]
        self.assertEqual((ended["active"], ended["completed"]), (0, 1))
        self.assertEqual(ended["completed_per_second"], 0.5)
        self.assertEqual(metrics.snapshot()["stages"]["identity"]["completed"], 0)
        self.assertEqual(sample["waiting"], 1)  # 已生成快照不随后续状态变动。

    async def test_cancellation_and_failure_release_active_slot(self):
        metrics = PipelineMetrics()
        lock = asyncio.Lock()
        with self.assertRaisesRegex(ValueError, "failure"):
            async with metrics.slot("identity", lock):
                raise ValueError("failure")
        with self.assertRaises(asyncio.CancelledError):
            async with metrics.slot("identity", lock):
                raise asyncio.CancelledError
        stage = metrics.snapshot()["stages"]["identity"]
        self.assertEqual((stage["active"], stage["waiting"]), (0, 0))
        self.assertEqual((stage["failed"], stage["cancelled"], stage["completed"]), (1, 1, 0))
        self.assertFalse(lock.locked())

    async def test_ordered_wait_ticket_is_idempotent_and_not_business_completion(self):
        metrics = PipelineMetrics()
        ticket = metrics.begin_wait("ready")
        self.assertEqual(metrics.snapshot()["stages"]["ready"]["waiting"], 1)
        ticket.close()
        ticket.close()
        sample = metrics.snapshot()["stages"]["ready"]
        self.assertEqual((sample["waiting"], sample["waits_finished"], sample["completed"]), (0, 1, 0))

    async def test_provider_failure_is_unavailable_not_zero_and_does_not_leak_message(self):
        metrics = PipelineMetrics()

        def failure():
            raise RuntimeError("sensitive input")

        metrics.register("good", lambda: {"queued": 2})
        metrics.register("bad", failure)
        metrics.register("nan", lambda: {"value": float("nan")})
        data = metrics.snapshot()
        self.assertEqual(data["queues"]["good"], {"queued": 2})
        self.assertEqual(data["queues"]["bad"], {"available": False, "error_type": "RuntimeError"})
        self.assertFalse(data["queues"]["nan"]["available"])
        self.assertNotIn("sensitive input", json.dumps(data, allow_nan=False))
        metrics.unregister("bad")
        self.assertNotIn("bad", metrics.snapshot()["queues"])

    async def test_counter_rate_uses_sample_window_and_keeps_total(self):
        now = [0.0]
        metrics = PipelineMetrics(clock=lambda: now[0])
        metrics.increment("image_pipeline_completed", 10)
        now[0] = 2.0
        first = metrics.snapshot()["counters"]["image_pipeline_completed"]
        self.assertEqual(first, {"count": 10, "total": 10, "per_second": 5.0})
        now[0] = 3.0
        self.assertEqual(metrics.snapshot()["counters"]["image_pipeline_completed"], {
            "count": 0, "total": 10, "per_second": 0.0,
        })

    async def test_jsonl_file_has_initial_periodic_final_records_and_closes(self):
        metrics = PipelineMetrics()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stats.jsonl"
            metrics.register("input", lambda: {"queued_messages": 7})
            metrics.start(path=str(path), interval_seconds=0.01)
            task = metrics._task
            try:
                metrics.increment("image_pipeline_completed")
                for _ in range(100):
                    await asyncio.sleep(0.005)
                    if len(path.read_text().splitlines()) >= 2:
                        break
            finally:
                await metrics.stop()
            await metrics.stop()
            self.assertTrue(task.done())
            self.assertIsNone(metrics._sink_id)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertGreaterEqual(len(rows), 3)
            self.assertFalse(rows[0]["final"])
            self.assertTrue(rows[-1]["final"])
            self.assertTrue(all(row["queues"]["input"]["queued_messages"] == 7 for row in rows))
            self.assertEqual(sum(row["counters"].get("image_pipeline_completed", {}).get("count", 0) for row in rows), 1)

    async def test_invalid_interval_does_not_open_sink(self):
        metrics = PipelineMetrics()
        for interval in (0, -1, float("nan"), float("inf")):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                metrics.start(path="unused.jsonl", interval_seconds=interval)
        self.assertIsNone(metrics._sink_id)

    async def test_start_emission_failure_closes_sink_and_allows_restart(self):
        from loguru import logger

        metrics = PipelineMetrics()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stats.jsonl"
            with patch.object(metrics, "_emit", side_effect=ValueError("private data")), patch.object(
                logger, "remove", wraps=logger.remove
            ) as remove:
                with self.assertRaisesRegex(ValueError, "private data"):
                    metrics.start(path=str(path))
                remove.assert_called_once()
            self.assertIsNone(metrics._sink_id)
            self.assertIsNone(metrics._task)
            self.assertIsNone(metrics._logger)
            metrics.start(path=str(path))
            await metrics.stop()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[-1]["final"])

    async def test_periodic_emission_failure_recovers_without_polluting_jsonl(self):
        from loguru import logger

        metrics = PipelineMetrics()
        original_emit = metrics._emit
        calls = 0
        recovered = asyncio.Event()
        business_records = []

        def fail_one_emission(*, final=False):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("private input must not appear in logs")
            original_emit(final=final)
            if calls >= 3:
                recovered.set()

        sink_id = logger.add(
            lambda message: business_records.append(message.record),
            filter=lambda record: "pipeline_stats_session" not in record["extra"],
        )
        try:
            with TemporaryDirectory() as directory:
                path = Path(directory) / "stats.jsonl"
                with patch.object(metrics, "_emit", side_effect=fail_one_emission):
                    metrics.start(path=str(path), interval_seconds=0.005)
                    try:
                        await asyncio.wait_for(recovered.wait(), timeout=1)
                        self.assertFalse(metrics._task.done())
                    finally:
                        await metrics.stop()
                lines = path.read_text().splitlines()
                rows = [json.loads(line) for line in lines]
                self.assertGreaterEqual(len(rows), 3)
                self.assertTrue(rows[-1]["final"])
                self.assertNotIn("private input", "\n".join(lines))
            warnings = [record for record in business_records if record["level"].name == "WARNING"]
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]["message"], "Pipeline statistics emission failed: RuntimeError")
        finally:
            logger.remove(sink_id)

    async def test_business_logger_failure_does_not_kill_periodic_reporting(self):
        from loguru import logger

        metrics = PipelineMetrics()
        recovered = asyncio.Event()
        original_emit = metrics._emit
        calls = 0

        def fail_one_emission(*, final=False):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("broken snapshot")
            original_emit(final=final)
            if calls >= 3:
                recovered.set()

        with TemporaryDirectory() as directory:
            with patch.object(metrics, "_emit", side_effect=fail_one_emission), patch.object(
                logger, "warning", side_effect=RuntimeError("logging unavailable")
            ):
                metrics.start(path=str(Path(directory) / "stats.jsonl"), interval_seconds=0.005)
                try:
                    await asyncio.wait_for(recovered.wait(), timeout=1)
                    self.assertFalse(metrics._task.done())
                finally:
                    await metrics.stop()

    async def test_cancellation_while_stopping_reporter_still_closes_sink(self):
        metrics = PipelineMetrics()
        started = asyncio.Event()
        reporter_cancelled = asyncio.Event()

        async def delayed_report(interval_seconds):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                reporter_cancelled.set()
                await asyncio.Event().wait()

        with TemporaryDirectory() as directory, patch.object(metrics, "_report", side_effect=delayed_report):
            metrics.start(path=str(Path(directory) / "stats.jsonl"))
            reporter = metrics._task
            await started.wait()
            stopping = asyncio.create_task(metrics.stop())
            await asyncio.wait_for(reporter_cancelled.wait(), timeout=1)
            stopping.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await stopping
            self.assertTrue(reporter.done())
            self.assertIsNone(metrics._sink_id)
            self.assertIsNone(metrics._logger)
            await metrics.stop()

    async def test_final_emission_failure_still_closes_sink(self):
        from loguru import logger

        metrics = PipelineMetrics()
        with TemporaryDirectory() as directory:
            metrics.start(path=str(Path(directory) / "stats.jsonl"))
            with patch.object(metrics, "_emit", side_effect=ValueError("private")), patch.object(
                logger, "warning"
            ) as warning:
                await metrics.stop()
                warning.assert_called_once_with("Pipeline statistics emission failed: {}", "ValueError")
            self.assertIsNone(metrics._sink_id)
            self.assertIsNone(metrics._logger)

    async def test_cpu_snapshot_retains_real_work_when_caller_is_cancelled(self):
        pool = BoundedCpuPool(1, thread_name_prefix="queue-test", operation_name="test", queue_operation_name="test-wait")
        thread_release = threading.Event()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def work():
            loop.call_soon_threadsafe(started.set)
            thread_release.wait(5)

        first = asyncio.create_task(pool.submit(work))
        second = None
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            second = asyncio.create_task(pool.submit(lambda: None))
            await asyncio.sleep(0)
            self.assertEqual(pool.queue_snapshot(), {
                "closed": False, "queued_jobs": 1, "inflight_jobs": 1, "capacity_jobs": 1,
            })
            first.cancel()
            second.cancel()
            await asyncio.gather(first, second, return_exceptions=True)
            sample = pool.queue_snapshot()
            self.assertEqual((sample["queued_jobs"], sample["inflight_jobs"]), (0, 1))
        finally:
            thread_release.set()
            await pool.close()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        self.assertEqual(pool.queue_snapshot()["inflight_jobs"], 0)


if __name__ == "__main__":
    unittest.main()
