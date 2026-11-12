import asyncio
import threading
import unittest
from unittest.mock import patch

from core.bounded_cpu_pool import BoundedCpuPool
from core.performance import RuntimePerformance


def make_pool(workers=1):
    return BoundedCpuPool(
        workers,
        thread_name_prefix="test-bounded-cpu",
        operation_name="person_crop_prepare",
        queue_operation_name="person_crop_prepare_queue_wait",
    )


class BoundedCpuPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_arbitrary_result_none_and_original_exception(self):
        metrics = RuntimePerformance()
        pool = make_pool()
        value = object()
        failure = ValueError("operation failed")

        def fail():
            raise failure

        try:
            self.assertIs(await pool.submit(lambda item: item, value, metrics=metrics), value)
            self.assertIsNone(await pool.submit(lambda: None, metrics=metrics))
            with self.assertRaises(ValueError) as raised:
                await pool.submit(fail, metrics=metrics)
            self.assertIs(raised.exception, failure)
            fields = metrics.snapshot()["operations"]
            self.assertEqual(fields["person_crop_prepare"]["count"], 3)
            self.assertEqual(fields["person_crop_prepare"]["errors"], 1)
            self.assertEqual(fields["person_crop_prepare_queue_wait"]["count"], 3)
        finally:
            await pool.close()

    async def test_metric_failure_preserves_result_and_releases_capacity(self):
        class BrokenMetrics:
            def record(self, *args, **kwargs):
                raise RuntimeError("metrics unavailable")

        pool = make_pool()
        loop = asyncio.get_running_loop()
        reported = []
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: reported.append(context))
        try:
            self.assertEqual(await pool.submit(lambda: 42, metrics=BrokenMetrics()), 42)
            self.assertEqual(await asyncio.wait_for(pool.submit(lambda: 43), 1), 43)
            self.assertEqual(len(reported), 1)
            self.assertEqual(str(reported[0]["exception"]), "metrics unavailable")
        finally:
            loop.set_exception_handler(original_handler)
            await pool.close()

    async def test_waiting_submission_is_rejected_on_close_without_running(self):
        entered, release = threading.Event(), threading.Event()
        ran_waiter = False

        def blocking():
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release worker")

        def waiting():
            nonlocal ran_waiter
            ran_waiter = True

        pool = make_pool()
        active = asyncio.create_task(pool.submit(blocking))
        waiter = closer = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            waiter = asyncio.create_task(pool.submit(waiting))
            await asyncio.sleep(0)
            closer = asyncio.create_task(pool.close())
            await asyncio.sleep(0)
            release.set()
            await active
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await waiter
            await closer
            self.assertFalse(ran_waiter)
        finally:
            release.set()
            await asyncio.gather(
                *(task for task in (active, waiter, closer) if task), return_exceptions=True
            )
            await pool.close()

    async def test_executor_submission_failure_does_not_lose_permit(self):
        pool = make_pool()
        try:
            with patch.object(pool._executor, "submit", side_effect=RuntimeError("submit failed")):
                with self.assertRaisesRegex(RuntimeError, "submit failed"):
                    await pool.submit(lambda: 1)
            self.assertEqual(await asyncio.wait_for(pool.submit(lambda: 2), 1), 2)
        finally:
            await pool.close()

    async def test_pool_rejects_use_from_another_loop(self):
        pool = make_pool()

        def other_loop():
            return asyncio.run(pool.submit(lambda: 1))

        try:
            with self.assertRaisesRegex(RuntimeError, "different event loop"):
                await asyncio.to_thread(other_loop)
        finally:
            await pool.close()

    async def test_drain_on_cancel_holds_capacity_until_real_operation_finishes(self):
        entered, release = threading.Event(), threading.Event()
        metrics = RuntimePerformance()

        def blocking():
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release worker")
            return 42

        pool = make_pool()
        active = asyncio.create_task(
            pool.submit(blocking, metrics=metrics, drain_on_cancel=True)
        )
        waiter = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            for _ in range(2):
                active.cancel()
                await asyncio.sleep(0)
                self.assertFalse(active.done())
            waiter = asyncio.create_task(pool.submit(lambda: 43))
            await asyncio.sleep(0)
            self.assertFalse(waiter.done())
            self.assertEqual(len(pool._pending), 1)

            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await active
            self.assertEqual(await asyncio.wait_for(waiter, 1), 43)
            operations = metrics.snapshot()["operations"]
            self.assertEqual(operations["person_crop_prepare"]["count"], 1)
            self.assertEqual(operations["person_crop_prepare"]["errors"], 0)
        finally:
            release.set()
            await asyncio.gather(
                *(task for task in (active, waiter) if task), return_exceptions=True
            )
            await pool.close()

    async def test_drain_cancelled_before_permit_does_not_submit_operation(self):
        entered, release = threading.Event(), threading.Event()
        ran_waiter = False

        def blocking():
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release worker")

        def waiting():
            nonlocal ran_waiter
            ran_waiter = True

        pool = make_pool()
        active = asyncio.create_task(pool.submit(blocking))
        waiter = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            waiter = asyncio.create_task(pool.submit(waiting, drain_on_cancel=True))
            await asyncio.sleep(0)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            release.set()
            await active
            self.assertFalse(ran_waiter)
        finally:
            release.set()
            await asyncio.gather(
                *(task for task in (active, waiter) if task), return_exceptions=True
            )
            await pool.close()


if __name__ == "__main__":
    unittest.main()
