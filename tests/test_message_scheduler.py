import asyncio
import unittest

from core.message_scheduler import MessageScheduler
from core.queue_metrics import PipelineMetrics


class MessageSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_admission_returns_before_completion_but_queue_join_does_not(self):
        release = asyncio.Event()
        started = asyncio.Event()
        queue = asyncio.Queue()
        await queue.put('message')
        async def handler(message):
            self.assertEqual(message, 'message')
            started.set()
            await release.wait()
        scheduler = MessageScheduler(handler, capacity=1, on_error=self.fail)
        await scheduler.submit(await queue.get(), on_done=queue.task_done)
        await started.wait()
        joined = asyncio.create_task(queue.join())
        await asyncio.sleep(0)
        self.assertFalse(joined.done())
        self.assertEqual(scheduler.queue_snapshot()['inflight_messages'], 1)
        release.set()
        await joined
        await scheduler.close()
        self.assertEqual(scheduler.queue_snapshot()['inflight_messages'], 0)

    async def test_capacity_bounds_tasks_and_cancelled_waiter_never_transfers_ownership(self):
        release = asyncio.Event()
        seen, done = [], []
        async def handler(value):
            seen.append(value)
            await release.wait()
        scheduler = MessageScheduler(handler, capacity=1, on_error=self.fail)
        await scheduler.submit(1, on_done=lambda: done.append(1))
        waiter = asyncio.create_task(scheduler.submit(2, on_done=lambda: done.append(2)))
        await asyncio.sleep(0)
        self.assertEqual(seen, [1])
        self.assertEqual(scheduler.queue_snapshot()['admission_waiters'], 1)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        release.set()
        await scheduler.close()
        self.assertEqual(done, [1])
        self.assertEqual(scheduler.queue_snapshot()['admission_waiters'], 0)

    async def test_failed_handler_releases_slot_and_reports_error_once(self):
        errors, done = [], []
        async def handler(value):
            if value == 1:
                raise ValueError('failed')
        scheduler = MessageScheduler(handler, capacity=1, on_error=errors.append)
        await scheduler.submit(1, on_done=lambda: done.append(1))
        await scheduler.submit(2, on_done=lambda: done.append(2))
        await scheduler.close()
        self.assertEqual(done, [1, 2])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)

    async def test_close_cancellation_does_not_cancel_owned_business(self):
        release = asyncio.Event()
        done, cancelled = [], []
        async def handler(value):
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(value)
                raise
        scheduler = MessageScheduler(handler, capacity=1, on_error=self.fail)
        await scheduler.submit(1, on_done=lambda: done.append(1))
        closing = asyncio.create_task(scheduler.close())
        await asyncio.sleep(0)
        closing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        self.assertEqual(cancelled, [])
        self.assertEqual(done, [])
        release.set()
        await scheduler.close()
        self.assertEqual(done, [1])

    async def test_close_rejects_waiting_admission_after_capacity_releases(self):
        release = asyncio.Event()
        done = []
        async def handler(value):
            await release.wait()
        scheduler = MessageScheduler(handler, capacity=1, on_error=self.fail)
        await scheduler.submit(1, on_done=lambda: done.append(1))
        waiter = asyncio.create_task(scheduler.submit(2, on_done=lambda: done.append(2)))
        await asyncio.sleep(0)
        closing = asyncio.create_task(scheduler.close())
        await asyncio.sleep(0)
        release.set()
        await closing
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            await waiter
        self.assertEqual(done, [1])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            await scheduler.submit(3, on_done=lambda: done.append(3))

    async def test_cancel_before_task_starts_still_accounts_done_once(self):
        async def handler(value):
            self.fail('cancelled task must not run')
        done = []
        scheduler = MessageScheduler(handler, capacity=1, on_error=self.fail)
        await scheduler.submit(None, on_done=lambda: done.append(True))
        next(iter(scheduler._tasks)).cancel()
        await scheduler.close()
        self.assertEqual(done, [True])
        self.assertEqual(scheduler.queue_snapshot()['inflight_messages'], 0)

    async def test_metrics_follow_business_activity_not_submit_return(self):
        metrics = PipelineMetrics()
        release = asyncio.Event()
        started = asyncio.Event()
        async def handler(value):
            started.set()
            await release.wait()
        scheduler = MessageScheduler(handler, capacity=1, on_error=self.fail, metrics=metrics)
        await scheduler.submit(None, on_done=lambda: None)
        await started.wait()
        self.assertEqual(metrics.snapshot()['stages']['message_processing']['active'], 1)
        release.set()
        await scheduler.close()
        state = metrics.snapshot()['stages']['message_processing']
        self.assertEqual(state['active'], 0)
        self.assertEqual(state['completed_total'], 1)

    def test_capacity_validation(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MessageScheduler(None, capacity=value, on_error=None)


if __name__ == '__main__':
    unittest.main()
