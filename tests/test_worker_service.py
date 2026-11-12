"""输入、调度与完成回调所有权测试；所有消费者与业务处理均为内存 fake。"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import unittest

from core.admission import AdmissionController
from core.message_scheduler import MessageScheduler
from core.queue_metrics import PipelineMetrics
from core.worker_service import WorkerService, consume_messages, dispatch_messages


class FakeLogger:
    def __init__(self):
        self.errors = []

    def info(self, *args):
        pass

    def error(self, message, *args):
        self.errors.append(message.format(*args))


class FakeConsumer:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.started = asyncio.Event()
        self.committed = asyncio.Event()
        self.after_batch = None
        self.stop_calls = 0
        self.commits = []
        self.batches = 0
        self.start_failure = None

    async def start(self):
        self.started.set()
        if self.start_failure is not None:
            raise self.start_failure

    async def getmany(self, **kwargs):
        self.batches += 1
        if self.messages:
            batch, self.messages = self.messages, []
            return {"partition": batch}
        if self.after_batch is not None:
            await self.after_batch()
        await asyncio.Event().wait()

    async def commit(self, offsets):
        self.commits.append(offsets)
        self.committed.set()

    async def stop(self):
        self.stop_calls += 1


class OpenAdmission:
    async def wait_for_capacity(self):
        pass


def message(index, value=None):
    return SimpleNamespace(offset=index, value=json.dumps({"index": index}).encode() if value is None else value)


class WorkerServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.logger = FakeLogger()
        self.metrics = PipelineMetrics()

    async def eventually(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0)

    def service(self, consumer, handler, *, admission=None, **kwargs):
        return WorkerService(
            consumer=consumer, handler=handler, admission=admission or OpenAdmission(),
            worker_count=kwargs.pop("worker_count", 1),
            message_capacity=kwargs.pop("message_capacity", 1),
            parse_context=lambda payload: payload["index"],
            logger=self.logger, metrics=self.metrics, **kwargs,
        )

    async def assert_drained(self, service):
        await asyncio.wait_for(service.queue.join(), 1)
        self.assertEqual(service.queue.qsize(), 0)
        self.assertEqual(service.scheduler.queue_snapshot()["inflight_messages"], 0)
        self.assertTrue(service.scheduler.queue_snapshot()["closed"])
        self.assertFalse(service._running)
        self.assertEqual(self.metrics._providers, {})

    async def test_normal_stop_drains_queued_and_owned_messages(self):
        entered, release = asyncio.Event(), asyncio.Event()
        completed = []

        async def handler(item):
            entered.set()
            await release.wait()
            completed.append(item.offset)

        consumer = FakeConsumer([message(0), message(1)])
        service = self.service(consumer, handler)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(stop))
        try:
            await entered.wait()
            await consumer.committed.wait()
            stop.set()
            await asyncio.sleep(0)
            self.assertFalse(running.done())
            release.set()
            await asyncio.wait_for(running, 1)
            self.assertEqual(completed, [0, 1])
            self.assertEqual(consumer.commits, [{"partition": 2}])
            self.assertEqual(consumer.stop_calls, 1)
            await self.assert_drained(service)
        finally:
            release.set()
            stop.set()
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    async def test_business_error_completes_ownership_and_other_messages_continue(self):
        completed = asyncio.Event()

        async def handler(item):
            if item.offset == 0:
                raise ValueError("business failure")
            completed.set()

        service = self.service(FakeConsumer([message(0), message(1)]), handler)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(stop))
        await asyncio.wait_for(completed.wait(), 1)
        stop.set()
        await asyncio.wait_for(running, 1)
        self.assertTrue(any("Message task failed: ValueError" in error for error in self.logger.errors))
        await self.assert_drained(service)

    async def test_cancelled_service_wait_does_not_cancel_owned_business(self):
        entered, release = asyncio.Event(), asyncio.Event()
        completed = []
        cancelled = []

        async def handler(item):
            entered.set()
            try:
                await release.wait()
                completed.append(item.offset)
            except asyncio.CancelledError:
                cancelled.append(item.offset)
                raise

        service = self.service(FakeConsumer([message(0), message(1)]), handler)
        running = asyncio.create_task(service.run(asyncio.Event()))
        await entered.wait()
        running.cancel()
        await asyncio.sleep(0)
        self.assertFalse(running.done())
        running.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(running, 1)
        self.assertEqual(cancelled, [])
        self.assertEqual(completed, [0, 1])
        await self.assert_drained(service)

    async def test_consumer_failure_waits_for_owned_message_and_accounts_queued_abandonment(self):
        entered, release = asyncio.Event(), asyncio.Event()
        completed = []

        async def handler(item):
            entered.set()
            await release.wait()
            completed.append(item.offset)

        async def fail():
            await entered.wait()
            raise RuntimeError("consumer failed")

        consumer = FakeConsumer([message(0), message(1), message(2)])
        consumer.after_batch = fail
        service = self.service(consumer, handler)
        running = asyncio.create_task(service.run(asyncio.Event()))
        await entered.wait()
        await self.eventually(lambda: consumer.stop_calls == 1)
        self.assertFalse(running.done())
        release.set()
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            await asyncio.wait_for(running, 1)
        self.assertEqual(completed, [0])
        self.assertTrue(any("unprocessed" in error for error in self.logger.errors))
        self.assertEqual(self.metrics.snapshot()["counters"]["messages_unprocessed"]["total"], 2)
        await self.assert_drained(service)

    async def test_consumer_start_failure_still_stops_consumer_and_unregisters_metrics(self):
        consumer = FakeConsumer()
        consumer.start_failure = RuntimeError("connect failed")
        service = self.service(consumer, lambda item: asyncio.sleep(0))
        with self.assertRaisesRegex(RuntimeError, "connect failed"):
            await asyncio.wait_for(service.run(asyncio.Event()), 1)
        self.assertEqual(consumer.stop_calls, 1)
        await self.assert_drained(service)

    async def test_backpressure_waits_before_dequeue_and_does_not_hold_identity_lock(self):
        output_entered, output_ready = asyncio.Event(), asyncio.Event()
        handled = asyncio.Event()
        identity_lock = asyncio.Lock()

        class Output:
            async def wait_for_capacity(self):
                if identity_lock.locked():
                    raise AssertionError("identity lock held during admission")
                output_entered.set()
                await output_ready.wait()

        async def handler(item):
            async with identity_lock:
                handled.set()

        service = self.service(FakeConsumer(), handler, admission=AdmissionController(Output()))
        service.queue.put_nowait(message(0))
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(stop))
        await output_entered.wait()
        self.assertEqual(service.queue.qsize(), 1)
        self.assertEqual(service.scheduler.queue_snapshot()["inflight_messages"], 0)
        async with identity_lock:
            self.assertFalse(handled.is_set())
        output_ready.set()
        await asyncio.wait_for(handled.wait(), 1)
        stop.set()
        await asyncio.wait_for(running, 1)
        await self.assert_drained(service)

    async def test_dispatch_cancel_before_transfer_balances_only_its_dequeued_item(self):
        queue = asyncio.Queue()
        queue.put_nowait(message(0))
        entered, release = asyncio.Event(), asyncio.Event()
        completed = []

        async def handler(item):
            entered.set()
            await release.wait()
            completed.append(item.offset)

        scheduler = MessageScheduler(handler, capacity=1, on_error=lambda error: None, metrics=self.metrics)
        await scheduler.submit(message(-1), on_done=lambda: None)
        await entered.wait()
        dispatch = asyncio.create_task(dispatch_messages("test", queue, asyncio.Event(), scheduler, OpenAdmission(), logger=self.logger, metrics=self.metrics))
        await self.eventually(lambda: scheduler.queue_snapshot()["admission_waiters"] == 1)
        dispatch.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await dispatch
        await asyncio.wait_for(queue.join(), 1)
        self.assertEqual(scheduler.queue_snapshot()["inflight_messages"], 1)
        self.assertEqual(self.metrics.snapshot()["counters"]["messages_unprocessed"]["total"], 1)
        release.set()
        await scheduler.close()
        self.assertEqual(completed, [-1])

    async def test_stop_signal_does_not_hide_dispatch_failure_or_hang_unclaimed_input(self):
        stop = asyncio.Event()

        class FailedAdmission:
            async def wait_for_capacity(self):
                stop.set()
                raise RuntimeError("admission failed at shutdown")

        service = self.service(FakeConsumer(), lambda item: asyncio.sleep(0), admission=FailedAdmission())
        service.queue.put_nowait(message(0))
        running = asyncio.create_task(service.run(stop))
        try:
            with self.assertRaisesRegex(RuntimeError, "admission failed at shutdown"):
                await asyncio.wait_for(asyncio.shield(running), 0.5)
            await self.assert_drained(service)
        finally:
            # 故障实现可能卡在queue.join；只为让失败测试退出而释放测试输入。
            while not service.queue.empty():
                service.queue.get_nowait()
                service.queue.task_done()
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    async def test_dispatch_failure_during_graceful_drain_is_not_hidden_by_queue_join(self):
        entered, fail = asyncio.Event(), asyncio.Event()
        stop = asyncio.Event()
        calls = 0

        class LaterFailedAdmission:
            async def wait_for_capacity(self):
                nonlocal calls
                calls += 1
                if calls > 1:
                    entered.set()
                    await fail.wait()
                    raise RuntimeError("admission failed during drain")

        consumer = FakeConsumer()
        service = self.service(consumer, lambda item: asyncio.sleep(0), admission=LaterFailedAdmission())
        service.queue.put_nowait(message(0))
        service.queue.put_nowait(message(1))
        running = asyncio.create_task(service.run(stop))
        try:
            await entered.wait()
            stop.set()
            await self.eventually(lambda: consumer.stop_calls == 1)
            fail.set()
            with self.assertRaisesRegex(RuntimeError, "admission failed during drain"):
                await asyncio.wait_for(asyncio.shield(running), 0.5)
            await self.assert_drained(service)
        finally:
            fail.set()
            while not service.queue.empty():
                service.queue.get_nowait()
                service.queue.task_done()
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    async def test_invalid_payloads_are_skipped_and_offset_stays_after_local_admission(self):
        consumer = FakeConsumer([message(0, b"not-json"), message(1), message(2, b"{}")] )
        queue = asyncio.Queue()
        stop = asyncio.Event()

        def validate(payload):
            if "index" not in payload:
                raise ValueError("invalid contract")

        task = asyncio.create_task(consume_messages(consumer, queue, stop, parse_context=validate, logger=self.logger, metrics=self.metrics))
        await consumer.committed.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(consumer.commits, [{"partition": 3}])
        self.assertEqual(queue.qsize(), 1)
        self.assertEqual(queue.get_nowait().offset, 1)
        queue.task_done()
        self.assertEqual(len(self.logger.errors), 2)
        self.assertEqual(consumer.stop_calls, 1)
