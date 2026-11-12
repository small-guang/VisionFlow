"""Kafka入口与任务所有权；只依赖消息处理器和接纳接口。"""

from __future__ import annotations

import asyncio
import json

from core.message_scheduler import MessageScheduler
from core.queue_metrics import pipeline_metrics


async def dispatch_messages(name, queue, stop_event, scheduler, admission, *, logger, metrics=pipeline_metrics):
    logger.info("Worker {} started.", name)
    while not stop_event.is_set() or not queue.empty():
        try:
            with metrics.waiting("output_backpressure"):
                await admission.wait_for_capacity()
            message = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        transferred = False
        try:
            await scheduler.submit(message, on_done=queue.task_done)
            transferred = True
        finally:
            if not transferred:
                queue.task_done()
                metrics.increment("messages_unprocessed")
                logger.error("Dispatcher left 1 claimed input message unprocessed before admission")


async def consume_messages(consumer, queue, stop_event, *, parse_context, logger, metrics=pipeline_metrics):
    try:
        await consumer.start()
        while not stop_event.is_set():
            result = await consumer.getmany(timeout_ms=1000, max_records=50)
            for partition, messages in result.items():
                for message in messages:
                    try:
                        payload = json.loads(message.value.decode("utf-8"))
                        parse_context(payload)
                    except (json.JSONDecodeError, AttributeError, ValueError) as error:
                        metrics.increment("invalid_messages")
                        logger.error("Invalid input message at offset {}: {}", message.offset, error)
                        continue
                    with metrics.waiting("input_dispatch"):
                        await queue.put(message)
                    metrics.increment("messages_enqueued")
                if messages:
                    # 兼容现有“本地入队后提交”边界，不伪装为业务成功提交。
                    await consumer.commit({partition: messages[-1].offset + 1})
    finally:
        await consumer.stop()


class WorkerService:
    def __init__(self, *, consumer, handler, admission, worker_count, message_capacity,
                 parse_context, logger, metrics=pipeline_metrics, queue_capacity=1000):
        self.consumer, self.handler, self.admission = consumer, handler, admission
        self.worker_count, self.message_capacity = worker_count, message_capacity
        self.parse_context, self.logger, self.metrics = parse_context, logger, metrics
        self.queue = asyncio.Queue(maxsize=queue_capacity)
        self.scheduler = MessageScheduler(
            handler, capacity=message_capacity, metrics=metrics,
            on_error=lambda error: logger.error("Message task failed: {}", type(error).__name__),
        )
        self._running = False

    async def run(self, stop_event):
        if self._running:
            raise RuntimeError("worker service is already running")
        self._running = True
        self.metrics.register("messages", self.scheduler.queue_snapshot)
        self.metrics.register("input", lambda: {
            "queued_messages": self.queue.qsize(), "capacity_messages": self.queue.maxsize,
            "worker_count": self.worker_count, "broker_lag": None,
        })
        consumer_task = asyncio.create_task(consume_messages(
            self.consumer, self.queue, stop_event, parse_context=self.parse_context,
            logger=self.logger, metrics=self.metrics,
        ), name="alog-consumer")
        workers = [asyncio.create_task(dispatch_messages(
            f"worker-{index}", self.queue, stop_event, self.scheduler, self.admission,
            logger=self.logger, metrics=self.metrics,
        ), name=f"alog-dispatch-{index}") for index in range(self.worker_count)]
        stopped = asyncio.create_task(stop_event.wait(), name="alog-stop")
        graceful = True
        try:
            done, _ = await asyncio.wait([stopped, consumer_task, *workers], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not stopped:
                    if task.cancelled() or task.exception() is not None:
                        graceful = False
                        if task.cancelled():
                            raise RuntimeError(f"{task.get_name()} was cancelled unexpectedly")
                        raise task.exception()
                    if not stop_event.is_set():
                        graceful = False
                        raise RuntimeError(f"{task.get_name()} exited before shutdown")
        finally:
            stop_event.set()
            stopped.cancel()
            consumer_task.cancel()
            cleanup = asyncio.create_task(self._drain(consumer_task, workers, graceful))
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                await cleanup
            finally:
                await asyncio.gather(stopped, return_exceptions=True)
                self.metrics.unregister("input")
                self.metrics.unregister("messages")
                self._running = False

    async def _drain(self, consumer_task, workers, graceful):
        await asyncio.gather(consumer_task, return_exceptions=True)
        failure = None
        joined = None
        try:
            if graceful:
                joined = asyncio.create_task(self.queue.join(), name="alog-input-drain")
                remaining = set(workers)
                while True:
                    done, _ = await asyncio.wait({joined, *remaining}, return_when=asyncio.FIRST_COMPLETED)
                    for worker in done - {joined}:
                        remaining.discard(worker)
                        if worker.cancelled():
                            raise RuntimeError(f"{worker.get_name()} was cancelled during shutdown")
                        if worker.exception() is not None:
                            raise worker.exception()
                    if joined in done:
                        break
                    if not remaining and not self.queue.empty():
                        raise RuntimeError("all dispatchers exited with queued input messages")
        except BaseException as error:
            failure = error
        finally:
            if joined is not None:
                joined.cancel()
                await asyncio.gather(joined, return_exceptions=True)
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.scheduler.close()
        if not graceful or failure is not None:
            abandoned = 0
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
                abandoned += 1
            if abandoned:
                self.metrics.increment("messages_unprocessed", abandoned)
                self.logger.error("Service failure left {} admitted input messages unprocessed", abandoned)
        if failure is not None:
            raise failure
