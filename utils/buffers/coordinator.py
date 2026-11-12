"""写出任务生命周期与入口背压，不参与身份判定或业务记录构造。"""
from __future__ import annotations

import asyncio
from typing import Any
from loguru import logger
from .background_writer import BackgroundWriter
from .kafka_publisher import KafkaPublisher


class OutputCoordinator:
    def __init__(self, *, vector_writers: dict[str, Any], publisher: KafkaPublisher,
                 flush_interval: float, high_watermark: int, low_watermark: int):
        if not 0 <= low_watermark < high_watermark:
            raise ValueError("output backlog must satisfy 0 <= low < high")
        self.vector_writers = dict(vector_writers)
        self.publisher = publisher
        self.flush_interval = flush_interval
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self._backpressured = False
        self._capacity_waiters = 0
        self._capacity_available = asyncio.Event()
        self._capacity_available.set()
        self._running = False
        self._background_enabled = False
        self._flush_task = None
        self._stop_task = None
        self.writers = {
            name: BackgroundWriter(name, owner.queue, owner.flush, owner.max_batch_size)
            for name, owner in self.vector_writers.items()
        }
        for name, queue in publisher.queues.items():
            async def flush(channel=name):
                await publisher.flush(channel)
            self.writers[name] = BackgroundWriter(name, queue, flush, publisher.max_batch_size)
        for writer in self.writers.values():
            writer.queue.on_change = self.update_capacity

    def outstanding_items(self) -> int:
        return sum(writer.queue.qsize() + writer.queue.inflight for writer in self.writers.values())

    def update_capacity(self) -> None:
        outstanding = self.outstanding_items()
        if outstanding >= self.high_watermark:
            self._backpressured = True
        elif outstanding <= self.low_watermark:
            self._backpressured = False
        if self._backpressured:
            self._capacity_available.clear()
        else:
            self._capacity_available.set()

    async def wait_for_capacity(self) -> None:
        if not self._running:
            return
        self.update_capacity()
        self._capacity_waiters += 1
        try:
            while self._running and self._backpressured:
                await self._capacity_available.wait()
        finally:
            self._capacity_waiters -= 1

    def queue_snapshot(self) -> dict[str, Any]:
        return {
            "unit": "output_items", "total_outstanding": self.outstanding_items(),
            "high_watermark": self.high_watermark, "low_watermark": self.low_watermark,
            "hard_limit": False, "backpressured": self._backpressured,
            "capacity_waiters": self._capacity_waiters, "background_running": self._running,
            "queues": {name: writer.queue.snapshot() for name, writer in self.writers.items()},
        }

    async def flush_if_full(self, name: str) -> None:
        writer = self.writers[name]
        if writer.queue.qsize() >= writer.max_batch_size:
            if self._background_enabled:
                writer.request()
            else:
                await writer.flush()

    async def start(self) -> None:
        if self._stop_task is not None:
            await asyncio.shield(self._stop_task)
            self._stop_task = None
        if self._running:
            return
        try:
            await self.publisher.start()
            self._running = True
            self.update_capacity()
            self._background_enabled = True
            for name, owner in self.vector_writers.items():
                owner.request_flush = self.writers[name].request
            self.publisher.request_flush = lambda name: self.writers[name].request()
            for writer in self.writers.values():
                writer.start()
            self._flush_task = asyncio.create_task(self._auto_flush(), name="output-flush-clock")
        except BaseException:
            # 兼容独立使用 FeatureBuffer 的调用方；部分启动也由同一退出任务回收。
            try:
                await self.stop()
            except BaseException:
                logger.exception("Output cleanup after startup failure did not complete")
            raise

    async def stop(self) -> None:
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._finish_stop(), name="output-coordinator-stop")
            self._stop_task.add_done_callback(self._observe_stop_result)
        await asyncio.shield(self._stop_task)

    @staticmethod
    def _observe_stop_result(task: asyncio.Task) -> None:
        if task.cancelled():
            logger.warning("Output cleanup cancelled before completion; pending items remain in memory")
        elif (error := task.exception()) is not None:
            logger.error("Output cleanup failed: {}", error)

    async def _finish_stop(self) -> None:
        self._running = False
        self._capacity_available.set()
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await asyncio.gather(*(writer.stop() for writer in self.writers.values()))
        await self.flush_all()
        self._background_enabled = False
        for owner in self.vector_writers.values():
            owner.request_flush = None
        self.publisher.request_flush = None
        if self.outstanding_items():
            logger.warning("Output shutdown left {} items pending after failed writes", self.outstanding_items())
        await self.publisher.stop()

    async def flush_all(self) -> None:
        # 向量完成产生的 Kafka 记录也必须纳入同一次最终排空。
        await asyncio.gather(*(self.writers[name].drain() for name in self.vector_writers))
        await asyncio.gather(*(self.writers[name].drain() for name in self.publisher.queues))

    async def _auto_flush(self) -> None:
        while self._running:
            await asyncio.sleep(self.flush_interval)
            for writer in self.writers.values():
                writer.request()
