"""有界消息所有权；接纳返回不表示图片业务已完成。"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from core.queue_metrics import PipelineMetrics, pipeline_metrics


class MessageScheduler:
    def __init__(
        self,
        handler: Callable[[Any], Awaitable[None]],
        *,
        capacity: int,
        on_error: Callable[[BaseException], None],
        metrics: PipelineMetrics = pipeline_metrics,
    ):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._handler = handler
        self._on_error = on_error
        self._metrics = metrics
        self._slots = asyncio.Semaphore(capacity)
        self._tasks: set[asyncio.Task] = set()
        self._waiting = 0
        self._closed = False
        self._close_task: asyncio.Task | None = None

    async def submit(self, message: Any, *, on_done: Callable[[], None]) -> None:
        """接纳后所有权转移；只有真实任务结束才调用 on_done，恰好一次。"""
        if self._closed:
            raise RuntimeError("message scheduler is closed")
        self._waiting += 1
        try:
            with self._metrics.waiting("message_admission"):
                await self._slots.acquire()
        finally:
            self._waiting -= 1
        if self._closed:
            self._slots.release()
            raise RuntimeError("message scheduler is closed")
        # No suspension between ownership transfer and returning to dispatcher.
        task = asyncio.create_task(self._run(message), name="alog-message")
        self._tasks.add(task)

        def finished(completed: asyncio.Task) -> None:
            self._tasks.discard(completed)
            self._slots.release()
            try:
                if not completed.cancelled():
                    error = completed.exception()
                    if error is not None:
                        self._on_error(error)
            finally:
                on_done()

        task.add_done_callback(finished)

    async def _run(self, message: Any) -> None:
        with self._metrics.activity("message_processing"):
            await self._handler(message)

    async def close(self) -> None:
        """停止接纳并等待全部自有任务；调用者取消不取消真实业务任务。"""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def queue_snapshot(self) -> dict:
        return {
            "unit": "messages",
            "capacity_messages": self.capacity,
            "inflight_messages": len(self._tasks),
            "admission_waiters": self._waiting,
            "closed": self._closed,
        }
