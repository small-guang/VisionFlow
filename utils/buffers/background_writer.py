"""Single-owner flush scheduling and item-level output backlog accounting."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
import time
from typing import Any

from loguru import logger


class WriteQueue(asyncio.Queue):
    """Track queued and checked-out items without adding waits inside identity locks.

    The queue remains unbounded. The service applies admission backpressure before
    taking another input frame; frames already in flight may exceed its watermark.
    """

    def __init__(self, *, unit: str, on_change: Callable[[], None] | None = None):
        super().__init__()
        self.unit = unit
        self.on_change = on_change
        self.flush_lock = asyncio.Lock()
        self.inflight = 0
        self.accepted_total = 0
        self.completed_total = 0
        self.retried_total = 0
        self.retry_queued = 0
        self._queued_times: deque[float] = deque()
        self._retry_flags: deque[bool] = deque()

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def put_nowait(self, item: Any) -> None:
        super().put_nowait(item)
        self._queued_times.append(time.monotonic())
        self._retry_flags.append(False)
        self.accepted_total += 1
        self._changed()

    def get_nowait(self) -> Any:
        item = super().get_nowait()
        self._queued_times.popleft()
        if self._retry_flags.popleft():
            self.retry_queued -= 1
        self.inflight += 1
        self._changed()
        return item

    def complete(self, count: int = 1) -> None:
        """Complete terminal policy outcomes, or records acknowledged by Kafka."""
        self.inflight -= count
        self.completed_total += count
        for _ in range(count):
            self.task_done()
        self._changed()

    def requeue(self, items: Iterable[Any]) -> None:
        """Return an unfinished batch atomically, retaining its outstanding count."""
        for item in items:
            super().put_nowait(item)
            self._queued_times.append(time.monotonic())
            self._retry_flags.append(True)
            self.retry_queued += 1
            self.retried_total += 1
            self.inflight -= 1
            self.task_done()
        self._changed()

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "unit": self.unit,
            "queued": self.qsize(),
            "inflight": self.inflight,
            "retry_queued": self.retry_queued,
            "accepted_total": self.accepted_total,
            "completed_total": self.completed_total,
            "retried_total": self.retried_total,
            "oldest_queued_ms": (
                (time.monotonic() - self._queued_times[0]) * 1000
                if self._queued_times else 0.0
            ),
        }
        if self.unit == "kafka_record":
            snapshot["acked_total"] = self.completed_total
        return snapshot


class BackgroundWriter:
    """One scheduled flusher per queue; retries wait for a later trigger."""

    def __init__(
        self,
        name: str,
        queue: WriteQueue,
        flush: Callable[[], Awaitable[None]],
        max_batch_size: int,
    ):
        self.name = name
        self.queue = queue
        self.flush = flush
        self.max_batch_size = max_batch_size
        self._wake = asyncio.Event()
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._wake.clear()
        self._task = asyncio.create_task(self._run(), name=f"buffer-{self.name}")
        if self.queue.qsize() >= self.max_batch_size:
            self.request()

    def request(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        # Let an actual write finish instead of cancelling an underlying I/O
        # thread and starting a second write during the shutdown drain.
        self._running = False
        self._wake.set()
        task = self._task
        if task is not None:
            # Cancelling the caller's shutdown wait must not cancel a DB await
            # whose underlying synchronous write may still be running.
            await asyncio.shield(task)
            if self._task is task:
                self._task = None

    async def _run(self) -> None:
        while self._running:
            await self._wake.wait()
            self._wake.clear()
            if not self._running:
                break
            retries_before = self.queue.retried_total
            completed_before = self.queue.completed_total
            try:
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected background buffer failure: {}", self.name)
            # A failed batch is retried on the timer or a later producer event.
            # Do not spin just because its requeued size remains >= batch size.
            if (
                self.queue.retried_total == retries_before
                and self.queue.completed_total > completed_before
                and self.queue.qsize() >= self.max_batch_size
            ):
                self.request()

    async def drain(self) -> None:
        """Drain successful batches; leave a failing batch available for retry."""
        while True:
            retries_before = self.queue.retried_total
            completed_before = self.queue.completed_total
            # Even an empty queue may have a batch in flight. The flusher's
            # owner lock also joins that write before this explicit drain ends.
            await self.flush()
            if (
                self.queue.empty()
                or self.queue.retried_total != retries_before
                or self.queue.completed_total == completed_before
            ):
                break
