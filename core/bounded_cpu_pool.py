"""有界 CPU 线程池；许可、结果和聚合指标由所属事件循环管理。"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Thread
import time
from typing import Callable, Generic, Protocol, TypeVar, cast


T = TypeVar("T")


class TimingRecorder(Protocol):
    def record(self, operation: str, duration_ms: float, *, failed: bool = False) -> None:
        ...


@dataclass
class _CpuOutcome(Generic[T]):
    value: T | None
    error: BaseException | None
    queue_wait_ms: float
    operation_ms: float


class BoundedCpuPool:
    """真实在途任务不超过 workers；取消等待方不会提前释放提交名额。"""

    def __init__(
        self,
        workers: int,
        *,
        thread_name_prefix: str,
        operation_name: str,
        queue_operation_name: str,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("CPU workers must be a positive integer")
        self.workers = workers
        self._loop = asyncio.get_running_loop()
        self._clock = clock
        self._operation_name = operation_name
        self._queue_operation_name = queue_operation_name
        self._thread_name_prefix = thread_name_prefix
        self._permits = asyncio.Semaphore(workers)
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=thread_name_prefix
        )
        self._pending: set[asyncio.Future] = set()
        self._waiting_jobs = 0
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def _run(
        self, operation: Callable[..., T], args: tuple, queued_at: float
    ) -> _CpuOutcome[T]:
        started = self._clock()
        value, error = None, None
        try:
            value = operation(*args)
        except BaseException as exc:
            error = exc
        return _CpuOutcome(
            value, error, (started - queued_at) * 1000,
            (self._clock() - started) * 1000,
        )

    def _complete(
        self, work: Future, result: asyncio.Future, metrics: TimingRecorder | None
    ) -> None:
        # 工作线程只计时；结果、许可及指标都在所属 loop 更新。
        try:
            outcome = work.result()
        except BaseException as exc:
            outcome = _CpuOutcome(None, exc, 0.0, 0.0)
        try:
            if metrics is not None:
                metrics.record(
                    self._operation_name, outcome.operation_ms,
                    failed=outcome.error is not None,
                )
                metrics.record(self._queue_operation_name, outcome.queue_wait_ms)
        except Exception as exc:
            # 观测失败不能替换原操作结果，也不能占住许可。
            self._loop.call_exception_handler({
                "message": "CPU performance metric recording failed",
                "exception": exc,
            })
        finally:
            if not result.done():
                result.set_result(outcome)
            self._pending.discard(result)
            self._permits.release()

    async def submit(
        self, operation: Callable[..., T], *args: object,
        metrics: TimingRecorder | None = None,
        drain_on_cancel: bool = False,
    ) -> T:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("CPU pool belongs to a different event loop")
        if self._closed:
            raise RuntimeError("CPU pool is closed")
        queued_at = self._clock()
        self._waiting_jobs += 1
        try:
            await self._permits.acquire()
        finally:
            self._waiting_jobs -= 1
        if self._closed:
            self._permits.release()
            raise RuntimeError("CPU pool is closed")
        result = self._loop.create_future()
        try:
            work = self._executor.submit(self._run, operation, args, queued_at)
        except BaseException:
            self._permits.release()
            raise
        self._pending.add(result)
        work.add_done_callback(
            lambda completed: self._loop.call_soon_threadsafe(
                self._complete, completed, result, metrics
            )
        )
        # shield 保留真实工作的结果和名额，调用方取消只停止它自己的等待。
        try:
            outcome = await asyncio.shield(result)
        except asyncio.CancelledError:
            if drain_on_cancel:
                # 共享内存复制仍在使用目标槽，真实完成前不能归还或销毁它。
                # 只保护已提交工作；上方等待 permit 时取消仍立即停止提交。
                while not result.done():
                    try:
                        await asyncio.shield(result)
                    except asyncio.CancelledError:
                        continue
            raise
        if outcome.error is not None:
            raise outcome.error
        return cast(T, outcome.value)

    def queue_snapshot(self) -> dict:
        """许可等待与真实线程在途分别计数；调用方取消不抹去尚未完成的工作。"""
        return {
            "closed": self._closed,
            "queued_jobs": self._waiting_jobs,
            "inflight_jobs": len(self._pending),
            "capacity_jobs": self.workers,
        }

    async def close(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("CPU pool belongs to a different event loop")
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(
                self._finish_close(), name=f"{self._thread_name_prefix}-close"
            )
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending)
        finished = self._loop.create_future()

        def complete(error: BaseException | None = None) -> None:
            if finished.done():
                return
            if error is None:
                finished.set_result(None)
            else:
                finished.set_exception(error)

        def shutdown() -> None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as error:
                self._loop.call_soon_threadsafe(complete, error)
            else:
                self._loop.call_soon_threadsafe(complete)

        # 关闭自有资源不能依赖可能已被慢 IO 占满的默认线程池。
        Thread(target=shutdown, name=f"{self._thread_name_prefix}-shutdown").start()
        await asyncio.shield(finished)
