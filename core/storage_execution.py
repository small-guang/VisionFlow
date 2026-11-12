"""共享存储线程池上的分类准入；上传不能占用下载保底额度。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import time
from typing import TypeVar

from core.bounded_cpu_pool import TimingRecorder
from core.execution import AsyncWorkExecutor


T = TypeVar("T")


class _AdmissionTiming:
    def __init__(self, recorder: TimingRecorder, waiting_ms: float) -> None:
        self.recorder = recorder
        self.waiting_ms = waiting_ms

    def record(self, operation: str, duration_ms: float, *, failed: bool = False) -> None:
        if operation in ("storage_io_queue_wait", "storage_upload_queue_wait"):
            duration_ms += self.waiting_ms
        self.recorder.record(operation, duration_ms, failed=failed)


class StorageWorkExecutor(AsyncWorkExecutor):
    """下载可用全池；上传在 loop 中等待准入，不占下载保留线程。

    每次上传由本执行器持有一个任务，复用底层真实 Future 的 drain 语义。
    调用方取消立即返回；已提交 SDK 工作仍完成后才归还上传和全池名额。
    """

    def __init__(
        self, workers: int, *, download_reserved: int = 0,
        metrics: TimingRecorder | None = None,
    ) -> None:
        if type(download_reserved) is not int or not 0 <= download_reserved < workers:
            raise ValueError("storage download reserve must be nonnegative and smaller than workers")
        super().__init__(workers, name="storage_io", metrics=metrics)
        self._loop = asyncio.get_running_loop()
        self.download_reserved = download_reserved
        self.upload_capacity = workers - download_reserved
        self._upload_condition = asyncio.Condition()
        self._upload_waiters = 0
        self._admitted_uploads = 0
        self._upload_tasks: set[asyncio.Task] = set()
        self._storage_closed = False
        self._storage_close_task: asyncio.Task | None = None

    async def run_storage(
        self, direction: str, operation: Callable[..., T], /, *args,
        metrics: TimingRecorder | None = None, drain_on_cancel: bool = False,
        **kwargs,
    ) -> T:
        if direction not in ("download", "upload"):
            raise ValueError("storage direction must be download or upload")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("storage pool belongs to a different event loop")
        if self._storage_closed:
            raise RuntimeError("storage pool is closed")
        if direction == "download" or not self.download_reserved:
            return await self.run_timed(
                f"storage_{direction}_run", operation, *args,
                queue_operation_name=f"storage_{direction}_queue_wait",
                metrics=metrics, drain_on_cancel=drain_on_cancel, **kwargs,
            )
        task = asyncio.create_task(
            self._run_upload(
                time.perf_counter(), operation, args, kwargs,
                self._metrics if metrics is None else metrics,
            ),
            name="alog-storage-upload",
        )
        self._upload_tasks.add(task)
        task.add_done_callback(self._upload_done)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # 未提交工作停止等待；真实线程的 drain 在 owned task 内继续。
            task.cancel()
            if drain_on_cancel:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
            raise

    def _upload_done(self, task: asyncio.Task) -> None:
        self._upload_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _run_upload(
        self, queued_at: float, operation: Callable[..., T], args, kwargs,
        recorder: TimingRecorder | None,
    ) -> T:
        async with self._upload_condition:
            self._upload_waiters += 1
            try:
                await self._upload_condition.wait_for(
                    lambda: self._storage_closed or self._admitted_uploads < self.upload_capacity,
                )
            finally:
                self._upload_waiters -= 1
            if self._storage_closed:
                raise RuntimeError("storage pool is closed")
            self._admitted_uploads += 1
        try:
            if recorder is not None:
                recorder = _AdmissionTiming(recorder, (time.perf_counter() - queued_at) * 1000)
            return await self.run_timed(
                "storage_upload_run", operation, *args,
                queue_operation_name="storage_upload_queue_wait",
                metrics=recorder, drain_on_cancel=True, **kwargs,
            )
        finally:
            async with self._upload_condition:
                self._admitted_uploads -= 1
                # 已唤醒者仍可能被取消，所有等待者重查容量避免丢失唤醒。
                self._upload_condition.notify_all()

    def queue_snapshot(self) -> dict:
        snapshot = super().queue_snapshot()
        snapshot["closed"] = self._storage_closed
        snapshot["queued_jobs"] += self._upload_waiters
        snapshot.update(
            download_reserved_jobs=self.download_reserved,
            upload_capacity_jobs=self.upload_capacity,
            upload_admission_waiters=self._upload_waiters,
            admitted_uploads=self._admitted_uploads,
        )
        return snapshot

    async def close(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("storage pool belongs to a different event loop")
        if self._storage_close_task is None:
            self._storage_closed = True
            self._storage_close_task = asyncio.create_task(
                self._close_storage(), name="alog-storage-close",
            )
        await asyncio.shield(self._storage_close_task)

    async def _close_storage(self) -> None:
        async with self._upload_condition:
            self._upload_condition.notify_all()
        try:
            await super().close()
        finally:
            if self._upload_tasks:
                await asyncio.gather(*self._upload_tasks, return_exceptions=True)
