"""由应用持有的阻塞工作资源；业务只接收所需的窄执行接口。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import copy_context
from functools import partial
from typing import Protocol, TypeVar

from core.bounded_cpu_pool import BoundedCpuPool, TimingRecorder
from core.vector_query import VectorQueryCoordinator, validate_query_limits


T = TypeVar("T")


class WorkExecutor(Protocol):
    async def run(self, operation: Callable[..., T], /, *args, **kwargs) -> T:
        ...


class _NamedTiming:
    def __init__(self, recorder: TimingRecorder, names: dict[str, str]) -> None:
        self.recorder = recorder
        self.names = names

    def record(self, operation: str, duration_ms: float, *, failed: bool = False) -> None:
        self.recorder.record(operation, duration_ms, failed=failed)
        named = self.names.get(operation)
        if named is not None and named != operation:
            self.recorder.record(named, duration_ms, failed=failed)


async def run_blocking(
    executor: WorkExecutor | None, operation: Callable[..., T], /, *args, **kwargs,
) -> T:
    """无注入时兼容独立调用；生产装配必须提供对应资源域。"""
    if executor is None:
        return await asyncio.to_thread(operation, *args, **kwargs)
    return await executor.run(operation, *args, **kwargs)


class AsyncWorkExecutor:
    """只把获得许可的工作提交给线程池，不在 executor 内累积无界队列。

    等待许可的调用者仍受上游消息/目标窗口约束，并单独计入 queued_jobs。
    调用者取消不提前归还真实工作许可；上下文和原始异常跨线程传递。
    """

    def __init__(
        self, workers: int, *, name: str, metrics: TimingRecorder | None = None,
    ) -> None:
        self.name = name
        self._metrics = metrics
        self._pool = BoundedCpuPool(
            workers,
            thread_name_prefix=f"alog-{name.replace('_', '-')}",
            operation_name=f"{name}_run",
            queue_operation_name=f"{name}_queue_wait",
        )

    @property
    def workers(self) -> int:
        return self._pool.workers

    async def run(
        self, operation: Callable[..., T], /, *args,
        metrics: TimingRecorder | None = None,
        drain_on_cancel: bool = False,
        **kwargs,
    ) -> T:
        context = copy_context()
        work = partial(operation, *args, **kwargs)
        return await self._pool.submit(
            context.run, work,
            metrics=self._metrics if metrics is None else metrics,
            drain_on_cancel=drain_on_cancel,
        )

    async def run_group(self, operation_name: str, /, *jobs: Callable[[], T]) -> list[T]:
        from core.thread_timing import gather_timed_threads

        return await gather_timed_threads(operation_name, *jobs, executor=self)

    async def run_timed(
        self, operation_name: str, operation: Callable[..., T], /, *args,
        queue_operation_name: str | None = None,
        metrics: TimingRecorder | None = None,
        **kwargs,
    ) -> T:
        """在同一次真实执行上附加业务步骤计时，不额外创建队列或线程池。"""
        recorder = self._metrics if metrics is None else metrics
        if recorder is not None:
            recorder = _NamedTiming(recorder, {
                f"{self.name}_run": operation_name,
                f"{self.name}_queue_wait": queue_operation_name or f"{operation_name}_queue_wait",
            })
        return await self.run(operation, *args, metrics=recorder, **kwargs)

    def queue_snapshot(self) -> dict:
        return self._pool.queue_snapshot()

    async def close(self) -> None:
        await self._pool.close()


class ExecutionResources:
    """一个 Worker 生命周期内共享的四个资源域，不依赖配置或业务单例。"""

    def __init__(
        self, *, image_workers: int, storage_workers: int,
        search_workers: int, write_workers: int,
        storage_download_reserved: int = 0,
        query_batch_size: int = 4, query_wait_ms: float = 1.0, query_capacity: int = 128,
        metrics: TimingRecorder | None = None,
    ) -> None:
        validate_query_limits(query_batch_size, query_wait_ms, query_capacity)
        budgets = {
            "image_cpu": image_workers,
            "storage_io": storage_workers,
            "vector_search": search_workers,
            "vector_write": write_workers,
        }
        self._initialize_resources(budgets, metrics, storage_download_reserved)
        self.image_cpu = self._executors["image_cpu"]
        self.storage_io = self._executors["storage_io"]
        self.vector_search = self._executors["vector_search"]
        self.vector_write = self._executors["vector_write"]
        self.vector_queries = VectorQueryCoordinator(
            self.vector_search, batch_size=query_batch_size, wait_ms=query_wait_ms,
            capacity=query_capacity, metrics=metrics,
        )

    def _initialize_resources(
        self, budgets: dict[str, int], metrics: TimingRecorder | None,
        storage_download_reserved: int = 0,
    ) -> None:
        from core.storage_execution import StorageWorkExecutor

        for name, workers in budgets.items():
            if type(workers) is not int or workers < 1:
                raise ValueError(f"{name} workers must be a positive integer")
        if (type(storage_download_reserved) is not int
                or not 0 <= storage_download_reserved < budgets.get("storage_io", 1)):
            raise ValueError("storage download reserve must be smaller than storage workers")
        self._executors = {
            name: (StorageWorkExecutor(
                workers, download_reserved=storage_download_reserved, metrics=metrics,
            ) if name == "storage_io" else AsyncWorkExecutor(workers, name=name, metrics=metrics))
            for name, workers in budgets.items()
        }
        self._close_task: asyncio.Task | None = None

    def queue_snapshot(self) -> dict[str, dict]:
        result = {name: executor.queue_snapshot() for name, executor in self._executors.items()}
        coordinator = self.vector_queries
        if coordinator is not None:
            result["vector_queries"] = coordinator.queue_snapshot()
        return result

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._finish_close(), name="alog-execution-close",
            )
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        outcomes = []
        coordinator = self.vector_queries
        if coordinator is not None:
            try:
                await coordinator.close()
            except BaseException as error:
                outcomes.append(error)
        outcomes.extend(await asyncio.gather(
            *(executor.close() for executor in self._executors.values()),
            return_exceptions=True,
        ))
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            raise RuntimeError("failed to close execution resources") from failures[0]

    async def __aenter__(self) -> ExecutionResources:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


class VisualExecutionResources(ExecutionResources):
    """视觉子进程只创建图像和存储池，不创建数据库执行资源。"""

    def __init__(
        self, *, image_workers: int, storage_workers: int,
        storage_download_reserved: int = 0,
        metrics: TimingRecorder | None = None,
    ) -> None:
        self._initialize_resources(
            {"image_cpu": image_workers, "storage_io": storage_workers}, metrics,
            storage_download_reserved,
        )
        self.image_cpu = self._executors["image_cpu"]
        self.storage_io = self._executors["storage_io"]
        self.vector_queries = None


class OwnerExecutionResources(ExecutionResources):
    """身份与输出 owner 只创建 Milvus 搜索、写入池。"""

    def __init__(
        self, *, search_workers: int, write_workers: int,
        query_batch_size: int = 4, query_wait_ms: float = 1.0, query_capacity: int = 128,
        metrics: TimingRecorder | None = None,
    ) -> None:
        validate_query_limits(query_batch_size, query_wait_ms, query_capacity)
        self._initialize_resources(
            {"vector_search": search_workers, "vector_write": write_workers}, metrics,
        )
        self.vector_search = self._executors["vector_search"]
        self.vector_write = self._executors["vector_write"]
        self.vector_queries = VectorQueryCoordinator(
            self.vector_search, batch_size=query_batch_size, wait_ms=query_wait_ms,
            capacity=query_capacity, metrics=metrics,
        )
