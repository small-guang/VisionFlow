"""记录注入资源域或兼容默认线程池的排队、执行和回调延迟。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass
import time
from typing import TypeVar

from core.performance import runtime_performance


T = TypeVar("T")


@dataclass(slots=True)
class _FutureThreadSpan:
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    callback_at: float | None = None
    failed: bool = False


async def gather_timed_threads(
    operation: str, /, *jobs: Callable[[], T], executor=None,
) -> list[T]:
    """并行提交阻塞工作，保留 gather 的结果顺序、异常及取消行为。

    每路独立复制 context；调用参数由 job（例如 partial）持有。取消未启动
    的 Future 可阻止执行，已启动的线程继续，并在完成后记录原三项线程指标。
    future_callback 是 Future 完成回调时间，不表示逐分支实际 await 恢复。
    完整成功组按最后完成回调选择关键分支，其五段合计 group_wait。
    注入 executor 时许可等待也包含在线程排队时间内。
    """
    loop = asyncio.get_running_loop()
    started_at = time.perf_counter()
    metrics = runtime_performance
    spans: list[_FutureThreadSpan] = []
    futures: list[asyncio.Future[T]] = []

    def submit(job: Callable[[], T]) -> asyncio.Future[T]:
        span = _FutureThreadSpan(submitted_at=time.perf_counter())
        spans.append(span)
        context = copy_context()

        def record_thread() -> None:
            callback_at = time.perf_counter()
            for suffix, elapsed in (
                ("thread_queue", span.started_at - span.submitted_at),
                ("thread_run", span.finished_at - span.started_at),
                ("loop_resume", callback_at - span.finished_at),
            ):
                metrics.record(
                    f"{operation}_{suffix}", max(0.0, elapsed) * 1000,
                    failed=span.failed,
                )

        def run() -> T:
            span.started_at = time.perf_counter()
            try:
                return job()
            except BaseException:
                span.failed = True
                raise
            finally:
                span.finished_at = time.perf_counter()
                try:
                    loop.call_soon_threadsafe(record_thread)
                except RuntimeError:
                    # 关闭事件循环不改变已有线程的返回值或异常。
                    pass

        def completed(future: asyncio.Future[T]) -> None:
            # Future 取消不表示已有线程完成，也不表示其完成回调被实际观察到。
            if future.cancelled() or span.finished_at is None:
                return
            span.callback_at = time.perf_counter()
            metrics.record(
                f"{operation}_future_callback",
                (span.callback_at - span.finished_at) * 1000,
                failed=span.failed,
            )

        if executor is None:
            future = loop.run_in_executor(None, context.run, run)
        else:
            future = asyncio.create_task(executor.run(context.run, run))
        future.add_done_callback(completed)
        return future

    failed = True
    try:
        try:
            for job in jobs:
                futures.append(submit(job))
        except BaseException:
            # 提交失败仍观察已提交的 Future；保留后台运行和原始提交异常。
            asyncio.gather(*futures, return_exceptions=True)
            raise
        result = await asyncio.gather(*futures)
        failed = False
        return result
    finally:
        resumed_at = time.perf_counter()
        metrics.record(
            f"{operation}_group_wait", (resumed_at - started_at) * 1000,
            failed=failed,
        )
        if not failed and spans and all(span.callback_at is not None for span in spans):
            critical = max(spans, key=lambda span: span.callback_at)
            for suffix, elapsed in (
                (
                    "future_group_submit_span",
                    max(span.submitted_at for span in spans)
                    - min(span.submitted_at for span in spans),
                ),
                ("future_critical_dispatch", critical.submitted_at - started_at),
                ("future_critical_thread_queue", critical.started_at - critical.submitted_at),
                ("future_critical_thread_run", critical.finished_at - critical.started_at),
                ("future_critical_callback", critical.callback_at - critical.finished_at),
                ("future_group_join", resumed_at - critical.callback_at),
            ):
                metrics.record(f"{operation}_{suffix}", elapsed * 1000)
