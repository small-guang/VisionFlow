"""逐帧有界准备流水线：准备可提前推进，提交始终遵循输入顺序。"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

from core.queue_metrics import PipelineMetrics, WaitTicket, pipeline_metrics


_Item = TypeVar("_Item")
_Prepared = TypeVar("_Prepared")


async def run_ordered_pipeline(
    items: Iterable[_Item],
    prepare: Callable[[_Item], Awaitable[_Prepared]],
    commit: Callable[[_Prepared], Awaitable[None]],
    *,
    concurrency: int,
    lookahead: int,
    stage_name: str,
    metrics: PipelineMetrics = pipeline_metrics,
) -> None:
    """最多同时准备 N 项，并为未完成提交的项目保留 N + M 容量。

    容量在启动准备前按输入顺序预留，覆盖准备中、已就绪和提交中项目，
    避免后续结果占满就绪队列、阻塞队首交付的死锁。准备一旦结束就释放
    执行槽；单独的提交任务按输入顺序运行。M=0 保持原有 N 项窗口。

    调用方负责跨帧的资源限流及提交结果收集。本函数只拥有当前帧任务；
    后序准备失败会停止接纳新项目，前序仍按顺序提交；提交到失败项时，
    或提交失败/外部取消时，先取消并排空剩余任务再返回异常。
    stage_name 的等待指标仅覆盖准备完成到实际开始提交的时间。
    """
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead < 0:
        raise ValueError("lookahead must be a non-negative integer")

    remaining = iter(items)
    capacity = concurrency + lookahead
    exhausted = False
    preparation_failed = False
    pending: deque[asyncio.Task[_Prepared]] = deque()
    preparing: set[asyncio.Task[_Prepared]] = set()
    committing: asyncio.Task[None] | None = None
    ready_tickets: dict[asyncio.Task, WaitTicket] = {}

    async def prepare_item(item: _Item) -> _Prepared:
        prepared = await prepare(item)
        # 在任务返回前创建，避免 done callback 晚于提交开始而导致计数泄漏。
        ready_tickets[asyncio.current_task()] = metrics.begin_wait(stage_name)
        return prepared

    async def commit_item(task: asyncio.Task[_Prepared]) -> None:
        prepared = task.result()
        ready_tickets.pop(task).close()
        await commit(prepared)

    try:
        while True:
            # 后序失败只停止继续填充；前序身份/输出仍遵循原有提交边界。
            for task in tuple(preparing):
                if task.done():
                    if task.cancelled() or task.exception() is not None:
                        preparation_failed = True
                    preparing.remove(task)
            if committing is not None and committing.done():
                committing.result()
                committing = None
                pending.popleft()

            if committing is None and pending and pending[0].done():
                committing = asyncio.create_task(
                    commit_item(pending[0]), name=f"alog-{stage_name}-commit",
                )

            # 不能先创建无界任务再等待许可；容量预留必须发生在准备开始前。
            while (
                not exhausted and not preparation_failed
                and len(pending) < capacity and len(preparing) < concurrency
            ):
                try:
                    item = next(remaining)
                except StopIteration:
                    exhausted = True
                    break
                task = asyncio.create_task(
                    prepare_item(item), name=f"alog-{stage_name}-prepare",
                )
                pending.append(task)
                preparing.add(task)

            if exhausted and not pending:
                return
            waiting = set(preparing)
            if committing is not None:
                waiting.add(committing)
            await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
    finally:
        owned = list(pending)
        if committing is not None:
            owned.append(committing)
        for task in owned:
            if not task.done():
                task.cancel()
        try:
            if owned:
                drain = asyncio.gather(*owned, return_exceptions=True)
                cancelled_while_draining = False
                # 再次取消父任务也不能跳过子任务的 finally 和资源归还。
                while not drain.done():
                    try:
                        await asyncio.shield(drain)
                    except asyncio.CancelledError:
                        cancelled_while_draining = True
                drain.result()
                if cancelled_while_draining:
                    raise asyncio.CancelledError
        finally:
            for ticket in ready_tickets.values():
                ticket.close()
