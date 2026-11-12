"""有界准备阶段内部的并行任务所有权。"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


async def run_owned_parallel(*operations: Coroutine[Any, Any, Any]) -> list[Any]:
    """按传入顺序返回结果；失败或取消时回收全部子协程。

    调用方负责外层容量限制。取消不会中止已经提交的 SDK 线程，线程
    仍由原执行池跟踪至结束并归还许可。这里保证不遗留准备阶段协程。
    """
    tasks = [asyncio.create_task(operation) for operation in operations]
    joined = asyncio.gather(*tasks)
    try:
        # 父任务的取消只由下面的清理路径分发一次，避免重复取消打断
        # 子协程自己的 finally（例如正在释放准入名额）。
        return await asyncio.shield(joined)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        drained = asyncio.gather(*tasks, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        # 父取消时 shield 可能已脱离 joined，主动取走它的异常。
        if joined.done() and not joined.cancelled():
            joined.exception()
        raise
