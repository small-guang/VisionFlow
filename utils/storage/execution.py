"""按方向选择共享存储池的准入与计时，兼容独立调用执行器。"""

from collections.abc import Callable
from typing import TypeVar

from core.execution import AsyncWorkExecutor, WorkExecutor, run_blocking
from core.storage_execution import StorageWorkExecutor


T = TypeVar("T")


async def run_storage_operation(
    executor: WorkExecutor | None, direction: str,
    operation: Callable[..., T], /, *args, **kwargs,
) -> T:
    if direction not in ("download", "upload"):
        raise ValueError("storage direction must be download or upload")
    if isinstance(executor, StorageWorkExecutor):
        return await executor.run_storage(direction, operation, *args, **kwargs)
    if isinstance(executor, AsyncWorkExecutor):
        return await executor.run_timed(
            f"storage_{direction}_run", operation, *args,
            queue_operation_name=f"storage_{direction}_queue_wait", **kwargs,
        )
    return await run_blocking(executor, operation, *args, **kwargs)
