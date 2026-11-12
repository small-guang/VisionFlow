"""进程角色的资源装配与全局预算分账；导入时不创建外部连接。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_SPLIT_POSITIVE_FIELDS = (
    "IMAGE_CPU_WORKERS", "TRITON_CLIENT_POOL_SIZE",
    "TRITON_EXECUTOR_CONCURRENCY", "TRITON_EXECUTOR_QUEUE_CAPACITY",
    "FACE_GLOBAL_CONCURRENCY", "PERSON_GLOBAL_CONCURRENCY",
)


def validate_visual_budgets(config: Any, worker_count: int) -> None:
    """不能通过向上取整把小于进程数的全局额度偷偷扩大。"""
    if type(worker_count) is not int or not 1 <= worker_count <= 4:
        raise ValueError("visual worker_count must be an integer between 1 and 4")
    values = {name: getattr(config, name) for name in _SPLIT_POSITIVE_FIELDS}
    values["STORAGE_IO_WORKERS"] = config.STORAGE_IO_WORKERS
    values["PERSON_FACE_CONCURRENCY (effective)"] = (
        config.PERSON_FACE_CONCURRENCY or config.PERSON_GLOBAL_CONCURRENCY
    )
    for name, value in values.items():
        if type(value) is not int or value < worker_count:
            raise ValueError(f"{name} must be at least VISUAL_WORKER_PROCESSES ({worker_count})")
    reserved = config.STORAGE_DOWNLOAD_RESERVED_WORKERS
    if (type(reserved) is not int or reserved < 0
            or (reserved and not worker_count <= reserved <= config.STORAGE_IO_WORKERS - worker_count)):
        raise ValueError(
            "STORAGE_DOWNLOAD_RESERVED_WORKERS must be 0 or reserve at least one download "
            "and one upload slot per VISUAL_WORKER_PROCESSES"
        )


@dataclass(frozen=True)
class VisualConfigView:
    """保留原配置优先级，只覆盖分配给本视觉进程的有效资源额度。"""

    base: Any
    overrides: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        # pickle 恢复对象时 base/overrides 尚不存在，不能递归调用 __getattr__。
        overrides = object.__getattribute__(self, "overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "base"), name)


def make_visual_config(
    config: Any, worker_index: int, worker_count: int | None = None,
) -> VisualConfigView:
    """商余分配保持各进程额度之和等于配置总额，帧额度始终由 owner 持有。"""
    if isinstance(config, VisualConfigView):
        raise ValueError("visual resource budgets have already been divided")
    count = config.VISUAL_WORKER_PROCESSES if worker_count is None else worker_count
    validate_visual_budgets(config, count)
    if type(worker_index) is not int or not 0 <= worker_index < count:
        raise ValueError("worker_index must identify one visual worker")

    def share(total: int) -> int:
        quotient, remainder = divmod(total, count)
        return quotient + int(worker_index < remainder)

    overrides = {name: share(getattr(config, name)) for name in _SPLIT_POSITIVE_FIELDS}
    overrides.update(
        MP_WORKER_INDEX=worker_index,
        MP_WORKER_COUNT=count,
        STORAGE_IO_WORKERS=share(config.STORAGE_IO_WORKERS),
        STORAGE_DOWNLOAD_RESERVED_WORKERS=share(config.STORAGE_DOWNLOAD_RESERVED_WORKERS),
        PERSON_FACE_CONCURRENCY=share(
            config.PERSON_FACE_CONCURRENCY or config.PERSON_GLOBAL_CONCURRENCY,
        ),
        TRITON_SYSTEM_SHM_SLOTS=share(config.TRITON_SYSTEM_SHM_SLOTS),
        TRITON_SYSTEM_SHM_COPY_WORKERS=(share(min(4, config.TRITON_SYSTEM_SHM_SLOTS)) or None),
    )
    for prefix in ("FACE", "PERSON"):
        overrides[f"{prefix}_FRAME_CONCURRENCY"] = min(
            getattr(config, f"{prefix}_FRAME_CONCURRENCY"),
            overrides[f"{prefix}_GLOBAL_CONCURRENCY"],
        )
    return VisualConfigView(config, overrides)


def visual_resources(config: Any):
    from core.execution import VisualExecutionResources
    from core.performance import runtime_performance

    return VisualExecutionResources(
        image_workers=config.IMAGE_CPU_WORKERS,
        storage_workers=config.STORAGE_IO_WORKERS,
        storage_download_reserved=config.STORAGE_DOWNLOAD_RESERVED_WORKERS,
        metrics=runtime_performance,
    )


def owner_resources(config: Any):
    from core.execution import OwnerExecutionResources
    from core.performance import runtime_performance

    return OwnerExecutionResources(
        search_workers=config.MILVUS_SEARCH_WORKERS,
        write_workers=config.MILVUS_WRITE_WORKERS,
        query_batch_size=config.MILVUS_QUERY_BATCH_SIZE,
        query_wait_ms=config.MILVUS_QUERY_WAIT_MS,
        query_capacity=config.MILVUS_QUERY_CAPACITY,
        metrics=runtime_performance,
    )
