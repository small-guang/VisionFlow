"""Triton system SHM 固定槽池；异常 RPC 的槽仅在关闭时销毁。"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import tritonclient.utils.shared_memory as shm

from core.bounded_cpu_pool import BoundedCpuPool
from core.performance import runtime_performance


@dataclass(eq=False)
class SystemSharedMemorySlot:
    name: str
    key: str
    handle: Any
    byte_size: int
    registration_attempted: bool = False
    poisoned: bool = False


class SystemSharedMemoryPool:
    """每次借用独占一个 region；不等待空槽，也不扩容或补建 poison 槽。"""

    def __init__(
        self,
        *,
        slots: int,
        slot_bytes: int,
        client_factory: Callable[[], Any],
        config_timeout: float,
        logger: logging.Logger | None = None,
        copy_workers: int | None = None,
    ) -> None:
        if type(slots) is not int or not 1 <= slots <= 64:
            raise ValueError("SHM slots must be an integer between 1 and 64")
        if type(slot_bytes) is not int or slot_bytes < 1024 * 1024:
            raise ValueError("SHM slot_bytes must be an integer of at least 1 MiB")
        if slots * slot_bytes > 512 * 1024 * 1024:
            raise ValueError("SHM regions must use at most 512 MiB")
        if copy_workers is not None and (
            type(copy_workers) is not int or not 1 <= copy_workers <= min(4, slots)
        ):
            raise ValueError("SHM copy_workers must be between 1 and min(4, slots)")
        self._loop = asyncio.get_running_loop()
        self._slot_count = slots
        self._slot_bytes = slot_bytes
        self._copy_workers = min(4, slots) if copy_workers is None else copy_workers
        self._client_factory = client_factory
        self._config_timeout = config_timeout
        self._logger = logger or logging.getLogger(__name__)
        self._generation = uuid4().hex
        self._slots: list[SystemSharedMemorySlot] = []
        self._available: deque[SystemSharedMemorySlot] = deque()
        self._leased: dict[int, SystemSharedMemorySlot] = {}
        self._idle = asyncio.Event()
        self._idle.set()
        self._control_client: Any = None
        self._copy_pool: BoundedCpuPool | None = None
        self._start_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._accepting = False
        self._closing = False

    async def start(self) -> None:
        if self._closing:
            raise RuntimeError("SHM pool is closing or closed")
        if self._start_task is None:
            self._start_task = asyncio.create_task(
                self._initialize(), name="triton-shm-start"
            )
        # 初始化由池持有；某个调用方取消不能中断注册及其清理。
        await asyncio.shield(self._start_task)
        if self._closing:
            raise RuntimeError("SHM pool is closing or closed")

    async def _initialize(self) -> None:
        try:
            stat = os.statvfs("/dev/shm")
            if self._slot_count * self._slot_bytes > stat.f_bavail * stat.f_frsize:
                raise RuntimeError("insufficient free space in /dev/shm for Triton SHM slots")
            self._control_client = self._client_factory()
            for index in range(self._slot_count):
                name = f"alog_{self._generation}_{index}"
                key = f"/{name}"
                handle = shm.create_shared_memory_region(
                    name, key, self._slot_bytes, create_only=True
                )
                slot = SystemSharedMemorySlot(name, key, handle, self._slot_bytes)
                self._slots.append(slot)
                # 即使 register 超时也可能已在服务端生效，清理时仍按 own name 注销。
                slot.registration_attempted = True
                await self._control_client.register_system_shared_memory(
                    name=name, key=key, byte_size=self._slot_bytes,
                    client_timeout=self._config_timeout,
                )
            self._copy_pool = BoundedCpuPool(
                self._copy_workers,
                thread_name_prefix="alog-triton-shm-copy",
                operation_name="triton_shm_copy_run",
                queue_operation_name="triton_shm_copy_queue",
            )
            if not self._closing:
                self._available.extend(self._slots)
                self._accepting = True
        except BaseException:
            try:
                await self._cleanup_regions()
            except BaseException:
                self._logger.exception("failed to clean up Triton SHM initialization")
            raise

    def borrow(self) -> SystemSharedMemorySlot | None:
        if not self._accepting or not self._available:
            return None
        slot = self._available.popleft()
        self._leased[id(slot)] = slot
        self._idle.clear()
        return slot

    async def copy(self, slot: SystemSharedMemorySlot, array: np.ndarray) -> None:
        if self._leased.get(id(slot)) is not slot or slot.poisoned:
            raise RuntimeError("SHM slot is not an active healthy lease")
        if array.nbytes > slot.byte_size:
            raise ValueError("input exceeds the SHM slot size")
        if self._copy_pool is None:
            raise RuntimeError("SHM pool has not started")
        # 已提交的复制在多次取消下也必须排空，之后才能归还或销毁其目标。
        await self._copy_pool.submit(
            shm.set_shared_memory_region, slot.handle, [array],
            metrics=runtime_performance, drain_on_cancel=True,
        )

    async def reregister_missing(self, slot: SystemSharedMemorySlot) -> bool:
        """仅供明确被服务端拒绝的当前租约修复注册；不写入或复用隔离槽。"""
        if self._leased.get(id(slot)) is not slot or slot.poisoned:
            raise RuntimeError("SHM slot is not an active healthy lease")
        runtime_performance.record("triton_shm_reregister_attempt", 0)
        try:
            repaired = await self._reregister_missing(slot)
        except BaseException as exc:
            runtime_performance.record("triton_shm_reregister_failed", 0)
            self._logger.warning(
                "Triton SHM registration repair failed: region=%s error=%s",
                slot.name, type(exc).__name__,
            )
            raise
        if repaired:
            runtime_performance.record("triton_shm_reregistered", 0)
            self._logger.warning("Triton SHM region re-registered: region=%s", slot.name)
        else:
            runtime_performance.record("triton_shm_reregister_failed", 0)
            self._logger.warning(
                "Triton SHM registration repair declined: region=%s closing=%s",
                slot.name, self._closing,
            )
        return repaired

    async def _reregister_missing(self, slot: SystemSharedMemorySlot) -> bool:
        if self._closing:
            return False
        # 查询全部注册仅用于确认自己的 name 缺失；绝不注销或覆盖其他客户端。
        status = await self._control_client.get_system_shared_memory_status(
            as_json=True, client_timeout=self._config_timeout,
        )
        if (not isinstance(status, dict) or set(status) - {"regions"}
                or not isinstance(status.get("regions", {}), dict)):
            raise RuntimeError("invalid Triton system shared memory status")
        if self._closing or slot.name in status.get("regions", {}):
            return False
        # 当前推理已明确因 name 缺失被拒绝；租约与像素始终不变。
        # register 必须真实结束后才释放租约，避免取消/关闭抢先注销后又注册。
        registration = asyncio.create_task(self._control_client.register_system_shared_memory(
            name=slot.name, key=slot.key, byte_size=slot.byte_size,
            client_timeout=self._config_timeout,
        ))
        try:
            await asyncio.shield(registration)
        except asyncio.CancelledError:
            while not registration.done():
                try:
                    await asyncio.shield(registration)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if not registration.cancelled():
                registration.exception()
            raise
        return not self._closing

    def poison(self, slot: SystemSharedMemorySlot) -> None:
        if self._leased.get(id(slot)) is not slot:
            raise RuntimeError("SHM slot is not leased by this pool")
        if not slot.poisoned:
            slot.poisoned = True
            runtime_performance.record("triton_shm_poisoned", 0)

    def release(self, slot: SystemSharedMemorySlot) -> None:
        if self._leased.pop(id(slot), None) is not slot:
            raise RuntimeError("SHM slot is not leased by this pool")
        if self._accepting and not slot.poisoned:
            self._available.append(slot)
        if not self._leased:
            self._idle.set()

    def stop_borrowing(self) -> None:
        self._closing = True
        self._accepting = False
        self._available.clear()

    async def close(self) -> None:
        self.stop_borrowing()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._finish_close(), name="triton-shm-close"
            )
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        if self._start_task is not None:
            await asyncio.gather(self._start_task, return_exceptions=True)
        await self._idle.wait()
        try:
            if self._copy_pool is not None:
                await self._copy_pool.close()
        finally:
            await self._cleanup_regions()

    async def _cleanup_regions(self) -> None:
        errors: list[BaseException] = []
        for slot in self._slots:
            if slot.registration_attempted and self._control_client is not None:
                try:
                    await self._control_client.unregister_system_shared_memory(
                        name=slot.name, client_timeout=self._config_timeout,
                    )
                except BaseException as exc:
                    errors.append(exc)
            try:
                # 不截断或重用对象；服务端已持有的映射可安全读到旧内容。
                shm.destroy_shared_memory_region(slot.handle)
            except BaseException as exc:
                errors.append(exc)
        self._slots.clear()
        self._available.clear()
        if self._control_client is not None:
            try:
                await self._control_client.close()
            except BaseException as exc:
                errors.append(exc)
            self._control_client = None
        if errors:
            raise RuntimeError(f"failed to clean up {len(errors)} Triton SHM resource(s)") from errors[0]
