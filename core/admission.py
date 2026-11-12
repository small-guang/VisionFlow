"""接纳与图片所有权；业务顺序和线程执行容量由各自模块维护。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol


class OutputCapacity(Protocol):
    async def wait_for_capacity(self) -> None: ...


class AdmissionController:
    def __init__(self, output: OutputCapacity):
        self._output = output

    async def wait_for_capacity(self) -> None:
        # 输出水位只阻止新输入；已接纳结果继续写出，避免身份/写出互锁。
        await self._output.wait_for_capacity()


@dataclass(eq=False)
class FrameLease:
    image_bytes: int = 0

    def observe_image(self, image, encoded_bytes: bytes) -> None:
        self.image_bytes = int(getattr(image, "nbytes", 0)) + len(encoded_bytes)


class FrameBudget:
    """下载前按帧预留估算容量；独立统计原图字节，不宣称限制整个RSS。

    不在已持有图片时等待扩容：多个帧同时扩容可能互相等待。crop、mask和
    Tensor仍受目标窗口和资源域容量约束，其字节不伪装成精确原图计数。
    """

    def __init__(self, capacity: int, *, memory_bytes: int = 0, reservation_bytes: int = 64 << 20):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("frame capacity must be a positive integer")
        if type(memory_bytes) is not int or memory_bytes < 0:
            raise ValueError("memory budget must be a nonnegative integer")
        if type(reservation_bytes) is not int or reservation_bytes < 1:
            raise ValueError("frame reservation must be a positive integer")
        if memory_bytes and memory_bytes < reservation_bytes:
            raise ValueError("memory budget must cover at least one frame reservation")
        self.capacity = min(capacity, memory_bytes // reservation_bytes) if memory_bytes else capacity
        self.memory_bytes = memory_bytes
        self.reservation_bytes = reservation_bytes
        self._slots = asyncio.Semaphore(self.capacity)
        self._leases: set[FrameLease] = set()
        self._waiting = 0

    @asynccontextmanager
    async def frame(self):
        self._waiting += 1
        try:
            await self._slots.acquire()
        finally:
            self._waiting -= 1
        lease = FrameLease()
        self._leases.add(lease)
        try:
            yield lease
        finally:
            self._leases.remove(lease)
            self._slots.release()

    def queue_snapshot(self) -> dict:
        return {
            "capacity_frames": self.capacity,
            "inflight_frames": len(self._leases),
            "waiting_frames": self._waiting,
            "memory_budget_bytes": self.memory_bytes,
            "reservation_bytes_per_frame": self.reservation_bytes,
            "reserved_bytes": len(self._leases) * self.reservation_bytes,
            "observed_image_bytes": sum(lease.image_bytes for lease in self._leases),
            "images_exceeding_reservation": sum(
                lease.image_bytes > self.reservation_bytes for lease in self._leases
            ),
            "hard_rss_limit": False,
        }
