"""身份首次持久化回执与识别记录放行协议。"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

RecognitionReady = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(eq=False)
class PersistenceTicket:
    """记录与持久化回执可任意先到；两者齐备后才允许一次放行。

    on_ready 的确认边界是本地记录队列接纳，生产路径不得在身份提交锁内
    等待 Kafka ACK。失败保留回执可重试，远程写入成功由写入端显式确认。
    """

    persisted: bool = False
    record: dict[str, Any] | None = None
    on_ready: RecognitionReady | None = field(default=None, repr=False)
    released: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def attach(self, record: dict[str, Any], on_ready: RecognitionReady) -> None:
        async with self.lock:
            if self.record is None:
                self.record, self.on_ready = record, on_ready
            await self._release_ready()

    async def mark_persisted(self) -> None:
        async with self.lock:
            self.persisted = True
            await self._release_ready()

    async def _release_ready(self) -> None:
        if self.released or not self.persisted or self.record is None or self.on_ready is None:
            return
        await self.on_ready(self.record)
        self.released = True
