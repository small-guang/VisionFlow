"""Shared queue plumbing for independently managed person feature buffers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .background_writer import WriteQueue


PersonVectorInsertedCallback = Callable[
    [list[dict[str, Any]], Any, str, str],
    Awaitable[None],
]


class PersonFeatureBufferBase:
    TARGET_TYPE: str

    def __init__(
        self,
        *,
        vector_db: Any,
        max_batch_size: int,
        on_vectors_inserted: PersonVectorInsertedCallback,
    ):
        self.vector_db = vector_db
        self.max_batch_size = max_batch_size
        self.on_vectors_inserted = on_vectors_inserted
        self.queue = WriteQueue(unit="vector_item")
        self.request_flush: Callable[[], None] | None = None

    async def _drain_batch(self) -> list[dict[str, Any]]:
        batch = []
        while not self.queue.empty() and len(batch) < self.max_batch_size:
            batch.append(await self.queue.get())
        return batch

    async def _requeue(self, batch: list[dict[str, Any]]) -> None:
        self.queue.requeue(batch)

    async def _flush_if_full(self) -> None:
        if self.queue.qsize() < self.max_batch_size:
            return
        if self.request_flush is not None:
            self.request_flush()
        else:
            # Keep standalone buffers and existing tools synchronous until the
            # owning FeatureBuffer has explicitly started its background tasks.
            await self.flush()

    async def _notify_inserted(
        self,
        batch: list[dict[str, Any]],
        insert_result: Any,
    ) -> None:
        await self.on_vectors_inserted(
            batch,
            insert_result,
            self.TARGET_TYPE,
            self.vector_db.collection_name,
        )
