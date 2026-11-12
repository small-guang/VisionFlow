"""TBPS/CLIP queue and Milvus flush lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
import numpy as np

from .person_feature_buffer import (
    PersonFeatureBufferBase,
    PersonVectorInsertedCallback,
)


class TbpsFeatureBuffer(PersonFeatureBufferBase):
    TARGET_TYPE = "person"

    def __init__(
        self,
        *,
        vector_db: Any,
        max_batch_size: int,
        on_vectors_inserted: PersonVectorInsertedCallback,
    ):
        super().__init__(
            vector_db=vector_db,
            max_batch_size=max_batch_size,
            on_vectors_inserted=on_vectors_inserted,
        )

    async def add_feature(
        self,
        *,
        face_id: int,
        re_id: int,
        embedding: np.ndarray,
        device_id: str,
        keyframe_id: str,
        picture_vector_id: int,
        keyframe_url: str,
        person_rect: dict[str, int],
        timestamp: int,
        ai_id: str,
        is_pano_keyframe: bool = False,
    ) -> None:
        item = {
            "faceId": face_id,
            "reId": re_id,
            "embedding": embedding,
            "deviceId": device_id,
            "keyframeId": keyframe_id,
            "pictureVectorId": picture_vector_id,
            "keyframeUrl": keyframe_url,
            "personRect": person_rect,
            "timestamp": timestamp,
            "aiId": ai_id,
            "isPanoKeyframe": bool(is_pano_keyframe),
        }
        await self.queue.put(item)
        await self._flush_if_full()

    async def flush(self) -> None:
        async with self.queue.flush_lock:
            batch = await self._drain_batch()
            if not batch:
                return
            try:
                insert_result = await self.vector_db.ainsert_embeddings(
                    faceId=[item["faceId"] for item in batch],
                    reId=[item["reId"] for item in batch],
                    embedding=[item["embedding"] for item in batch],
                    deviceId=[item["deviceId"] for item in batch],
                    trackId=[0 for _ in batch],
                    keyframeId=[item["pictureVectorId"] for item in batch],
                    personRect=[item["personRect"] for item in batch],
                    timestamp=[item["timestamp"] for item in batch],
                )
                logger.info(f"Successfully wrote {len(batch)} TBPS person vectors")
                await self._notify_inserted(batch, insert_result)
                self.queue.complete(len(batch))
            except asyncio.CancelledError:
                await self._requeue(batch)
                raise
            except Exception as error:
                logger.error(f"Failed to write TBPS person vectors: {error}")
                await self._requeue(batch)
