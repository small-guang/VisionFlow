"""Legacy-schema TBPS/CLIP person retrieval gallery."""

from __future__ import annotations

from .person_feature import PersonVectorGalleryBase
from core.execution import run_blocking


class TbpsGallery(PersonVectorGalleryBase):
    COLLECTION_NAME = "alogTBPS"
    EMBEDDING_DIM = 512

    def _create_schema(self):
        schema = self._create_person_schema_prefix()
        self._complete_person_schema(schema)
        self._create_collection_if_missing(schema)

    async def ainsert_embeddings(
        self,
        faceId,
        reId,
        embedding,
        deviceId,
        trackId,
        keyframeId,
        personRect,
        timestamp,
    ):
        return await run_blocking(
            getattr(self, "write_executor", None),
            self.insert_embeddings,
            faceId,
            reId,
            embedding,
            deviceId,
            trackId,
            keyframeId,
            personRect,
            timestamp,
        )

    def insert_embeddings(
        self,
        faceId,
        reId,
        embedding,
        deviceId,
        trackId,
        keyframeId,
        personRect,
        timestamp,
    ):
        return self._insert_person_columns(
            [
                "faceId",
                "reId",
                "embedding",
                "deviceId",
                "trackId",
                "keyframeId",
                "personRect",
                "timestamp",
            ],
            [
                faceId,
                reId,
                embedding,
                deviceId,
                trackId,
                keyframeId,
                personRect,
                timestamp,
            ],
        )
