"""Shared mechanical helpers for person vector galleries."""

from __future__ import annotations

from pymilvus import DataType

from .picture_feature import MilvusVectorDB


PERSON_GALLERY_TTL_SECONDS = 604800


class PersonVectorGalleryBase(MilvusVectorDB):
    """Common collection and record plumbing without SBS/TBPS policy."""

    COLLECTION_NAME: str
    EMBEDDING_DIM: int

    def __init__(
        self, *, collection_name=None, search_executor=None,
        write_executor=None, config=None,
    ):
        super().__init__(
            collection_name=self.COLLECTION_NAME if collection_name is None else collection_name,
            dim=self.EMBEDDING_DIM,
            search_executor=search_executor,
            write_executor=write_executor,
            config=config,
        )

    def _create_person_schema_prefix(self, *, caller_generated_id: bool = False):
        schema = self.client.create_schema(
            auto_id=not caller_generated_id,
            enable_dynamic_field=False,
        )
        if caller_generated_id:
            schema.add_field(
                "id",
                DataType.VARCHAR,
                max_length=36,
                is_primary=True,
                auto_id=False,
                description="caller-generated prototype UUID",
            )
        else:
            schema.add_field(
                "id",
                DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="auto id",
            )
        schema.add_field(
            "faceId",
            DataType.VARCHAR,
            max_length=36,
            description="face id",
        )
        schema.add_field(
            "reId",
            DataType.VARCHAR,
            max_length=36,
            description="person reid",
        )
        return schema

    def _complete_person_schema(self, schema):
        schema.add_field(
            "embedding",
            DataType.FLOAT_VECTOR,
            dim=self.dim,
            description="person embedding",
        )
        schema.add_field(
            "deviceId",
            DataType.VARCHAR,
            max_length=255,
            is_partition_key=True,
            description="device id",
        )
        schema.add_field("trackId", DataType.INT64, description="track id")
        schema.add_field(
            "keyframeId",
            DataType.INT64,
            description="key frame id",
        )
        schema.add_field(
            "personRect",
            DataType.JSON,
            description="person rect",
        )
        schema.add_field("timestamp", DataType.INT64, description="timestamp")
        return schema

    def _create_collection_if_missing(self, schema) -> bool:
        self.schema = schema
        if self.client.has_collection(self.collection_name):
            return False
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=self._embedding_index_params(),
            properties={"collection.ttl.seconds": PERSON_GALLERY_TTL_SECONDS},
        )
        return True

    def _insert_person_columns(self, field_names, columns):
        records = self._records_from_columns(field_names, columns)
        return self.client.insert(
            collection_name=self.collection_name,
            data=records,
        )
