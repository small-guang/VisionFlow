from typing import Any, Dict, Iterable, List
import logging

import numpy as np
from pymilvus import DataType, MilvusClient

from core.config import settings
from core.execution import run_blocking


class MilvusVectorDB:
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 1000},
    }
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    def __init__(
        self, collection_name="image_collection", dim=512, *,
        search_executor=None, write_executor=None, config=None,
    ):
        self.search_executor = search_executor
        self.write_executor = write_executor
        self.collection_name = collection_name
        self.dim = dim
        config = settings if config is None else config
        self.client = MilvusClient(
            uri=config.milvus_uri,
            token=config.milvus_token,
        )
        try:
            self._create_schema()
            self.load_collection()
        except BaseException:
            # 构造失败对象不会交给 Runtime；本层负责回收已打开的客户端。
            close = getattr(self.client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    logging.getLogger(__name__).warning(
                        "Milvus client cleanup failed: %s", type(error).__name__,
                    )
            raise

    @staticmethod
    def _to_milvus_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _batch_size(primary_key_values: Any) -> int:
        if isinstance(primary_key_values, (list, tuple)):
            return len(primary_key_values)
        return 1

    @classmethod
    def _as_column(cls, value: Any, batch_size: int, is_columnar: bool) -> List[Any]:
        if is_columnar:
            if not isinstance(value, (list, tuple)):
                if batch_size == 1:
                    return [cls._to_milvus_value(value)]
                raise ValueError(
                    f"Column value must contain {batch_size} items, got scalar {type(value).__name__}"
                )
            if len(value) != batch_size:
                raise ValueError(
                    f"Column value length mismatch: expected {batch_size}, got {len(value)}"
                )
            return [cls._to_milvus_value(item) for item in value]
        return [cls._to_milvus_value(value)]

    @classmethod
    def _records_from_columns(cls, field_names: List[str], columns: Iterable[Any]) -> List[Dict[str, Any]]:
        columns = list(columns)
        is_columnar = isinstance(columns[0], (list, tuple))
        batch_size = cls._batch_size(columns[0])
        normalized = [
            cls._as_column(column, batch_size, is_columnar) for column in columns
        ]
        return [
            {field_name: normalized[column_index][row_index] for column_index, field_name in enumerate(field_names)}
            for row_index in range(batch_size)
        ]

    @staticmethod
    def _hit_entity(hit: Dict[str, Any]) -> Dict[str, Any]:
        return hit.get("entity") or hit

    @staticmethod
    def _hit_score(hit: Dict[str, Any]) -> float:
        return float(hit.get("distance", hit.get("score", 0.0)))

    def _embedding_index_params(self):
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="embedding", **self.index_params)
        return index_params

    def _ensure_embedding_index(self):
        indexes = self.client.list_indexes(
            collection_name=self.collection_name,
            field_name="embedding",
        )
        if indexes:
            return
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=self._embedding_index_params(),
        )

    def _create_schema(self):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("keyframeId", DataType.INT64, is_primary=True, description="key frame id")
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim, description="key frame embedding")
        schema.add_field("deviceId", DataType.VARCHAR, max_length=255, is_partition_key=True, description="device id")
        schema.add_field("imagePath", DataType.VARCHAR, max_length=255, description="key frame OSS path")
        schema.add_field("timestamp", DataType.INT64, description="timestamp")
        self.schema = schema

        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=self.schema,
                index_params=self._embedding_index_params(),
                properties={"collection.ttl.seconds": 604800},
            )

    async def ainsert_embeddings(self, keyframeId, embedding, deviceId, imagePath, timestamp):
        return await run_blocking(
            getattr(self, "write_executor", None),
            self.insert_embeddings,
            keyframeId,
            embedding,
            deviceId,
            imagePath,
            timestamp,
        )

    def insert_embeddings(self, keyframeId, embedding, deviceId, imagePath, timestamp):
        records = self._records_from_columns(
            ["keyframeId", "embedding", "deviceId", "imagePath", "timestamp"],
            [keyframeId, embedding, deviceId, imagePath, timestamp],
        )
        return self.client.insert(collection_name=self.collection_name, data=records)

    def load_collection(self):
        self._ensure_embedding_index()
        self.client.load_collection(collection_name=self.collection_name)
