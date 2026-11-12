"""Milvus 写入回执到 Kafka 向量元数据；不拥有数据库和后台任务。"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, List
from loguru import logger
from utils.records.recognition_record_payloads import build_vector_record
from .kafka_publisher import KafkaPublisher


class VectorRecordEmitter:
    def __init__(self, publisher: KafkaPublisher, collection_names: dict[str, str]):
        self.publisher = publisher
        self.collection_names = dict(collection_names)

    @staticmethod
    def _coerce_primary_key_values(values: Any) -> List[Any]:
        if values is None:
            return []
        if isinstance(values, (str, bytes, Mapping)):
            return [values]
        if isinstance(values, Iterable):
            return list(values)
        return [values]

    @classmethod
    def _extract_primary_keys(cls, insert_result: Any) -> List[Any]:
        """Return Milvus auto primary keys from insert result, if available."""
        if insert_result is None:
            return []

        if isinstance(insert_result, dict):
            for key in ("primary_keys", "ids"):
                values = insert_result.get(key)
                if values is not None:
                    return cls._coerce_primary_key_values(values)

        candidates = [
            getattr(insert_result, "primary_keys", None),
            getattr(insert_result, "ids", None),
            getattr(insert_result, "insert_ids", None),
            getattr(getattr(insert_result, "mutation_result", None), "primary_keys", None),
        ]
        for values in candidates:
            if values is None:
                continue
            return cls._coerce_primary_key_values(values)
        return []

    async def _queue_vector_records(
        self,
        batch: List[Dict[str, Any]],
        vector_ids: List[Any],
        target_type: str,
        vector_db_name: str | None = None,
    ):
        if len(vector_ids) != len(batch):
            logger.error(
                f"无法获取完整向量主键，跳过 {target_type} 向量信息 Kafka 消息: "
                f"vector_ids={len(vector_ids)}, batch={len(batch)}"
            )
            return

        if vector_db_name is None:
            vector_db_name = self.collection_names.get(target_type, "")

        for item, vector_id in zip(batch, vector_ids):
            if target_type == "face" and item.get("__faceVectorRecordQueued"):
                continue
            record = build_vector_record(
                vector_id=vector_id,
                embedding=item.get('embedding'),
                keyframe_id=item.get('keyframeId'),
                picture_vector_id=item.get('pictureVectorId'),
                dev_sn=item.get('deviceId', ""),
                timestamp=item.get('timestamp'),
                keyframe_url=item.get('keyframeUrl') or item.get('imagePath') or "",
                target_type=target_type,
                ai_id=item.get('aiId', ""),
                vector_db_name=vector_db_name,
                is_pano_keyframe=item.get('isPanoKeyframe', False),
            )
            if target_type == "face":
                # Unbounded put + marker must precede any cancellable Kafka send.
                self.publisher.queues["vector_record"].put_nowait(record)
                item["__faceVectorRecordQueued"] = True
                await self.publisher.flush_if_full("vector_record")
            else:
                await self.publisher.add_vector_record(record)

    async def _queue_inserted_person_vectors(
        self,
        batch: List[Dict[str, Any]],
        insert_result: Any,
        target_type: str,
        vector_db_name: str,
    ):
        await self._queue_vector_records(
            batch,
            self._extract_primary_keys(insert_result),
            target_type,
            vector_db_name=vector_db_name,
        )

