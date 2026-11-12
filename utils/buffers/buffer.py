"""持久化端口装配与旧接口兼容；业务状态和写出任务由独立对象拥有。"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional
import numpy as np
from aiokafka import AIOKafkaProducer
from core.config import settings

from .pending import IdentityRegistry, FacePendingIndex
from .kafka_publisher import KafkaPublisher
from .vector_records import VectorRecordEmitter
from .vector_writers import FaceVectorWriter, PictureVectorWriter
from .coordinator import OutputCoordinator
from .sbs_feature_buffer import SbsFeatureBuffer
from .tbps_feature_buffer import TbpsFeatureBuffer
from .background_writer import WriteQueue

_UNSET = object()


class FeatureBuffer:
    """兼容 façade；新调用方应注入 face_pending_view/face_features/records 等窄端口。"""

    def __init__(
        self, max_batch_size: int = 100, flush_interval: float = 5.0, *,
        backlog_high_watermark: int = 2000, backlog_low_watermark: int = 1000,
        face_db: Any = _UNSET, picture_db: Any = _UNSET,
        sbs_db: Any = _UNSET, tbps_db: Any = _UNSET,
        producer_factory=None, topics: dict[str, str] | None = None,
        identity_registry: IdentityRegistry | None = None,
    ):
        if not 0 <= backlog_low_watermark < backlog_high_watermark:
            raise ValueError("output backlog must satisfy 0 <= low < high")
        if max_batch_size < 1 or flush_interval <= 0:
            raise ValueError("batch size and flush interval must be positive")
        # 仅兼容显式构造时解析默认 DB；import 本模块不会建立连接。
        dependencies = [face_db, picture_db, sbs_db, tbps_db]
        if any(item is _UNSET for item in dependencies):
            defaults = importlib.import_module("db.zilliz")
            dependencies = [
                getattr(defaults, name) if value is _UNSET else value
                for value, name in zip(dependencies, (
                    "face_embedding_db", "picture_clip_embedding_db",
                    "person_sbs_embedding_db", "person_clip_embedding_db",
                ))
            ]
        face_db, picture_db, sbs_db, tbps_db = dependencies
        self.max_batch_size, self.flush_interval = max_batch_size, flush_interval
        self.backlog_high_watermark = backlog_high_watermark
        self.backlog_low_watermark = backlog_low_watermark
        self.identity_registry = identity_registry if identity_registry is not None else IdentityRegistry()
        self.face_pending_view = self.identity_registry.faces
        self.sbs_pending_view = self.identity_registry.sbs
        if topics is None:
            topics = {
                "face_detection": settings.KAFKA_FACE_DETECTION_RECORD_TOPIC,
                "face_recognition": settings.KAFKA_FACE_RECOGNITION_RECORD_TOPIC,
                "person_detection": settings.KAFKA_PERSON_DETECTION_RECORD_TOPIC,
                "person_recognition": settings.KAFKA_PERSON_RECOGNITION_RECORD_TOPIC,
                "vector_record": settings.KAFKA_VECTOR_RECORD_TOPIC,
            }
        self.records = KafkaPublisher(
            topics=topics, max_batch_size=max_batch_size,
            producer_factory=producer_factory or (lambda: AIOKafkaProducer(**settings.kafka_client_config)),
        )
        self.vector_records = VectorRecordEmitter(self.records, {
            "face": getattr(face_db, "collection_name", ""),
            "picture": getattr(picture_db, "collection_name", ""),
            "person_sbs": getattr(sbs_db, "collection_name", ""),
            "person": getattr(tbps_db, "collection_name", ""),
        })
        self.face_features = FaceVectorWriter(
            vector_db=face_db, pending=self.face_pending_view, emitter=self.vector_records,
            records=self.records, max_batch_size=max_batch_size,
        )
        self.picture_features = PictureVectorWriter(
            vector_db=picture_db, emitter=self.vector_records, max_batch_size=max_batch_size,
        )
        self.sbs = SbsFeatureBuffer(
            vector_db=sbs_db, max_batch_size=max_batch_size,
            on_vectors_inserted=self.vector_records._queue_inserted_person_vectors,
            pending_index=self.identity_registry.sbs,
        )
        self.tbps = TbpsFeatureBuffer(
            vector_db=tbps_db, max_batch_size=max_batch_size,
            on_vectors_inserted=self.vector_records._queue_inserted_person_vectors,
        )
        self.output = OutputCoordinator(
            vector_writers={"face": self.face_features, "picture": self.picture_features,
                            "sbs": self.sbs, "tbps": self.tbps},
            publisher=self.records, flush_interval=flush_interval,
            high_watermark=backlog_high_watermark, low_watermark=backlog_low_watermark,
        )
        # 以下别名只用于现有调用方/离线工具；所有状态仍有唯一所有者。
        self.face_buffer = self.face_features.queue
        self.picture_buffer = self.picture_features.queue
        self.pending_faces = self.face_pending_view.records
        self._pending_lock = self.face_pending_view.lock
        self._face_flush_lock = self.face_buffer.flush_lock
        self._writers = self.output.writers
        for name, queue in self.records.queues.items():
            attribute = name if name == "vector_record" else name + "_record"
            setattr(self, attribute + "_buffer", queue)
            setattr(self, attribute + "_topic", self.records.topics[name])

    @property
    def pending_faces(self):
        return self.face_pending_view.records

    @pending_faces.setter
    def pending_faces(self, records):
        self.face_pending_view.records = records

    @property
    def _kafka_producer(self):
        return self.records.producer

    @_kafka_producer.setter
    def _kafka_producer(self, producer):
        self.records.producer = producer

    def queue_snapshot(self) -> dict[str, Any]:
        snapshot = self.output.queue_snapshot()
        snapshot.update(pending_faces=len(self.pending_faces), pending_sbs=len(self.sbs.pending))
        return snapshot

    async def _add_pending(self, pending, item):
        if pending is self.pending_faces:
            await self.face_pending_view.add(item)
        else:
            # 旧工具可维护辅助字典；不应改变 Face 查询版本。
            async with self._pending_lock:
                item.setdefault("__pendingKey", id(item))
                pending[item["__pendingKey"]] = item

    async def _remove_pending(self, pending, items):
        if pending is self.pending_faces:
            await self.face_pending_view.remove(items)
        else:
            async with self._pending_lock:
                for item in items:
                    pending.pop(item.get("__pendingKey"), None)

    async def _pending_snapshot(self, pending):
        if pending is self.pending_faces:
            return await self.face_pending_view.snapshot()
        async with self._pending_lock:
            return list(pending.values())

    def pending_face_removal_revision(self) -> int:
        return self.face_pending_view.removal_revision()

    _as_vector = staticmethod(FacePendingIndex.as_vector)
    _score_vectors = staticmethod(FacePendingIndex.score_vectors)
    _device_allowed = staticmethod(FacePendingIndex.device_allowed)
    _extract_primary_keys = staticmethod(VectorRecordEmitter._extract_primary_keys)
    _coerce_primary_key_values = staticmethod(VectorRecordEmitter._coerce_primary_key_values)

    async def asearch_pending_faces_with_hit_info(self, embedding: np.ndarray, threshold: float, device_ids: Optional[List[str]]=None):
        return await self.face_pending_view.asearch_pending_faces_with_hit_info(embedding, threshold, device_ids)

    async def asearch_pending_face_samples(self, embedding: np.ndarray, device_ids: Optional[List[str]]=None):
        return await self.face_pending_view.asearch_pending_face_samples(embedding, device_ids)

    async def add_face_feature(self, face_id: str, embedding: np.ndarray, device_id: str, keyframe_id: str, picture_vector_id: int, keyframe_url: str, face_rect: Dict[str, int], face_key_points: List[Dict[str, float]], face_crop_path: str, timestamp: int, ai_id: str, quality_score: float, is_pano_keyframe: bool=False, new_face_recognition_record: Optional[Dict[str, Any]]=None):
        return await self.face_features.add_face_feature(face_id, embedding, device_id, keyframe_id, picture_vector_id, keyframe_url, face_rect, face_key_points, face_crop_path, timestamp, ai_id, quality_score, is_pano_keyframe, new_face_recognition_record)

    async def add_picture_feature(self, keyframe_id: str, picture_vector_id: int, embedding: np.ndarray, device_id: str, image_path: str, timestamp: int, is_pano_keyframe: bool=False):
        return await self.picture_features.add_picture_feature(keyframe_id, picture_vector_id, embedding, device_id, image_path, timestamp, is_pano_keyframe)

    async def add_face_detection_record(self, record: Dict[str, Any]):
        return await self.records.add_face_detection_record(record)

    async def add_face_recognition_record(self, record: Dict[str, Any]):
        return await self.records.add_face_recognition_record(record)

    async def add_person_detection_record(self, record: Dict[str, Any]):
        return await self.records.add_person_detection_record(record)

    async def add_person_recognition_record(self, record: Dict[str, Any]):
        return await self.records.add_person_recognition_record(record)

    async def add_vector_record(self, record: Dict[str, Any]):
        return await self.records.add_vector_record(record)

    async def _queue_vector_records(self, batch: List[Dict[str, Any]], vector_ids: List[Any], target_type: str, vector_db_name: str | None=None):
        return await self.vector_records._queue_vector_records(batch, vector_ids, target_type, vector_db_name)

    async def _queue_inserted_person_vectors(self, batch: List[Dict[str, Any]], insert_result: Any, target_type: str, vector_db_name: str):
        return await self.vector_records._queue_inserted_person_vectors(batch, insert_result, target_type, vector_db_name)

    async def _flush_kafka_records(self, record_buffer: WriteQueue, topic: str, record_name: str):
        return await self.records.flush_queue(record_buffer, topic, record_name)

    async def _flush_faces(self):
        return await self.face_features.flush()

    async def _flush_face_batch(self):
        return await self.face_features._flush_face_batch()

    async def _flush_pictures(self):
        return await self.picture_features.flush()

    async def start(self):
        return await self.output.start()

    async def stop(self):
        return await self.output.stop()

    async def flush_all(self):
        return await self.output.flush_all()

    async def _flush_if_full(self, name: str):
        return await self.output.flush_if_full(name)

    async def wait_for_capacity(self):
        return await self.output.wait_for_capacity()

    async def _start_kafka_producer(self):
        return await self.records.start()

    async def _stop_kafka_producer(self):
        return await self.records.stop()

    async def _flush_face_detection_records(self):
        return await self.records.flush("face_detection")

    async def _flush_face_recognition_records(self):
        return await self.records.flush("face_recognition")

    async def _flush_person_detection_records(self):
        return await self.records.flush("person_detection")

    async def _flush_person_recognition_records(self):
        return await self.records.flush("person_recognition")

    async def _flush_vector_records(self):
        return await self.records.flush("vector_record")

    def _outstanding_items(self):
        return self.output.outstanding_items()

    def _update_capacity(self):
        self.output.update_capacity()

    @property
    def _running(self):
        return self.output._running

    @property
    def _background_enabled(self):
        return self.output._background_enabled

    @property
    def _capacity_available(self):
        return self.output._capacity_available

    @property
    def _capacity_waiters(self):
        return self.output._capacity_waiters

    @property
    def _backpressured(self):
        return self.output._backpressured

    @property
    def _stop_task(self):
        return self.output._stop_task

    @property
    def _flush_task(self):
        return self.output._flush_task


class _LazyFeatureBuffer:
    """旧全局入口只在使用时解析；runtime 可绑定其显式装配的实例。"""

    def __init__(self):
        object.__setattr__(self, "_instance", None)

    def _resolve(self):
        if self._instance is None:
            object.__setattr__(self, "_instance", FeatureBuffer(
                backlog_high_watermark=getattr(settings, "OUTPUT_BACKLOG_HIGH_WATERMARK", 2000),
                backlog_low_watermark=getattr(settings, "OUTPUT_BACKLOG_LOW_WATERMARK", 1000),
            ))
        return self._instance

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __setattr__(self, name, value):
        setattr(self._resolve(), name, value)


feature_buffer = _LazyFeatureBuffer()


def bind_default_feature_buffer(instance: FeatureBuffer | None) -> None:
    """兼容调用方共享当前 runtime；不在模块导入时连接数据库。"""
    object.__setattr__(feature_buffer, "_instance", instance)
