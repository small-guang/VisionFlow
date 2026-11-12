from dataclasses import dataclass, field
from functools import wraps
import json
import math
import threading
from typing import Any
import uuid

import numpy as np
from loguru import logger
from core.algorithm_config import algorithm_config
from db.face_schema import FACE_MAX_CAPACITY, build_face_schema, validate_face_schema
from utils.reid.face_matching import FaceSampleMatch, resolve_face_matches

from .picture_feature import MilvusVectorDB
from core.execution import run_blocking


@dataclass
class _FaceCandidateWriteState:
    lock: Any = field(default_factory=threading.Lock)
    revision: int = 0
    writer_depth: dict[int, int] = field(default_factory=dict)


_CANDIDATE_STATE_INIT_LOCK = threading.Lock()


def _track_candidate_write(method):
    """由真实写线程登记版本，覆盖嵌套调用、写锁等待和异常退出。"""

    @wraps(method)
    def tracked(self, *args, **kwargs):
        state = self._candidate_write_state()
        thread_id = threading.get_ident()
        with state.lock:
            depth = state.writer_depth.get(thread_id, 0)
            state.writer_depth[thread_id] = depth + 1
            if depth == 0:
                state.revision += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            with state.lock:
                depth = state.writer_depth[thread_id]
                if depth == 1:
                    del state.writer_depth[thread_id]
                    state.revision += 1
                else:
                    state.writer_depth[thread_id] = depth - 1

    return tracked


class FaceGallery(MilvusVectorDB):
    def __init__(
        self, collection_name="alogFace", dim=512, *, max_vectors_per_face=None,
        search_executor=None, write_executor=None, config=None, query_coordinator=None,
    ):
        self.max_vectors_per_face = (
            algorithm_config.face.gallery.max_vectors_per_id
            if max_vectors_per_face is None else max_vectors_per_face
        )
        if (
            isinstance(self.max_vectors_per_face, bool)
            or not isinstance(self.max_vectors_per_face, int)
            or not 1 <= self.max_vectors_per_face <= FACE_MAX_CAPACITY
        ):
            raise ValueError(f"max_vectors_per_face must be an integer in [1, {FACE_MAX_CAPACITY}]")
        self.query_coordinator = query_coordinator
        self._candidate_state = _FaceCandidateWriteState()
        self._quality_update_lock = threading.RLock()
        super().__init__(
            collection_name=collection_name, dim=dim,
            search_executor=search_executor, write_executor=write_executor,
            config=config,
        )

    def _candidate_write_state(self) -> _FaceCandidateWriteState:
        state = getattr(self, "_candidate_state", None)
        if state is None:
            # 兼容以 __new__ 构造的离线适配器；并发初始化只创建一份状态。
            with _CANDIDATE_STATE_INIT_LOCK:
                state = getattr(self, "_candidate_state", None)
                if state is None:
                    state = _FaceCandidateWriteState()
                    self._candidate_state = state
        return state

    def candidate_revision(self) -> tuple[int, int]:
        """本实例的 (revision, active_writers)，不等待远程写入锁。

        异步等待者取消后，真实写线程退出前仍保持 active。该状态只用于
        检测本进程写入与候选预取重叠，不提供跨进程或跨 pending 的事务。
        """
        state = self._candidate_write_state()
        with state.lock:
            return state.revision, len(state.writer_depth)

    def _create_schema(self):
        self.schema = build_face_schema(self.client, self.dim)
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=self.schema,
                index_params=self._embedding_index_params(),
            )
        else:
            validate_face_schema(self.client.describe_collection(self.collection_name), self.dim)

    async def ainsert_embeddings(
        self,
        faceId,
        embedding,
        deviceId,
        trackId,
        keyframeId,
        faceRect,
        faceKeyPoints,
        faceCropPath,
        timestamp,
        qualityScore,
        sampleId,
    ):
        return await run_blocking(
            getattr(self, "write_executor", None),
            self.insert_embeddings,
            faceId,
            embedding,
            deviceId,
            trackId,
            keyframeId,
            faceRect,
            faceKeyPoints,
            faceCropPath,
            timestamp,
            qualityScore,
            sampleId,
        )

    @_track_candidate_write
    def insert_embeddings(
        self,
        faceId,
        embedding,
        deviceId,
        trackId,
        keyframeId,
        faceRect,
        faceKeyPoints,
        faceCropPath,
        timestamp,
        qualityScore,
        sampleId,
    ):
        records = self._records_from_columns(
            [
                "faceId",
                "embedding",
                "deviceId",
                "trackId",
                "keyframeId",
                "faceRect",
                "faceKeyPoints",
                "faceCropPath",
                "timestamp",
                "qualityScore",
                "sampleId",
            ],
            [
                faceId,
                embedding,
                deviceId,
                trackId,
                keyframeId,
                faceRect,
                faceKeyPoints,
                faceCropPath,
                timestamp,
                qualityScore,
                sampleId,
            ],
        )
        # Validate the entire batch before any side effects. Retries use the same sampleId.
        for record in records:
            self._validate_record(record)
        with self._quality_update_lock:
            # 仅复用本批、同一写锁内的已确认视图；异常退出后不保留缓存。
            # 不按 faceId 重排，后续样本必须看到原输入顺序中已确认的更新。
            existing_by_face = {}
            outcomes = []
            for record in records:
                face_id = record["faceId"]
                if face_id not in existing_by_face:
                    existing_by_face[face_id] = self._read_existing_samples(face_id)
                outcomes.append(self._update_one(
                    record, existing=existing_by_face[face_id],
                ))
        ids = [outcome["primary_key"] for outcome in outcomes if outcome["primary_key"] is not None]
        return {"ids": ids, "outcomes": outcomes}

    def _validate_record(self, record):
        try:
            quality = float(record["qualityScore"])
        except (TypeError, ValueError):
            raise ValueError("qualityScore is required and must be finite") from None
        if not math.isfinite(quality) or not algorithm_config.face.quality.enroll_min_score <= quality <= 10.0:
            raise ValueError(f"qualityScore must be in [{algorithm_config.face.quality.enroll_min_score}, 10]")
        record["qualityScore"] = quality
        for field in ("faceId", "sampleId"):
            value = str(record.get(field) or "").strip()
            if not value or value == "0" or len(value) > 36:
                raise ValueError(f"{field} must be a nonzero identifier of at most 36 characters")
            record[field] = value
        vector = np.asarray(record["embedding"], dtype=np.float32).flatten()
        if vector.size != self.dim or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 0:
            raise ValueError(f"embedding must contain {self.dim} finite, nonzero vector components")
        record["embedding"] = vector.tolist()
        record["timestamp"] = int(record["timestamp"])
        if not record.get("faceCropPath"):
            raise ValueError("faceCropPath is required")

    @staticmethod
    def _quality_rank(record):
        return float(record["qualityScore"]), int(record["timestamp"])

    def _read_existing_samples(self, face_id):
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            filter=f'faceId == {json.dumps(face_id)}',
            output_fields=["id", "sampleId", "qualityScore", "timestamp"],
            batch_size=1000,
            consistency_level="Strong",
        )
        existing = []
        try:
            while batch := iterator.next():
                existing.extend(batch)
        finally:
            iterator.close()
        return existing

    @_track_candidate_write
    def _update_one(self, record, *, existing=None):
        if existing is None:
            existing = self._read_existing_samples(record["faceId"])
        existing.sort(key=lambda row: (*self._quality_rank(row), row["id"]), reverse=True)
        # A reduced configured limit takes effect when this identity is next observed.
        surplus = existing[self.max_vectors_per_face:]
        for start in range(0, len(surplus), 1000):
            removed = surplus[start:start + 1000]
            result = self.client.delete(
                collection_name=self.collection_name, ids=[row["id"] for row in removed]
            )
            if int(result.get("delete_count", 0)) < len(removed):
                raise RuntimeError("Milvus did not confirm face capacity cleanup")
        if surplus:
            del existing[self.max_vectors_per_face:]

        for row in existing:
            if row["sampleId"] == record["sampleId"]:
                return {"action": "already_persisted", "persisted": True, "primary_key": row["id"]}

        if len(existing) >= self.max_vectors_per_face:
            worst = existing[-1]
            if self._quality_rank(record) <= self._quality_rank(worst):
                return {"action": "skipped", "persisted": True, "primary_key": None}
            record["id"] = worst["id"]
            action = "replaced"
        else:
            # Fixed integer slots retain the existing vectorId payload type and
            # prevent a retry from inventing an additional slot for the identity.
            used_ids = {row["id"] for row in existing}
            for slot in range(self.max_vectors_per_face):
                slot_id = uuid.uuid5(uuid.NAMESPACE_URL, f'{self.collection_name}/{record["faceId"]}/{slot}').int & ((1 << 63) - 1)
                if slot_id not in used_ids:
                    record["id"] = slot_id
                    break
            action = "inserted"
        result = self.client.upsert(collection_name=self.collection_name, data=[record])
        if int(result.get("upsert_count", 0)) != 1:
            raise RuntimeError("Milvus did not confirm the face vector upsert")
        # 只能在数据库确认后前推批内视图。回执丢失或部分失败会抛出，
        # 下一次整批重试重新 Strong 查询，以 sampleId 确认真实持久化状态。
        existing[:] = [row for row in existing if row["id"] != record["id"]]
        existing.append({
            key: record[key] for key in ("id", "sampleId", "qualityScore", "timestamp")
        })
        return {"action": action, "persisted": True, "primary_key": record["id"]}

    async def asearch_similar_faces_with_hit_info(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
        device_ids: list = None,
    ):
        return resolve_face_matches(await self.asearch_face_samples(embedding, device_ids), threshold)

    def search_similar_faces_with_hit_info(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
        device_ids: list = None,
    ):
        return resolve_face_matches(
            self.search_face_samples(embedding, device_ids), threshold
        )

    async def asearch_face_samples(
        self, embedding: np.ndarray, device_ids: list = None
    ) -> list[FaceSampleMatch]:
        if self.query_coordinator is None:
            raise RuntimeError("async face search requires a vector query coordinator")
        query = self._face_search_query(embedding, device_ids)
        if query is None:
            return []
        vector, parameters = query
        hits = await self.query_coordinator.search(self.client, vector, **parameters)
        return self._face_sample_matches(hits)

    def search_face_samples(
        self, embedding: np.ndarray, device_ids: list = None
    ) -> list[FaceSampleMatch]:
        query = self._face_search_query(embedding, device_ids)
        if query is None:
            return []
        vector, parameters = query
        results = self.client.search(data=[vector], **parameters)
        return self._face_sample_matches(results[0] if results else [])

    def _face_search_query(self, embedding: np.ndarray, device_ids: list = None):
        if embedding.ndim == 1:
            vector = embedding.tolist()
        elif embedding.ndim == 2 and embedding.shape[0] == 1:
            vector = embedding[0].tolist()
        else:
            logger.error(
                f"Invalid embedding shape for search: {embedding.shape}. "
                "Expected 1D array or 2D array with 1 row."
            )
            return None

        expr = ""
        if device_ids and isinstance(device_ids, list) and len(device_ids) > 0:
            device_expr_parts = []
            for did in device_ids:
                if isinstance(did, str):
                    device_expr_parts.append(f'deviceId == "{did}"')
                else:
                    device_expr_parts.append(f"deviceId == {did}")
            if device_expr_parts:
                expr = f"({ ' || '.join(device_expr_parts) })"

        logger.debug(
            f"Milvus search_face_samples - expr: '{expr}', "
            f"embedding_shape: {embedding.shape}"
        )

        return vector, {
            "collection_name": self.collection_name,
            "anns_field": "embedding",
            "search_params": self.search_params,
            "limit": 100,
            "filter": expr,
            "output_fields": ["faceId", "deviceId", "sampleId", "faceCropPath"],
            "consistency_level": "Strong",
        }

    def _face_sample_matches(self, hits) -> list[FaceSampleMatch]:
        if not hits:
            logger.debug("No results from Milvus search or empty result set.")
            return []

        return [
            FaceSampleMatch(
                face_id=str(self._hit_entity(hit).get("faceId") or ""),
                score=self._hit_score(hit),
                sample_id=str(self._hit_entity(hit).get("sampleId") or ""),
                crop_path=str(self._hit_entity(hit).get("faceCropPath") or ""),
            )
            for hit in hits
        ]
