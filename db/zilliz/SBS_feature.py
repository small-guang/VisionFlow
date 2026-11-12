"""Body-aware SBS person ReID gallery."""

from __future__ import annotations

from collections.abc import Collection
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import partial, wraps
import threading
import uuid
from typing import Any

from loguru import logger
from pymilvus import DataType

from core.algorithm_config import algorithm_config
from core.thread_timing import gather_timed_threads

from utils.reid.person_body import (
    gallery_body_types_for_query,
    gallery_body_value,
    person_sbs_threshold,
)
from utils.reid.prototype_policy import (
    PROTOTYPE_ACTION_FROZEN,
    PROTOTYPE_ACTION_REPLACE,
    plan_prototype_update,
)
from utils.reid.reid_models import (
    REID_SOURCE_MILVUS,
    ReidCandidate,
    normalize_face_id,
)

from .person_feature import PersonVectorGalleryBase
from core.execution import run_blocking


PERSON_REID_SEARCH_GROUP_LIMIT_PER_BODY_TYPE = 50  # 每种图库身型最多召回的 ReID 分组数
PERSON_REID_SEARCH_GROUP_SIZE = 3  # 每个 ReID 分组最多返回的原型数
PERSON_BODY_COMPLETENESS_MAX_LENGTH = 32  # bodyCompleteness 字段最大字符数
PERSON_CROP_URL_MAX_LENGTH = 512  # cropUrl 字段最大字符数
SBS_REQUIRED_FIELDS = frozenset(
    {
        "id",
        "faceId",
        "reId",
        "bodyCompleteness",
        "cropUrl",
        "qualityScore",
        "embedding",
        "deviceId",
        "trackId",
        "keyframeId",
        "personRect",
        "timestamp",
    }
)


@dataclass
class _CandidateWriteState:
    lock: Any = field(default_factory=threading.Lock)
    revision: int = 0
    writer_depth: dict[int, int] = field(default_factory=dict)


_CANDIDATE_STATE_INIT_LOCK = threading.Lock()


def _track_candidate_write(method):
    """在真正执行写入的线程中追踪整个调用，包括等待写互斥锁。"""

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


class SbsReidGallery(PersonVectorGalleryBase):
    COLLECTION_NAME = "alogSBS"
    EMBEDDING_DIM = 2048

    def __init__(
        self, *, collection_name=None, search_executor=None,
        write_executor=None, config=None,
    ):
        self._candidate_state = _CandidateWriteState()
        self._prototype_update_lock = threading.RLock()
        super().__init__(
            collection_name=collection_name, search_executor=search_executor,
            write_executor=write_executor, config=config,
        )

    def _candidate_write_state(self) -> _CandidateWriteState:
        state = getattr(self, "_candidate_state", None)
        if state is None:
            # 兼容通过 __new__ 构造的离线测试；并发首次访问也只创建一份状态。
            with _CANDIDATE_STATE_INIT_LOCK:
                state = getattr(self, "_candidate_state", None)
                if state is None:
                    state = _CandidateWriteState()
                    self._candidate_state = state
        return state

    def candidate_revision(self) -> tuple[int, int]:
        """返回本实例的 (revision, active_writers)，不等待远程写入锁。

        版本只描述本进程写入是否与预取重叠，不承诺 Milvus 跨进程可见性。
        异步等待者取消后，实际写线程退出前 active_writers 仍非零。
        """
        state = self._candidate_write_state()
        with state.lock:
            return state.revision, len(state.writer_depth)

    @_track_candidate_write
    def insert_embeddings(self, keyframeId, embedding, deviceId, imagePath, timestamp):
        return super().insert_embeddings(
            keyframeId, embedding, deviceId, imagePath, timestamp
        )

    @_track_candidate_write
    def _insert_person_columns(self, field_names, columns):
        return super()._insert_person_columns(field_names, columns)

    @staticmethod
    def _insert_result_confirms_write(insert_result: Any) -> bool:
        if insert_result is None:
            return False

        if isinstance(insert_result, dict):
            insert_count = insert_result.get("insert_count")
            if insert_count is not None:
                try:
                    return int(insert_count) > 0
                except (TypeError, ValueError):
                    return False
            primary_keys = insert_result.get("primary_keys")
            if primary_keys is None:
                primary_keys = insert_result.get("ids")
        else:
            insert_count = getattr(insert_result, "insert_count", None)
            if insert_count is not None:
                try:
                    return int(insert_count) > 0
                except (TypeError, ValueError):
                    return False
            primary_keys = getattr(insert_result, "primary_keys", None)
            if primary_keys is None:
                primary_keys = getattr(insert_result, "ids", None)

        if primary_keys is None:
            return False
        if isinstance(primary_keys, (str, bytes)):
            return bool(primary_keys)
        try:
            return any(True for _ in primary_keys)
        except TypeError:
            return bool(primary_keys)

    def _create_schema(self):
        schema = self._create_person_schema_prefix(caller_generated_id=True)
        schema.add_field(
            "bodyCompleteness",
            DataType.VARCHAR,
            max_length=PERSON_BODY_COMPLETENESS_MAX_LENGTH,
            description="person body completeness",
        )
        schema.add_field(
            "cropUrl",
            DataType.VARCHAR,
            max_length=PERSON_CROP_URL_MAX_LENGTH,
            description="person crop OSS object name",
        )
        schema.add_field(
            "qualityScore",
            DataType.FLOAT,
            description="person quality score at enrollment",
        )
        self._complete_person_schema(schema)
        if not self._create_collection_if_missing(schema):
            self._validate_existing_schema()

    def _validate_existing_schema(self):
        describe_collection = getattr(self.client, "describe_collection", None)
        if not callable(describe_collection):
            raise RuntimeError(
                f"Cannot validate existing Milvus collection {self.collection_name!r}: "
                "client.describe_collection is unavailable"
            )
        description = describe_collection(collection_name=self.collection_name)
        fields = description.get("fields") if isinstance(description, dict) else None
        if fields is None and isinstance(description, dict):
            schema = description.get("schema")
            if isinstance(schema, dict):
                fields = schema.get("fields")
        fields = fields or []
        field_names = {
            str(field.get("name") or field.get("field_name") or "")
            for field in fields
            if isinstance(field, dict)
        }
        missing_fields = sorted(SBS_REQUIRED_FIELDS - field_names)
        unexpected_fields = sorted(field_names - SBS_REQUIRED_FIELDS)
        if missing_fields or unexpected_fields:
            raise RuntimeError(
                f"Milvus collection {self.collection_name!r} does not match the "
                f"fresh alogSBS schema; missing={missing_fields}, "
                f"unexpected={unexpected_fields}. Stop workers and recreate "
                "alogSBS. Other collections are not modified."
            )
        id_field = next(
            (
                field
                for field in fields
                if isinstance(field, dict)
                and str(field.get("name") or field.get("field_name") or "") == "id"
            ),
            None,
        )
        raw_id_type = (id_field or {}).get("type") or (id_field or {}).get(
            "data_type"
        )
        id_type = str(raw_id_type or "").upper()
        collection_auto_id = bool(
            description.get("auto_id") if isinstance(description, dict) else False
        )
        field_auto_id = bool((id_field or {}).get("auto_id", False))
        id_is_varchar = raw_id_type == DataType.VARCHAR or "VARCHAR" in id_type
        if not id_is_varchar or collection_auto_id or field_auto_id:
            raise RuntimeError(
                "Milvus collection 'alogSBS' requires a VARCHAR(36) caller-generated "
                "primary key with auto_id=false."
            )

    @_track_candidate_write
    def _insert_prototype(
        self,
        faceId,
        reId,
        bodyCompleteness,
        cropUrl,
        embedding,
        deviceId,
        trackId,
        keyframeId,
        personRect,
        timestamp,
        qualityScore,
        prototypeId=None,
    ):
        batch_size = len(faceId) if isinstance(faceId, (list, tuple)) else 1

        def batch_value(value):
            if batch_size == 1:
                return value
            if isinstance(value, (list, tuple)):
                return value
            return [value for _ in range(batch_size)]

        # Kafka trackId is intentionally ignored.  The SBS contract requires 0.
        fixed_track_ids = 0 if batch_size == 1 else [0 for _ in range(batch_size)]
        if isinstance(prototypeId, (list, tuple)):
            prototype_ids = [
                str(value or uuid.uuid4()) for value in prototypeId
            ]
        elif prototypeId:
            prototype_ids = [str(prototypeId) for _ in range(batch_size)]
        else:
            prototype_ids = [str(uuid.uuid4()) for _ in range(batch_size)]
        records = self._records_from_columns(
            [
                "faceId",
                "reId",
                "bodyCompleteness",
                "cropUrl",
                "qualityScore",
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
                bodyCompleteness,
                batch_value(cropUrl),
                batch_value(qualityScore),
                embedding,
                deviceId,
                fixed_track_ids,
                keyframeId,
                personRect,
                timestamp,
            ],
        )
        for index, record in enumerate(records):
            record["id"] = prototype_ids[index]
            record["bodyCompleteness"] = gallery_body_value(
                record.get("bodyCompleteness")
            )
            record["trackId"] = 0
            face_id = normalize_face_id(record.get("faceId"))
            if not face_id:
                raise ValueError("faceId is required for alogSBS")
            record["faceId"] = face_id
            record["reId"] = str(record.get("reId") or "")
            record["cropUrl"] = str(record.get("cropUrl") or "")
            if record.get("qualityScore") is None:
                raise ValueError("qualityScore is required")
            record["qualityScore"] = float(record["qualityScore"])
        return self.client.insert(
            collection_name=self.collection_name,
            data=records,
        )

    async def asearch_similar_person_candidates_with_hit_info(
        self,
        embedding,
        query_body_completeness,
        device_ids=None,
        face_id=None,
        excluded_reids: Collection[str] | None = None,
        query_timestamp: int | None = None,
    ):
        allowed_body_types = gallery_body_types_for_query(
            query_body_completeness
        )
        if not allowed_body_types:
            return [], 0.0, "", ""

        base_filter = self._build_person_reid_base_filter(
            device_ids,
            face_id,
            excluded_reids,
            query_timestamp,
        )
        group_results = await gather_timed_threads(
            "sbs_search",
            *(
                partial(
                    self._search_single_gallery_body_type,
                    embedding,
                    query_body_completeness,
                    gallery_body_type,
                    base_filter,
                )
                for gallery_body_type in allowed_body_types
            ),
            executor=getattr(self, "search_executor", None),
        )
        return self._merge_person_reid_group_results(group_results)

    @staticmethod
    def _build_person_reid_base_filter(
        device_ids=None,
        face_id=None,
        excluded_reids: Collection[str] | None = None,
        query_timestamp: int | None = None,
    ):
        reference_timestamp = int(
            query_timestamp
            if query_timestamp is not None
            else datetime.now().timestamp() * 1000
        )
        query_window_start = int(
            reference_timestamp
            - timedelta(
                days=algorithm_config.reid.classification.query_window_days
            ).total_seconds()
            * 1000
        )
        expr = f"timestamp >= {query_window_start}"
        if device_ids:
            device_expr = " || ".join(
                [f'deviceId == "{device_id}"' for device_id in device_ids]
            )
            expr = f"({expr}) && ({device_expr})"
        normalized_face_id = normalize_face_id(face_id)
        if normalized_face_id:
            escaped_face_id = normalized_face_id.replace("\\", "\\\\").replace(
                '"', '\\"'
            )
            expr = f'({expr}) && faceId == "{escaped_face_id}"'
        normalized_excluded_reids = sorted(
            {
                str(reid).strip()
                for reid in (excluded_reids or ())
                if str(reid).strip()
            }
        )
        if normalized_excluded_reids:
            excluded_values = ", ".join(
                json.dumps(reid) for reid in normalized_excluded_reids
            )
            expr = f"({expr}) && reId not in [{excluded_values}]"
        return expr

    def _search_single_gallery_body_type(
        self,
        embedding,
        query_body_completeness,
        gallery_body_type,
        base_filter,
    ):
        expr = (
            f"({base_filter}) && bodyCompleteness == "
            f"{json.dumps(gallery_body_type)}"
        )

        results = self.client.search(
            collection_name=self.collection_name,
            data=[self._to_milvus_value(embedding)],
            anns_field="embedding",
            search_params=self.search_params,
            limit=PERSON_REID_SEARCH_GROUP_LIMIT_PER_BODY_TYPE,
            filter=expr,
            output_fields=[
                "id",
                "reId",
                "faceId",
                "bodyCompleteness",
                "deviceId",
            ],
            group_by_field="reId",
            group_size=PERSON_REID_SEARCH_GROUP_SIZE,
            strict_group_size=False,
            # pending 移除后，后续候选查询必须看到已确认持久化的原型。
            consistency_level="Strong",
        )

        logger.debug(
            "Search results for gallery body type {}: {}",
            gallery_body_type,
            results,
        )
        if not results or not results[0]:
            return [], []

        raw_hits = []
        candidates = []
        threshold = person_sbs_threshold(
            query_body_completeness,
            gallery_body_type,
        )
        for hit in results[0]:
            entity = self._hit_entity(hit)
            hit_body_type = str(entity.get("bodyCompleteness") or "")
            if hit_body_type != gallery_body_type:
                continue
            raw_hits.append(hit)
            score = self._hit_score(hit)
            candidates.append(
                ReidCandidate(
                    reid=str(entity.get("reId") or ""),
                    score=score,
                    source=REID_SOURCE_MILVUS,
                    threshold=threshold,
                    body_completeness=hit_body_type,
                    face_id=str(entity.get("faceId") or "0"),
                    device_id=str(entity.get("deviceId") or ""),
                    prototype_id=(
                        str(entity["id"]) if entity.get("id") is not None else None
                    ),
                )
            )
        return raw_hits, candidates

    def _merge_person_reid_group_results(self, group_results):
        raw_hits = []
        candidates = []
        for group_raw_hits, group_candidates in group_results:
            raw_hits.extend(group_raw_hits)
            candidates.extend(group_candidates)

        if not raw_hits:
            return [], 0.0, "", ""

        # max 在分数相同时保留先出现的结果，因此平局按兼容矩阵顺序稳定选择。
        highest_hit = max(raw_hits, key=self._hit_score)
        highest_score = self._hit_score(highest_hit)
        highest_reid = str(
            self._hit_entity(highest_hit).get("reId") or ""
        )
        return (
            candidates,
            highest_score,
            highest_reid,
            REID_SOURCE_MILVUS if highest_reid else "",
        )

    async def aapply_prototype_updates(self, batch):
        """Apply bounded prototype decisions and return per-item write results."""
        return await run_blocking(
            getattr(self, "write_executor", None), self.apply_prototype_updates, batch,
        )

    @_track_candidate_write
    def apply_prototype_updates(self, batch):
        lock = getattr(self, "_prototype_update_lock", None)
        if lock is None:
            with _CANDIDATE_STATE_INIT_LOCK:
                lock = getattr(self, "_prototype_update_lock", None)
                if lock is None:
                    lock = threading.RLock()
                    self._prototype_update_lock = lock
        with lock:
            outcomes = []
            for item in batch:
                try:
                    outcomes.append(self._apply_prototype_update(item))
                except Exception as error:
                    logger.error(
                        "SBS prototype update will be retried: reid={}, error={}",
                        item.get("reId"),
                        error,
                    )
                    outcomes.append(
                        {
                            "item": item,
                            "action": "retry_error",
                            "reason": "storage_error",
                            "retry": True,
                            "persisted": False,
                            "error": str(error),
                            "insert_result": None,
                        }
                    )
            return outcomes

    @_track_candidate_write
    def _apply_prototype_update(self, item: dict[str, Any]):
        reid = str(item.get("reId") or "").strip()
        if not reid:
            return {
                "item": item,
                "action": "reject_invalid_reid",
                "reason": "empty_reid",
                "persisted": False,
                "insert_result": None,
            }
        item["cropUrl"] = str(item.get("cropUrl") or "")
        prototype_id = str(item.get("prototypeId") or "").strip()
        if not prototype_id:
            prototype_id = str(uuid.uuid4())
            item["prototypeId"] = prototype_id

        existing_records = self.client.query(
            collection_name=self.collection_name,
            filter=f"reId == {json.dumps(reid)}",
            output_fields=[
                "id",
                "faceId",
                "reId",
                "bodyCompleteness",
                "cropUrl",
                "embedding",
                "deviceId",
                "trackId",
                "keyframeId",
                "personRect",
                "timestamp",
                "qualityScore",
            ],
            limit=100,
            consistency_level="Strong",
        )
        if any(
            str(record.get("id") or "") == prototype_id
            for record in existing_records
        ):
            return {
                "item": item,
                "action": "already_persisted",
                "reason": "prototype_id_exists",
                "retry": False,
                "persisted": True,
                "insert_result": {"primary_keys": [prototype_id]},
            }
        plan = plan_prototype_update(existing_records, item)
        if not plan.writes_vector:
            return {
                "item": item,
                "action": plan.action,
                "reason": plan.reason,
                # A sibling prototype can have won a prior/reordered retry.
                # The ReID is durable even though this observation is skipped.
                "persisted": bool(existing_records),
                "insert_result": None,
            }

        if plan.action == PROTOTYPE_ACTION_REPLACE and plan.replace_id is not None:
            # Enforce the hard cap even when deletion fails. Losing one redundant
            # prototype temporarily is safer than allowing an unbounded class.
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    ids=[plan.replace_id],
                )
            except Exception as error:
                logger.error(
                    "SBS replacement cancelled because old prototype deletion "
                    "failed: reid={}, old_id={}, error={}",
                    reid,
                    plan.replace_id,
                    error,
                )
                return {
                    "item": item,
                    "action": PROTOTYPE_ACTION_FROZEN,
                    "reason": "replacement_delete_failed",
                    "replace_id": plan.replace_id,
                    "cleanup_error": str(error),
                    "persisted": bool(existing_records),
                    "insert_result": None,
                }

        insert_result = self._insert_prototype(
            faceId=item.get("faceId") or "0",
            reId=reid,
            bodyCompleteness=item.get("bodyCompleteness"),
            cropUrl=item.get("cropUrl") or "",
            embedding=item.get("embedding"),
            deviceId=item.get("deviceId") or "",
            trackId=0,
            keyframeId=item.get("pictureVectorId"),
            personRect=item.get("personRect") or {},
            timestamp=item.get("timestamp"),
            qualityScore=item["qualityScore"],
            prototypeId=prototype_id,
        )
        if not self._insert_result_confirms_write(insert_result):
            raise RuntimeError(
                "Milvus insert returned without confirming an SBS write"
            )
        return {
            "item": item,
            "action": plan.action,
            "reason": plan.reason,
            "replace_id": plan.replace_id,
            "cleanup_error": "",
            "persisted": True,
            "insert_result": {
                "insert_count": 1,
                "primary_keys": [prototype_id],
            },
        }
