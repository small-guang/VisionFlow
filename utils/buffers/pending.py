"""进程内待持久化身份视图；不依赖写队列、数据库或 Kafka。"""
from __future__ import annotations

import asyncio
from collections.abc import Collection
from datetime import datetime, timedelta
import itertools
import threading
from typing import Any, Optional

import numpy as np
from utils.reid.face_matching import FaceSampleMatch, resolve_face_matches
from core.algorithm_config import algorithm_config
from utils.reid.person_body import gallery_body_types_for_query, person_sbs_threshold
from utils.reid.reid_models import REID_SOURCE_BUFFER, ReidCandidate, normalize_face_id


class PendingIndex:
    """持有 pending 样本和移除版本，供提交时复核远程候选。"""

    def __init__(self):
        self.records: dict[int, dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self._keys = itertools.count(1)
        self._revision = 0
        self._revision_lock = threading.Lock()

    async def add(self, item: dict[str, Any]) -> None:
        if "__pendingKey" not in item:
            item["__pendingKey"] = next(self._keys)
        async with self.lock:
            self.records[item["__pendingKey"]] = item

    async def remove(self, items: list[dict[str, Any]]) -> None:
        async with self.lock:
            with self._revision_lock:
                removed = False
                for item in items:
                    key = item.get("__pendingKey")
                    if key is not None and key in self.records:
                        self.records.pop(key)
                        removed = True
                if removed:
                    self._revision += 1

    def removal_revision(self) -> int:
        with self._revision_lock:
            return self._revision

    async def snapshot(self) -> list[dict[str, Any]]:
        async with self.lock:
            return list(self.records.values())

    @staticmethod
    def as_vector(value: Any) -> np.ndarray:
        try:
            return np.asarray(value, dtype=np.float32).flatten()
        except (TypeError, ValueError):
            return np.asarray([], dtype=np.float32)

    @classmethod
    def score_vectors(cls, query: np.ndarray, candidate: Any) -> float | None:
        candidate = cls.as_vector(candidate)
        if query.size == 0 or candidate.size == 0 or query.shape != candidate.shape:
            return None
        return float(np.dot(query, candidate))

    @staticmethod
    def device_allowed(record: dict[str, Any], device_ids: list[str] | None) -> bool:
        return not device_ids or str(record.get("deviceId") or "") in {
            str(device_id) for device_id in device_ids
        }


class FacePendingIndex(PendingIndex):
    """人脸 pending 查询端口；写入完成前的样本仍参与身份解析。"""

    def pending_face_removal_revision(self) -> int:
        return self.removal_revision()

    async def asearch_pending_face_samples(
        self, embedding: np.ndarray, device_ids: list[str] | None = None,
    ) -> list[FaceSampleMatch]:
        query = self.as_vector(embedding)
        samples = []
        for record in await self.snapshot():
            if not self.device_allowed(record, device_ids):
                continue
            score = self.score_vectors(query, record.get("embedding"))
            if score is not None:
                samples.append(FaceSampleMatch(
                    face_id=str(record.get("faceId") or ""), score=score,
                    sample_id=str(record.get("sampleId") or ""),
                    crop_path=str(record.get("faceCropPath") or ""),
                ))
        return samples

    async def asearch_pending_faces_with_hit_info(
        self, embedding: np.ndarray, threshold: float,
        device_ids: list[str] | None = None,
    ):
        return resolve_face_matches(
            await self.asearch_pending_face_samples(embedding, device_ids), threshold,
        )


class SbsPendingIndex(PendingIndex):
    """SBS 待写原型查询视图；过滤规则与正式图库候选合并保持一致。"""

    async def asearch_candidates_with_hit_info(
        self,
        embedding: np.ndarray,
        query_body_completeness: str,
        device_ids: Optional[list[str]] = None,
        face_id: Optional[str] = None,
        excluded_reids: Optional[Collection[str]] = None,
        query_timestamp: Optional[int] = None,
    ):
        allowed_body_types = gallery_body_types_for_query(
            query_body_completeness
        )
        if not allowed_body_types:
            return [], 0.0, "", ""

        query = self.as_vector(embedding)
        event_time = int(
            query_timestamp
            if query_timestamp is not None
            else datetime.now().timestamp() * 1000
        )
        records = await self.snapshot()
        normalized_excluded_reids = {
            str(reid).strip()
            for reid in (excluded_reids or ())
            if str(reid).strip()
        }
        query_window_start = int(
            event_time
            - timedelta(
                days=algorithm_config.reid.classification.query_window_days
            ).total_seconds()
            * 1000
        )
        highest_score = 0.0
        highest_reid = ""
        candidates = []
        query_face = normalize_face_id(face_id)
        for record in records:
            reid = str(record.get("reId") or "")
            if reid in normalized_excluded_reids:
                continue
            if not self.device_allowed(record, device_ids):
                continue
            if query_face and normalize_face_id(record.get("faceId")) != query_face:
                continue
            record_time = int(record.get("timestamp") or 0)
            if record_time < query_window_start:
                continue
            gallery_body_type = str(record.get("bodyCompleteness") or "")
            if gallery_body_type not in allowed_body_types:
                continue
            score = self.score_vectors(query, record.get("embedding"))
            if score is None:
                continue
            if score > highest_score:
                highest_score = score
                highest_reid = reid
            threshold = person_sbs_threshold(
                query_body_completeness,
                gallery_body_type,
            )
            if reid:
                candidates.append(
                    ReidCandidate(
                        reid=reid,
                        score=score,
                        source=REID_SOURCE_BUFFER,
                        threshold=threshold,
                        body_completeness=gallery_body_type,
                        face_id=str(record.get("faceId") or "0"),
                        device_id=str(record.get("deviceId") or ""),
                        prototype_id=(
                            str(record.get("prototypeId") or "").strip()
                            or (
                                int(record["__pendingKey"])
                                if record.get("__pendingKey") is not None
                                else None
                            )
                        ),
                    )
                )
        return (
            candidates,
            highest_score,
            highest_reid,
            REID_SOURCE_BUFFER if highest_reid else "",
        )


class IdentityRegistry:
    """每个 Worker runtime 独占的 pending 状态，不拥有提交锁或 I/O。"""

    def __init__(self):
        self.faces = FacePendingIndex()
        self.sbs = SbsPendingIndex()
