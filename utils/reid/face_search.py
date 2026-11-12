"""Face 候选预取与锁内复核；身份分配和登记仍由调用方负责。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.face_query_observability import (
    current_face_identity_metrics, face_identity_locked_query_scope,
    face_identity_lock_metrics_scope,
)
from core.performance import runtime_performance

from .face_matching import FaceSampleMatch, resolve_face_matches

def _measure_identity_locked(operation: str):
    metrics = current_face_identity_metrics()
    if metrics is not None:
        return metrics.measure(operation)
    return nullcontext()


@dataclass(frozen=True)
class FaceCandidateState:
    gallery_revision: int
    pending_removal_revision: int


@dataclass(frozen=True)
class FaceCandidatePrefetch:
    """单次请求的远程候选；不能直接作为身份提交结果。"""

    state: FaceCandidateState
    embedding: np.ndarray
    device_ids: tuple | None
    milvus: tuple[FaceSampleMatch, ...]


class FaceSearchService:
    def __init__(
        self, *, face_embedding_db: Any, feature_buffer: Any = None,
        pending_faces: Any = None,
    ) -> None:
        self.face_embedding_db = face_embedding_db
        # 旧构造仍可传 facade；运行时只依赖 pending 读取与版本端口。
        self.pending_faces = pending_faces if pending_faces is not None else feature_buffer

    @property
    def feature_buffer(self):
        """旧注入入口的别名；新调用方使用 pending_faces。"""
        return self.pending_faces

    @feature_buffer.setter
    def feature_buffer(self, value):
        self.pending_faces = value

    def snapshot_candidate_state(self) -> FaceCandidateState | None:
        """缺少版本接口或本实例真实写线程活跃时，不启用锁外预取。"""
        gallery_state = getattr(self.face_embedding_db, "candidate_revision", None)
        pending_state = getattr(self.pending_faces, "pending_face_removal_revision", None)
        if not callable(gallery_state) or not callable(pending_state):
            return None
        revision, active_writers = gallery_state()
        if active_writers:
            return None
        return FaceCandidateState(revision, pending_state())

    async def prefetch(
        self,
        embedding: np.ndarray,
        *,
        state: FaceCandidateState,
        device_ids: list | None = None,
    ) -> FaceCandidatePrefetch:
        # 保存过滤参数，避免调用者随后修改原 list 而复用不同查询的结果。
        devices = tuple(device_ids) if device_ids is not None else None
        with runtime_performance.measure("face_milvus_search"):
            samples = await self.face_embedding_db.asearch_face_samples(
                embedding, device_ids=list(devices) if devices is not None else None,
            )
        return FaceCandidatePrefetch(state, embedding, devices, tuple(samples))

    async def resolve_prefetched(
        self,
        prefetched: FaceCandidatePrefetch,
        embedding: np.ndarray,
        *,
        threshold: float,
        device_ids: list | None = None,
    ) -> tuple[str | None, float, str] | None:
        """调用者持身份锁；整体 None 表示失效，需要完整锁内查询。

        正常未匹配仍返回 (None, score, raw_id)，不能与预取失效混淆。
        """
        devices = tuple(device_ids) if device_ids is not None else None
        if (prefetched.embedding is not embedding or prefetched.device_ids != devices
                or self.snapshot_candidate_state() != prefetched.state):
            return None
        with _measure_identity_locked("face_identity_locked_prefetch_pending"):
            pending = await self._pending_samples(embedding, device_ids)
        # pending 快照可能等待锁；读取后再次验证。最后检查到解析完成之间
        # 不让出事件循环，避免把旧远程候选与移除后的 pending 合并为假 miss。
        if self.snapshot_candidate_state() != prefetched.state:
            return None
        with _measure_identity_locked("face_identity_locked_prefetch_resolve"), \
                runtime_performance.measure("face_match_resolve"):
            return resolve_face_matches([*prefetched.milvus, *pending], threshold)

    async def search(
        self,
        embedding: np.ndarray,
        threshold: float,
        device_ids: list | None = None,
    ) -> tuple[str | None, float, str]:
        """完整路径保持 pending → Strong Milvus → 合并解析的原有顺序。"""
        with _measure_identity_locked("face_identity_locked_full_pending"):
            pending = await self._pending_samples(embedding, device_ids)
        with _measure_identity_locked("face_identity_locked_full_milvus"), \
                runtime_performance.measure("face_milvus_search"), \
                face_identity_locked_query_scope():
            milvus = await self.face_embedding_db.asearch_face_samples(
                embedding, device_ids=device_ids,
            )
        with _measure_identity_locked("face_identity_locked_full_resolve"), \
                runtime_performance.measure("face_match_resolve"):
            return resolve_face_matches([*milvus, *pending], threshold)

    async def _pending_samples(
        self, embedding: np.ndarray, device_ids: list | None,
    ) -> list[FaceSampleMatch]:
        pending_search = getattr(self.pending_faces, "asearch_pending_face_samples", None)
        if pending_search is None:
            return []
        with runtime_performance.measure("face_pending_search"):
            return await pending_search(embedding, device_ids=device_ids)
