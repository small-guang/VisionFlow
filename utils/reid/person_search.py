"""I/O orchestration for person ReID candidate search."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.performance import runtime_performance

from .reid_models import (
    REID_SEARCH_MODE_ANONYMOUS_GLOBAL,
    REID_SEARCH_MODE_SAME_FACE,
    PersonReidSearchResult,
    ReidClassResolution,
    normalize_face_id,
)
from .reid_resolution import ReidClassResolver
from .reid_state import REID_STATE_UNASSIGNED


@dataclass(frozen=True)
class ReidSearchAttempt:
    mode: str
    candidate_face_id: str | None


class ReidQueryPlanner:
    def plan(self, face_id: Any) -> tuple[ReidSearchAttempt, ...]:
        normalized_face = normalize_face_id(face_id)
        return (
            ReidSearchAttempt(
                mode=(
                    REID_SEARCH_MODE_SAME_FACE
                    if normalized_face
                    else REID_SEARCH_MODE_ANONYMOUS_GLOBAL
                ),
                candidate_face_id=normalized_face or None,
            ),
        )


@dataclass(frozen=True)
class _CandidateBatch:
    candidates: tuple[Any, ...]
    highest_score: float
    highest_reid: str
    highest_source: str


@dataclass(frozen=True)
class CandidateState:
    gallery_revision: int
    pending_removal_revision: int


@dataclass(frozen=True)
class _CandidateQuery:
    body_completeness: str
    device_ids: tuple | None
    timestamp: int | None
    excluded_reids: frozenset[str]
    attempt: ReidSearchAttempt

    def kwargs(self) -> dict[str, Any]:
        return {
            "query_body_completeness": self.body_completeness,
            "device_ids": list(self.device_ids) if self.device_ids is not None else None,
            "query_timestamp": self.timestamp,
            "face_id": self.attempt.candidate_face_id,
            "excluded_reids": self.excluded_reids,
        }


@dataclass(frozen=True)
class ReidCandidatePrefetch:
    """请求私有的远程候选；不能在锁外解析或直接提交身份。"""

    state: CandidateState
    query: _CandidateQuery
    embedding: np.ndarray
    milvus: tuple


class PersonReidSearchService:
    def __init__(
        self,
        *,
        person_sbs_embedding_db: Any,
        sbs_buffer: Any,
        resolver: ReidClassResolver | None = None,
        planner: ReidQueryPlanner | None = None,
    ) -> None:
        self.person_sbs_embedding_db = person_sbs_embedding_db
        self.sbs_buffer = sbs_buffer
        self.resolver = resolver or ReidClassResolver()
        self.planner = planner or ReidQueryPlanner()

    def snapshot_candidate_state(self) -> CandidateState | None:
        """仅本进程版本；缺少追踪能力或实际写线程活跃时使用原锁内路径。"""
        gallery_state = getattr(self.person_sbs_embedding_db, "candidate_revision", None)
        pending_state = getattr(self.sbs_buffer, "pending_removal_revision", None)
        if not callable(gallery_state) or not callable(pending_state):
            return None
        revision, active_writers = gallery_state()
        if active_writers:
            return None
        return CandidateState(revision, pending_state())

    def _candidate_query(
        self, query_body_completeness, device_ids, face_id, excluded_reids,
        query_timestamp,
    ) -> _CandidateQuery:
        return _CandidateQuery(
            body_completeness=query_body_completeness,
            device_ids=tuple(device_ids) if device_ids is not None else None,
            timestamp=int(query_timestamp) if query_timestamp is not None else None,
            excluded_reids=frozenset(
                str(reid).strip() for reid in (excluded_reids or ())
                if str(reid).strip()
            ),
            attempt=self.planner.plan(face_id)[0],
        )

    async def prefetch(
        self,
        embedding: np.ndarray,
        *,
        state: CandidateState,
        query_body_completeness: str,
        device_ids: list | None,
        face_id: str | None = None,
        excluded_reids: Collection[str] | None = None,
        query_device_id: str | None = None,
        query_timestamp: int | None = None,
    ) -> ReidCandidatePrefetch:
        query = self._candidate_query(
            query_body_completeness, device_ids, face_id, excluded_reids,
            query_timestamp,
        )
        with runtime_performance.measure("person_reid_milvus_search"):
            milvus = await self.person_sbs_embedding_db.asearch_similar_person_candidates_with_hit_info(
                embedding, **query.kwargs(),
            )
        return ReidCandidatePrefetch(state, query, embedding, milvus)

    async def resolve_prefetched(
        self,
        prefetched: ReidCandidatePrefetch,
        embedding: np.ndarray,
        *,
        query_body_completeness: str,
        device_ids: list | None,
        face_id: str | None = None,
        excluded_reids: Collection[str] | None = None,
        query_device_id: str | None = None,
        query_timestamp: int | None = None,
    ) -> PersonReidSearchResult | None:
        """调用者持身份锁；状态变化返回 None，由调用者完整查询一次。"""
        query = self._candidate_query(
            query_body_completeness, device_ids, face_id, excluded_reids,
            query_timestamp,
        )
        if (prefetched.embedding is not embedding or prefetched.query != query
                or self.snapshot_candidate_state() != prefetched.state):
            return None
        with runtime_performance.measure("person_reid_pending_search"):
            pending = await self.sbs_buffer.asearch_candidates_with_hit_info(
                embedding, **query.kwargs(),
            )
        # pending 查询可能等待其快照锁。必须在读取后再次检查，之后直至
        # resolver/身份登记不得让出事件循环，也不合并旧 pending 快照。
        if self.snapshot_candidate_state() != prefetched.state:
            return None
        batch = self._combine_candidates(prefetched.milvus, pending)
        with runtime_performance.measure("person_reid_resolve"):
            resolution = self._resolve(
                batch, face_id, self._effective_query_device(query_device_id, device_ids),
            )
            return self._result(resolution, batch, query.attempt.mode)

    async def search(
        self,
        embedding: np.ndarray,
        query_body_completeness: str,
        device_ids: list | None,
        face_id: str | None = None,
        excluded_reids: Collection[str] | None = None,
        query_device_id: str | None = None,
        query_timestamp: int | None = None,
    ) -> PersonReidSearchResult:
        excluded = frozenset(
            str(reid).strip()
            for reid in (excluded_reids or ())
            if str(reid).strip()
        )
        effective_device = self._effective_query_device(query_device_id, device_ids)
        attempts = self.planner.plan(face_id)
        attempt = attempts[0]
        batch = await self._collect_candidates(
            embedding,
            query_body_completeness=query_body_completeness,
            device_ids=device_ids,
            query_timestamp=query_timestamp,
            excluded_reids=excluded,
            attempt=attempt,
        )
        with runtime_performance.measure("person_reid_resolve"):
            resolution = self._resolve(batch, face_id, effective_device)
            return self._result(resolution, batch, attempt.mode)

    async def _collect_candidates(
        self,
        embedding: np.ndarray,
        *,
        query_body_completeness: str,
        device_ids: list | None,
        query_timestamp: int | None,
        excluded_reids: frozenset[str],
        attempt: ReidSearchAttempt,
    ) -> _CandidateBatch:
        common = {
            "query_body_completeness": query_body_completeness,
            "device_ids": device_ids,
            "query_timestamp": (
                int(query_timestamp) if query_timestamp is not None else None
            ),
            "face_id": attempt.candidate_face_id,
            "excluded_reids": excluded_reids,
        }
        with runtime_performance.measure("person_reid_milvus_search"):
            milvus = await self.person_sbs_embedding_db.asearch_similar_person_candidates_with_hit_info(
                embedding,
                **common,
            )
        with runtime_performance.measure("person_reid_pending_search"):
            pending = await self.sbs_buffer.asearch_candidates_with_hit_info(
                embedding,
                **common,
            )
        return self._combine_candidates(milvus, pending)

    @staticmethod
    def _combine_candidates(milvus, pending) -> _CandidateBatch:
        with runtime_performance.measure("person_reid_combine_candidates"):
            milvus_candidates, milvus_score, milvus_reid, milvus_source = milvus
            pending_candidates, pending_score, pending_reid, pending_source = pending
            combined = []
            prototype_positions = {}
            for candidate in (*milvus_candidates, *pending_candidates):
                # 已写入但尚未完成 metadata/release 的原型仍在 pending。
                # 两个来源的同一稳定 UUID 只提供一份支持，避免匿名误匹配。
                prototype_id = candidate.prototype_id
                key = ((candidate.reid, str(prototype_id))
                       if prototype_id is not None and str(prototype_id) else None)
                position = prototype_positions.get(key) if key is not None else None
                if position is None:
                    if key is not None:
                        prototype_positions[key] = len(combined)
                    combined.append(candidate)
                elif candidate.score > combined[position].score:
                    combined[position] = candidate
            if pending_score > milvus_score:
                highest = (pending_score, str(pending_reid or ""), pending_source)
            else:
                highest = (milvus_score, str(milvus_reid or ""), milvus_source)
            return _CandidateBatch(tuple(combined), *highest)

    def _resolve(
        self,
        batch: _CandidateBatch,
        face_id: Any,
        query_device_id: str,
    ) -> ReidClassResolution:
        return self.resolver.resolve(
            batch.candidates,
            query_face_id=face_id,
            query_device_id=query_device_id,
        )

    @staticmethod
    def _effective_query_device(
        query_device_id: str | None,
        device_ids: list | None,
    ) -> str:
        effective = str(query_device_id or "").strip()
        search_devices = list(device_ids or ())
        if not effective and len(search_devices) == 1:
            return str(search_devices[0])
        return effective

    @staticmethod
    def _result(
        resolution: ReidClassResolution,
        highest: _CandidateBatch,
        mode: str,
    ) -> PersonReidSearchResult:
        return PersonReidSearchResult(
            reid=resolution.reid,
            score=(
                resolution.score
                if resolution.reid is not None
                else highest.highest_score
            ),
            vector_hit_reid=highest.highest_reid,
            source=(
                resolution.source
                if resolution.reid is not None
                else highest.highest_source
            ),
            state=(
                resolution.state
                if resolution.reid is not None
                else REID_STATE_UNASSIGNED
            ),
            canonical_face_id=resolution.canonical_face_id,
            reason=resolution.reason,
            support_count=resolution.support_count,
            top1_score=resolution.best_score,
            mean_score=resolution.mean_score,
            class_score=resolution.score,
            runner_up_score=resolution.runner_up_score,
            margin=resolution.margin,
            required_margin=resolution.required_margin,
            raw_runner_up_score=resolution.raw_runner_up_score,
            raw_margin=resolution.raw_margin,
            search_mode=mode,
        )
