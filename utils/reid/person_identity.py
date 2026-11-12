"""Atomic person identity assignment and SBS enrollment orchestration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Collection, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable
import uuid

import numpy as np

from core.performance import runtime_performance
from core.queue_metrics import pipeline_metrics

if TYPE_CHECKING:
    from utils.buffers.persistence_ticket import PersistenceTicket
from utils.keyframe.keyframe_flow_helpers import (
    QUALITY_LEVEL_ENROLL,
    UNMATCHED_REID,
    format_reid_source,
)

from .reid_enrollment_state import (
    REID_ENROLLMENT_STATE_NONE,
    normalize_reid_enrollment_state,
)
from .reid_models import (
    REID_SEARCH_MODE_NOT_EXECUTED,
    PersonReidSearchResult,
    normalize_face_id,
)
from .person_search import PersonReidSearchService, ReidCandidatePrefetch
from .reid_state import (
    REID_STATE_ACTIVE,
    REID_STATE_UNASSIGNED,
    normalize_reid_state,
)

REID_DECISION_MATCH_EXISTING = "MATCH_EXISTING"
REID_DECISION_CREATE_NEW = "CREATE_NEW"
REID_DECISION_UNASSIGNED = "UNASSIGNED"
REID_DECISION_SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ReidDiagnostics:
    top1_score: float = 0.0
    mean_score: float = 0.0
    class_score: float = 0.0
    support_count: int = 0
    runner_up_score: float = 0.0
    margin: float = 0.0
    required_margin: float = 0.0
    raw_runner_up_score: float = 0.0
    raw_margin: float = 0.0
    same_frame_excluded_reid_count: int = 0


@dataclass(frozen=True)
class IdentityRequest:
    embedding: np.ndarray | None
    observed_face_id: str
    person_quality_level: int
    person_quality_score: float
    body_completeness: str
    device_id: str
    timestamp: int
    keyframe_id: str
    picture_vector_id: int
    keyframe_url: str
    person_rect: Mapping[str, int]
    ai_id: str
    segment_index: int
    crop_url: str = ""
    excluded_reids: Collection[str] = ()
    sbs_failure: str = ""
    sbs_failure_reason: str = ""
    is_pano_keyframe: bool = False


@dataclass(frozen=True)
class _AssignmentDecision:
    action: str
    reid: str | None
    state: str
    status: int
    source: str
    canonical_face_id: str
    is_new: bool


@dataclass(frozen=True)
class IdentityOutcome:
    reid: str | None = UNMATCHED_REID
    score: float = 0.0
    vector_hit_reid: str = ""
    status: int = 0
    state: str = REID_STATE_UNASSIGNED
    enrollment_state: str = REID_ENROLLMENT_STATE_NONE
    enrollment_reason: str = ""
    info: str = ""
    exclusion_info: str = ""
    decision: str = REID_DECISION_SKIPPED
    decision_reason: str = ""
    search_mode: str = REID_SEARCH_MODE_NOT_EXECUTED
    source: str = ""
    is_new: bool = False
    canonical_face_id: str = "0"
    diagnostics: ReidDiagnostics = ReidDiagnostics()
    new_reid_recognition_release: PersistenceTicket | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class ReidAssignmentPolicy:
    """Pure match/create/unassigned decision after candidate search."""

    def decide(
        self,
        search: PersonReidSearchResult,
        *,
        quality_level: int,
        face_id: str,
    ) -> _AssignmentDecision:
        if search.reid not in (None, "", UNMATCHED_REID):
            return _AssignmentDecision(
                action=REID_DECISION_MATCH_EXISTING,
                reid=search.reid,
                state=REID_STATE_ACTIVE,
                status=2,
                source=search.source,
                canonical_face_id=str(search.canonical_face_id or "0"),
                is_new=False,
            )
        normalized_face_id = normalize_face_id(face_id)
        if quality_level == QUALITY_LEVEL_ENROLL and normalized_face_id:
            return _AssignmentDecision(
                action=REID_DECISION_CREATE_NEW,
                reid=None,
                state=REID_STATE_ACTIVE,
                status=1,
                source="NEW",
                canonical_face_id=normalized_face_id,
                is_new=True,
            )
        return _AssignmentDecision(
            action=REID_DECISION_UNASSIGNED,
            reid=UNMATCHED_REID,
            state=REID_STATE_UNASSIGNED,
            status=0,
            source="",
            canonical_face_id="0",
            is_new=False,
        )


class PersonIdentityService:
    def __init__(
        self,
        *,
        identity_lock: asyncio.Lock,
        search_persons: Callable[..., Awaitable[PersonReidSearchResult]],
        sbs_buffer: Any,
        logger: Any,
        assignment_policy: ReidAssignmentPolicy | None = None,
        id_factory: Callable[[], str] | None = None,
        search_service: PersonReidSearchService | None = None,
    ) -> None:
        self.identity_lock = identity_lock
        self.search_persons = search_persons
        self.search_service = search_service
        self.sbs_buffer = sbs_buffer
        self.logger = logger
        self.assignment_policy = assignment_policy or ReidAssignmentPolicy()
        self.id_factory = id_factory or (lambda: str(uuid.uuid4()))

    async def assign(
        self,
        request: IdentityRequest,
        *,
        register_assignment: Callable[[str], None],
        excluded_reids_factory: Callable[[], Collection[str]] | None = None,
    ) -> IdentityOutcome:
        if request.embedding is None:
            return IdentityOutcome(
                info=(
                    f"质量等级 {request.person_quality_level}，"
                    f"{request.sbs_failure}，未执行 ID 分配"
                ),
                decision_reason=request.sbs_failure_reason,
            )

        runtime_performance.record("person_identity_requests", 0)
        prefetched = None
        if self.search_service is not None:
            # 每个帧 worker 同时最多执行一次 assign，不额外创建预取 task。
            # 帧内提交顺序仍由 pipeline 保证；这里只重叠不同帧的远程查询。
            async with self._locked(snapshot=True):
                prefetch_request = self._with_exclusions(request, excluded_reids_factory)
                state = self.search_service.snapshot_candidate_state()
            if state is not None:
                kwargs, _, _ = self._search_parameters(prefetch_request)
                try:
                    with runtime_performance.measure("person_identity_prefetch"):
                        prefetched = await self.search_service.prefetch(
                            prefetch_request.embedding, state=state, **kwargs,
                        )
                except Exception:
                    runtime_performance.record("person_identity_prefetch_failed", 0)
                    raise
            else:
                runtime_performance.record("person_identity_prefetch_skipped", 0)

        refreshed = False
        while True:
            refresh_state = None
            async with self._locked():
                # 每次重新取得锁都从原请求/factory 固定排除值，不能沿用上次
                # 的 frozenset；其间同帧占用表或调用方集合可能已变化。
                current = self._with_exclusions(request, excluded_reids_factory)
                with runtime_performance.measure("person_identity_search"):
                    search_result = await self._search(
                        current, prefetched=prefetched, allow_fallback=False,
                    )
                    if search_result is not None:
                        if refreshed:
                            runtime_performance.record("person_identity_refresh_used", 0)
                    else:
                        if refreshed and prefetched is not None:
                            runtime_performance.record("person_identity_refresh_invalid", 0)
                        if not refreshed and self.search_service is not None:
                            refresh_state = self.search_service.snapshot_candidate_state()
                        if refresh_state is None:
                            runtime_performance.record(
                                "person_identity_refresh_exhausted" if refreshed else
                                "person_identity_refresh_unavailable", 0,
                            )
                            with runtime_performance.measure("person_identity_locked_search"):
                                search_result = await self._search(current)
                if refresh_state is None:
                    # 从最新 pending 校验到决策/登记不释放锁。业务异常不属于
                    # 候选重读的范围，不能在已经登记后重新分配或重复入队。
                    return await self._assign_locked(
                        current, register_assignment, search_result,
                    )

            # 仅候选读取允许再离锁一次；实际写线程活跃时不等待其空闲，
            # 再次失效或刷新失败时保留原有完整锁内回退，不忙重试。
            refreshed = True
            kwargs, _, _ = self._search_parameters(current)
            try:
                with runtime_performance.measure("person_identity_refresh"):
                    with runtime_performance.measure("person_identity_prefetch"):
                        prefetched = await self.search_service.prefetch(
                            current.embedding, state=refresh_state, **kwargs,
                        )
            except Exception:
                runtime_performance.record("person_identity_prefetch_failed", 0)
                runtime_performance.record("person_identity_refresh_failed", 0)
                prefetched = None

    @asynccontextmanager
    async def _locked(self, *, snapshot: bool = False):
        prefix = "person_identity_snapshot_lock" if snapshot else "person_identity_lock"
        started = time.perf_counter()
        async with pipeline_metrics.slot(
            "person_identity_snapshot" if snapshot else "person_identity",
            self.identity_lock,
        ):
            runtime_performance.record(prefix + "_wait", (time.perf_counter() - started) * 1000)
            with runtime_performance.measure(prefix + "_hold"):
                yield

    @staticmethod
    def _with_exclusions(
        request: IdentityRequest,
        factory: Callable[[], Collection[str]] | None,
    ) -> IdentityRequest:
        # 即使调用者传入可变集合，也在持锁时固定当前查询的排除值。
        return replace(request, excluded_reids=frozenset(
            factory() if factory is not None else request.excluded_reids
        ))

    async def _assign_locked(
        self,
        request: IdentityRequest,
        register_assignment: Callable[[str], None],
        search_result: tuple[PersonReidSearchResult, str, str],
    ) -> IdentityOutcome:
        search, face_info, exclusion_info = search_result
        with runtime_performance.measure("person_identity_decision"):
            decision = self.assignment_policy.decide(
                search,
                quality_level=request.person_quality_level,
                face_id=str(request.observed_face_id or "0"),
            )
            assigned_reid = decision.reid
            if decision.action == REID_DECISION_CREATE_NEW:
                assigned_reid = self.id_factory()

            diagnostics = ReidDiagnostics(
                top1_score=search.top1_score,
                mean_score=search.mean_score,
                class_score=search.class_score,
                support_count=search.support_count,
                runner_up_score=search.runner_up_score,
                margin=search.margin,
                required_margin=search.required_margin,
                raw_runner_up_score=search.raw_runner_up_score,
                raw_margin=search.raw_margin,
                same_frame_excluded_reid_count=len(request.excluded_reids),
            )
            outcome = IdentityOutcome(
                reid=assigned_reid,
                score=float(search.score),
                vector_hit_reid=search.vector_hit_reid,
                status=decision.status,
                state=decision.state,
                info=self._format_assignment_info(
                    request.person_quality_level,
                    search,
                    decision,
                    face_info,
                ),
                exclusion_info=exclusion_info,
                decision=decision.action,
                decision_reason=search.reason,
                search_mode=search.search_mode,
                source=decision.source,
                is_new=decision.is_new,
                canonical_face_id=decision.canonical_face_id,
                diagnostics=diagnostics,
            )
            if search.reid not in (None, "", UNMATCHED_REID):
                outcome = replace(
                    outcome,
                    info=(
                        f"{outcome.info}, ReID match={search.reason}"
                        f", support={search.support_count}"
                        f", rawRunner={search.raw_runner_up_score:.4f}"
                        f", rawMargin={search.raw_margin:.4f}"
                        f", canonicalFaceId={search.canonical_face_id or '0'}"
                    ),
                )
            if assigned_reid not in (None, UNMATCHED_REID):
                register_assignment(str(assigned_reid))
        if (
            request.person_quality_level == QUALITY_LEVEL_ENROLL
            and normalize_face_id(request.observed_face_id)
        ):
            with runtime_performance.measure("person_identity_enroll"):
                outcome = await self._enroll(request, outcome)
        return outcome

    async def _search(
        self,
        request: IdentityRequest,
        *,
        prefetched: ReidCandidatePrefetch | None = None,
        allow_fallback: bool = True,
    ) -> tuple[PersonReidSearchResult, str, str] | None:
        kwargs, face_info, exclusion_info = self._search_parameters(request)
        if request.excluded_reids:
            self.logger.debug(
                "Person same-frame ReID exclusion: "
                f"keyframe_id={request.keyframe_id}, "
                f"segment_index={request.segment_index}, "
                f"excluded_reids={sorted(request.excluded_reids)}, "
                f"sn={request.device_id}"
            )
        result = None
        if prefetched is not None:
            result = await self.search_service.resolve_prefetched(
                prefetched, request.embedding, **kwargs,
            )
            runtime_performance.record(
                "person_identity_prefetch_used" if result is not None
                else "person_identity_prefetch_fallback", 0,
            )
        if result is None:
            if not allow_fallback:
                return None
            # 状态变化时完整重查一次，包含远程过滤与最新 pending；不对旧
            # top-k 本地过滤，不无限重试，也不在锁外选定最终身份。
            result = await self.search_persons(request.embedding, **kwargs)
        if not isinstance(result, PersonReidSearchResult):
            raise TypeError("Person ReID search must return PersonReidSearchResult")
        return result, face_info, exclusion_info

    @staticmethod
    def _search_parameters(request: IdentityRequest):
        face_id = str(request.observed_face_id or "0")
        kwargs: dict[str, Any] = {
            "query_body_completeness": request.body_completeness,
            "device_ids": None,
            "query_device_id": request.device_id,
            "query_timestamp": request.timestamp,
        }
        exclusion_info = ""
        if request.excluded_reids:
            kwargs["excluded_reids"] = request.excluded_reids
            exclusion_info = (
                f"同帧互斥已排除 {len(request.excluded_reids)} 个 reid"
            )

        if face_id != "0":
            kwargs["face_id"] = face_id
            face_info = f"，仅查询 faceId={face_id} 的纯净外观簇"
        else:
            face_info = "，无脸人体采用严格类级检索"
        return kwargs, face_info, exclusion_info

    async def _enroll(
        self,
        request: IdentityRequest,
        outcome: IdentityOutcome,
    ) -> IdentityOutcome:
        enrollment_kwargs = {
            "observed_face_id": request.observed_face_id,
            "canonical_face_id": str(outcome.canonical_face_id or "0"),
            "re_id": outcome.reid,
            "identity_state": outcome.state,
            "body_completeness": request.body_completeness,
            "embedding": request.embedding,
            "device_id": request.device_id,
            "keyframe_id": request.keyframe_id,
            "picture_vector_id": request.picture_vector_id,
            "keyframe_url": request.keyframe_url,
            "crop_url": request.crop_url,
            "person_rect": request.person_rect,
            "timestamp": request.timestamp,
            "ai_id": request.ai_id,
            "quality_score": request.person_quality_score,
            "is_pano_keyframe": request.is_pano_keyframe,
        }
        if outcome.is_new:
            enrollment_kwargs["defer_new_reid_recognition"] = True
        with runtime_performance.measure("person_identity_offer_feature"):
            enrollment = await self.sbs_buffer.offer_feature(**enrollment_kwargs)
        state = outcome.state
        status = outcome.status
        enrollment_state = normalize_reid_enrollment_state(
            enrollment.enrollment_state
        )
        normalized_state = normalize_reid_state(enrollment.reid_state)
        if normalized_state == REID_STATE_ACTIVE:
            state = REID_STATE_ACTIVE
            if not outcome.is_new:
                status = 2
        recognition_release = enrollment.recognition_release
        if outcome.is_new and recognition_release is None:
            raise RuntimeError(
                "New ReID enrollment requires an SBS persistence release"
            )
        return replace(
            outcome,
            state=state,
            status=status,
            enrollment_state=enrollment_state,
            enrollment_reason=str(enrollment.reason or ""),
            info=f"{outcome.info}，SBS入库={enrollment.reason}",
            new_reid_recognition_release=recognition_release,
        )

    @staticmethod
    def _format_assignment_info(
        quality_level: int,
        search: PersonReidSearchResult,
        decision: _AssignmentDecision,
        face_info: str,
    ) -> str:
        formatted_source = format_reid_source(search.source)
        source_info = f"，来源 {formatted_source}" if formatted_source else ""
        highest_info = (
            f"，最高命中来源 {formatted_source}" if formatted_source else ""
        )
        if decision.action == REID_DECISION_MATCH_EXISTING:
            return (
                f"质量等级 {quality_level}，SBS 命中已有 reid，"
                f"最高相似度 {float(search.score):.4f}{source_info}{face_info}"
            )
        if decision.action == REID_DECISION_CREATE_NEW:
            return (
                f"质量等级 {quality_level}，SBS 未命中人型向量，生成新 reid，"
                f"最高相似度 {float(search.score):.4f}{highest_info}{face_info}"
            )
        return (
            f"质量等级 {quality_level}，SBS 未命中人型向量，未生成新 reid，"
            f"最高相似度 {float(search.score):.4f}{highest_info}{face_info}"
        )
