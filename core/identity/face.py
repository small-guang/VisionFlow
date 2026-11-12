"""独立人脸提交与人物内部 Face 解析共用的身份协调服务。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
from typing import Any
import uuid

import numpy as np

from core.algorithm_config import algorithm_config
from core.face_search_pipeline import prefetch_face_search, resolve_face_search
from core.performance import runtime_performance
from core.queue_metrics import pipeline_metrics
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_ENROLL, UNMATCHED_FACE_ID
from utils.reid.face_search import FaceSearchService, face_identity_lock_metrics_scope

from .ports import FaceFeatureSink, FaceGallery, FacePendingView, FaceRecordSink


@dataclass(frozen=True)
class FaceIdentityRequest:
    embedding: np.ndarray
    quality_level: int
    feature_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FaceIdentityOutcome:
    face_id: str = UNMATCHED_FACE_ID
    score: float = 0.0
    vector_hit_face_id: str = ""
    status: int = 0
    info: str = ""
    search_failure_reason: str = ""


@dataclass(frozen=True)
class _FaceSearchResolution:
    """在身份锁内传递查询结果或异常，不把业务提交纳入重试范围。"""

    value: tuple[str | None, float, str] | None = None
    error: Exception | None = None

    def result(self) -> tuple[str | None, float, str]:
        if self.error is not None:
            raise self.error
        if self.value is None:
            raise RuntimeError("Face search produced no terminal resolution")
        return self.value


class FaceIdentityService:
    """保留锁外候选预取；最终复核、决策及 pending 登记在同一锁内。

    records 与 face_features 是输出端口。新生记录交给向量端口等待首次
    持久化；服务不启动 publisher，不操作 writer 队列，也不持有 DB 写锁。
    """

    def __init__(
        self,
        *,
        face_features: FaceFeatureSink,
        records: FaceRecordSink,
        logger: Any,
        face_gallery: FaceGallery | None = None,
        pending_faces: FacePendingView | None = None,
        prefetch_enabled: bool = True,
        identity_lock: Any = None,
        search_service: Any = None,
        search_faces: Callable[..., Awaitable[Any]] | None = None,
        id_factory: Callable[[], str] | None = None,
        metrics: Any = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.identity_lock = identity_lock if identity_lock is not None else asyncio.Lock()
        self.face_features = face_features
        self.records = records
        self.logger = logger
        self.metrics = metrics if metrics is not None else runtime_performance
        self.clock = clock if clock is not None else time.perf_counter
        self.id_factory = id_factory or (lambda: str(uuid.uuid4()))
        if search_service is None and face_gallery is not None:
            search_service = FaceSearchService(
                face_embedding_db=face_gallery, pending_faces=pending_faces,
            )
        self.search_service = search_service if prefetch_enabled else None
        self.search_faces = search_faces or (
            search_service.search if search_service is not None else None
        )
        if self.search_faces is None:
            raise ValueError("Face identity requires a gallery or a search callback")

    @asynccontextmanager
    async def _locked(self):
        started = self.clock()
        async with pipeline_metrics.slot("face_identity", self.identity_lock):
            self.metrics.record("face_identity_lock_wait", (self.clock() - started) * 1000)
            with self.metrics.measure("face_identity_lock_hold"), face_identity_lock_metrics_scope(self.metrics):
                yield

    async def search(
        self, embedding: np.ndarray, threshold: float, device_ids: list | None = None,
    ) -> tuple[str | None, float, str]:
        """人物内部的只读 Face 解析，与独立人脸提交共享同一协调锁。"""
        async with self._resolution_scope(embedding, threshold, device_ids) as resolution:
            return resolution.result()

    async def _resolve(self, prefetched, embedding, threshold, device_ids):
        return await resolve_face_search(
            self.search_service, prefetched, embedding,
            threshold=threshold, device_ids=device_ids, fallback=self.search_faces,
        )

    @asynccontextmanager
    async def _resolution_scope(self, embedding, threshold, device_ids):
        """最多锁外刷新一次；终态交给调用方时仍持锁直到 pending 登记。

        刷新只重做候选读取，不能跨越身份决策。active writer 或缺少版本
        接口时立即沿原锁内完整查询；不等待 writer 空闲，也不忙重试。
        """
        self.metrics.record("face_identity_requests", 0)
        prefetched = await prefetch_face_search(
            self.search_service, self.identity_lock, embedding, device_ids=device_ids,
        )
        refreshed = False
        while True:
            refresh_state = None
            async with self._locked():
                try:
                    result = None
                    if prefetched is not None:
                        result = await self.search_service.resolve_prefetched(
                            prefetched, embedding, threshold=threshold,
                            device_ids=device_ids,
                        )
                        if result is not None:
                            self.metrics.record("face_identity_prefetch_used", 0)
                            if refreshed:
                                self.metrics.record("face_identity_refresh_used", 0)
                        else:
                            self.metrics.record("face_identity_prefetch_fallback", 0)
                            if refreshed:
                                self.metrics.record("face_identity_refresh_invalid", 0)
                    if result is None:
                        if not refreshed and self.search_service is not None:
                            refresh_state = self.search_service.snapshot_candidate_state()
                        if refresh_state is None:
                            self.metrics.record(
                                "face_identity_refresh_exhausted" if refreshed else
                                "face_identity_refresh_unavailable", 0,
                            )
                            with self.metrics.measure("face_identity_locked_search"):
                                result = await self._resolve(
                                    None, embedding, threshold, device_ids,
                                )
                    resolution = _FaceSearchResolution(value=result)
                except Exception as error:
                    # 与原路径相同：commit 将查询失败转换成阻止新生身份的
                    # 结果，search 向上传播；取消不捕获。业务代码尚未执行。
                    resolution = _FaceSearchResolution(error=error)
                    refresh_state = None
                if refresh_state is None:
                    # yield 必须在查询 try/except 外：业务异常不能触发重查，
                    # 也不能吞掉取消；调用方登记 pending 前一直持有此锁。
                    yield resolution
                    return
            refreshed = True
            try:
                with pipeline_metrics.activity("face_search_prefetch"):
                    with self.metrics.measure("face_identity_refresh"):
                        with self.metrics.measure("face_identity_prefetch"):
                            prefetched = await self.search_service.prefetch(
                                embedding, state=refresh_state, device_ids=device_ids,
                            )
            except Exception:
                self.metrics.record("face_identity_prefetch_failed", 0)
                self.metrics.record("face_identity_refresh_failed", 0)
                prefetched = None

    async def commit(
        self,
        request: FaceIdentityRequest,
        build_record: Callable[[FaceIdentityOutcome], dict[str, Any]],
    ) -> FaceIdentityOutcome:
        """按调用方的检测顺序提交；原子登记后才释放身份协调锁。"""
        async with self._resolution_scope(
            request.embedding,
            algorithm_config.face.matching.general_similarity_threshold, None,
        ) as resolution:
            try:
                matched, score, hit = resolution.result()
            except Exception as error:
                reason = f"人脸向量库检索失败，未生成新 face_id：{error}"
                outcome = FaceIdentityOutcome(info=reason, search_failure_reason=reason)
                self.logger.error(
                    "Face gallery search failed; new face creation was blocked: "
                    f"sn={request.feature_metadata.get('device_id', '')}, "
                    f"keyframe_id={request.feature_metadata.get('keyframe_id', '')}, error={error}"
                )
            else:
                outcome = self._decide(request.quality_level, matched, float(score), hit)
            record = build_record(outcome)
            if self._should_enroll(request.quality_level, outcome):
                feature = dict(request.feature_metadata)
                feature.update(face_id=outcome.face_id, embedding=request.embedding)
                if outcome.status == 1:
                    feature["new_face_recognition_record"] = record
                await self.face_features.add_face_feature(**feature)
        if outcome.status != 1:
            await self.records.add_face_recognition_record(record)
        return outcome

    def _decide(self, quality_level, matched, score, hit) -> FaceIdentityOutcome:
        if matched is not None:
            info = f"质量等级 {quality_level}，命中已有 face_id，最高相似度 {score:.4f}"
            if score < algorithm_config.face.matching.single_sample_similarity_threshold:
                info += "，灰区双样本确认，本次特征不入库"
            return FaceIdentityOutcome(matched, score, hit, 2, info)
        if score >= algorithm_config.face.matching.general_similarity_threshold:
            info = (
                "灰区匹配缺少同一 ID 的两个独立历史样本，身份待确认，"
                f"未生成新 face_id，最高相似度 {score:.4f}"
            )
            return FaceIdentityOutcome(score=score, vector_hit_face_id=hit, info=info)
        if quality_level == QUALITY_LEVEL_ENROLL:
            return FaceIdentityOutcome(
                self.id_factory(), score, hit, 1,
                f"质量等级 {quality_level}，未命中人脸向量，生成新 face_id，最高相似度 {score:.4f}",
            )
        return FaceIdentityOutcome(
            score=score, vector_hit_face_id=hit,
            info=f"质量等级 {quality_level}，未命中人脸向量，未生成新 face_id，最高相似度 {score:.4f}",
        )

    @staticmethod
    def _should_enroll(quality_level: int, outcome: FaceIdentityOutcome) -> bool:
        return (
            quality_level == QUALITY_LEVEL_ENROLL
            and not outcome.search_failure_reason
            and (outcome.status == 1 or (
                outcome.status == 2
                and outcome.score >= algorithm_config.face.matching.single_sample_similarity_threshold
            ))
        )
