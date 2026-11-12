"""Face 查询阶段：锁外 I/O、锁内版本复核；不负责身份分配。"""
from __future__ import annotations

import time

from core.performance import runtime_performance
from core.queue_metrics import pipeline_metrics


async def prefetch_face_search(service, identity_lock, embedding, *, device_ids=None):
    if service is None:
        return None
    started = time.perf_counter()
    async with pipeline_metrics.slot("face_identity_snapshot", identity_lock):
        runtime_performance.record(
            "face_identity_snapshot_lock_wait", (time.perf_counter() - started) * 1000,
        )
        with runtime_performance.measure("face_identity_snapshot_lock_hold"):
            state = service.snapshot_candidate_state()
    if state is None:
        runtime_performance.record("face_identity_prefetch_skipped", 0)
        return None
    try:
        with pipeline_metrics.activity("face_search_prefetch"):
            with runtime_performance.measure("face_identity_prefetch"):
                return await service.prefetch(embedding, state=state, device_ids=device_ids)
    except Exception:
        # Preserve the original locked search/error handling before any identity
        # decision. Cancellation is deliberately not caught.
        runtime_performance.record("face_identity_prefetch_failed", 0)
        return None


async def resolve_face_search(
    service, prefetched, embedding, *, threshold, device_ids, fallback,
):
    """调用方持有共享 Face 身份锁；直到登记 pending 前不得释放。"""
    if prefetched is not None:
        result = await service.resolve_prefetched(
            prefetched, embedding, threshold=threshold, device_ids=device_ids,
        )
        if result is not None:
            runtime_performance.record("face_identity_prefetch_used", 0)
            return result
        runtime_performance.record("face_identity_prefetch_fallback", 0)
    return await fallback(embedding, threshold=threshold, device_ids=device_ids)
