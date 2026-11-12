"""聚合 Worker 耗时，避免逐图、逐推理请求输出日志。

记录与快照都运行在 Worker 的事件循环线程。解码在线程中计时，完成后回到
事件循环更新聚合计数；人物裁图准备也在线程计时，mask 重建仍同步执行。
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
import re
import time
from collections import OrderedDict
from collections.abc import Iterator
from core.queue_metrics import pipeline_metrics

SUMMARY_PREFIX = "ALog performance summary: "
REPLAY_UID_PREFIX = "alogtest-v1:"
REPLAY_MAX_SEQUENCE = 65535
REPLAY_MAX_RUNS = 16
REPLAY_RUN_TTL_SECONDS = 3600
_REPLAY_UID_PATTERN = re.compile(
    r"alogtest-v1:([0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}):(0|[1-9][0-9]{0,4})\Z"
)
_OPERATION_NAMES = (
    "image_cpu_run", "image_cpu_queue_wait",
    "storage_io_run", "storage_io_queue_wait",
    "storage_download_run", "storage_download_queue_wait",
    "storage_upload_run", "storage_upload_queue_wait",
    "vector_search_run", "vector_search_queue_wait",
    "vector_query_admission_wait", "vector_query_batch_wait", "vector_query_rpc_run",
    "vector_write_run", "vector_write_queue_wait",
    "image_decode", "image_decode_queue_wait", "person_seg_unpack", "event_loop_lag",
    "person_crop_prepare", "person_crop_prepare_queue_wait",
    "person_observation_build", "person_observation_build_queue_wait",
    "face_identity_lock_wait", "face_identity_lock_hold",
    "face_identity_snapshot_lock_wait", "face_identity_snapshot_lock_hold",
    "face_identity_prefetch", "face_identity_prefetch_used",
    "face_identity_prefetch_fallback", "face_identity_prefetch_skipped",
    "face_identity_prefetch_failed", "face_milvus_search",
    "face_identity_requests", "face_identity_refresh",
    "face_identity_refresh_used", "face_identity_refresh_invalid",
    "face_identity_refresh_failed", "face_identity_refresh_unavailable",
    "face_identity_locked_search", "face_identity_refresh_exhausted",
    "face_identity_locked_full_pending", "face_identity_locked_full_milvus",
    "face_identity_locked_full_resolve", "face_identity_locked_prefetch_pending",
    "face_identity_locked_prefetch_resolve",
    "face_identity_locked_query_admission", "face_identity_locked_query_batch",
    "face_identity_locked_query_executor", "face_identity_locked_query_sdk",
    "face_identity_locked_query_resume",
    "face_pending_search", "face_match_resolve",
    "person_identity_lock_wait", "person_identity_lock_hold",
    "person_identity_snapshot_lock_wait", "person_identity_snapshot_lock_hold",
    "person_identity_prefetch", "person_identity_prefetch_used",
    "person_identity_prefetch_fallback", "person_identity_prefetch_skipped",
    "person_identity_requests", "person_identity_prefetch_failed",
    "person_identity_refresh", "person_identity_refresh_used",
    "person_identity_refresh_invalid", "person_identity_refresh_failed",
    "person_identity_refresh_unavailable", "person_identity_refresh_exhausted",
    "person_identity_locked_search",
    "sbs_search_thread_queue", "sbs_search_thread_run", "sbs_search_loop_resume",
    "sbs_search_group_wait",
    "sbs_search_future_callback", "sbs_search_future_group_submit_span",
    "sbs_search_future_critical_dispatch", "sbs_search_future_critical_thread_queue",
    "sbs_search_future_critical_thread_run", "sbs_search_future_critical_callback",
    "sbs_search_future_group_join",
    "person_identity_search", "person_identity_decision", "person_identity_enroll",
    "person_identity_offer_feature", "person_reid_milvus_search",
    "person_reid_pending_search", "person_reid_resolve", "person_reid_combine_candidates",
    "triton_shm_copy_queue", "triton_shm_copy_run", "triton_shm_used",
    "triton_shm_fallback", "triton_shm_poisoned",
    "triton_shm_reregister_attempt", "triton_shm_reregistered", "triton_shm_reregister_failed",
)
_INFERENCE_TIMING_NAMES = ("batch_wait_ms", "executor_queue_wait_ms", "rpc_ms")


def parse_replay_uid(uid: object) -> tuple[str, int] | None:
    """只接受显式测试标记，生产 UUID 不创建任何按 run 聚合状态。"""
    if not isinstance(uid, str) or not uid.startswith(REPLAY_UID_PREFIX):
        return None
    match = _REPLAY_UID_PATTERN.fullmatch(uid)
    if match is None:
        return None
    sequence = int(match[2])
    return (match[1], sequence) if sequence <= REPLAY_MAX_SEQUENCE else None


class _ReplayFrames:
    def __init__(self):
        self._seen = bytearray((REPLAY_MAX_SEQUENCE + 1) // 8)
        self.count = 0
        self.duplicates = 0
        self.first_ms: float | None = None
        self.last_ms: float | None = None
        self.min_sequence: int | None = None
        self.max_sequence: int | None = None
        self.clock_went_backwards = False

    def add(self, sequence: int, completed_ms: float) -> None:
        byte, bit = divmod(sequence, 8)
        mask = 1 << bit
        if self._seen[byte] & mask:
            self.duplicates += 1
            return
        self._seen[byte] |= mask
        if self.last_ms is not None and completed_ms < self.last_ms:
            self.clock_went_backwards = True
        self.count += 1
        self.first_ms = completed_ms if self.first_ms is None else self.first_ms
        self.last_ms = completed_ms
        self.min_sequence = sequence if self.min_sequence is None else min(self.min_sequence, sequence)
        self.max_sequence = sequence if self.max_sequence is None else max(self.max_sequence, sequence)

    def fields(self) -> dict:
        return {
            "completed_count": self.count,
            "duplicate_completions": self.duplicates,
            "first_completed_unix_ms": self.first_ms,
            "last_completed_unix_ms": self.last_ms,
            "min_sequence": self.min_sequence,
            "max_sequence": self.max_sequence,
            "clock_went_backwards": self.clock_went_backwards,
        }


@dataclass
class _ReplayRun:
    topic: str
    generation: int
    touched_at: float
    main: _ReplayFrames
    pano: _ReplayFrames
    conflicting_topic: bool = False

    def fields(self) -> dict:
        return {
            "topic": self.topic,
            "generation": self.generation,
            "conflicting_topic": self.conflicting_topic,
            "main": self.main.fields(),
            "pano": self.pano.fields(),
        }


@dataclass
class _Timing:
    count: int = 0
    errors: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def add(self, duration_ms: float, *, failed: bool = False) -> None:
        self.count += 1
        self.errors += int(failed)
        self.total_ms += duration_ms
        self.max_ms = max(self.max_ms, duration_ms)

    def fields(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "errors": self.errors,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.total_ms / self.count, 3) if self.count else 0.0,
            "max_ms": round(self.max_ms, 3),
        }


class RuntimePerformance:
    """仅在周期汇总时输出日志，按操作和模型保存内存计数。"""

    def __init__(self, *, clock=time.perf_counter, wall_clock=time.time, log=None):
        self._clock = clock
        self._wall_clock = wall_clock
        self._logger = log
        self._operations = {name: _Timing() for name in _OPERATION_NAMES}
        self._messages: dict[str, int] = {}
        self._images: dict[str, int] = {}
        self._inference: dict[str, dict] = {}
        self._started_at = self._clock()
        self._started_at_unix_ms = int(self._wall_clock() * 1000)
        self._tasks: list[asyncio.Task] = []
        self._replay_runs: OrderedDict[str, _ReplayRun] = OrderedDict()
        self._replay_generation = 0
        self._replay_evictions = 0

    def record(self, operation: str, duration_ms: float, *, failed: bool = False) -> None:
        if not math.isfinite(duration_ms) or duration_ms < 0:
            return
        self._operations[operation].add(duration_ms, failed=failed)

    def inference_completed(
        self,
        model: str,
        *,
        batch_size: int,
        batch_wait_ms: float,
        executor_queue_wait_ms: float | None,
        rpc_ms: float | None,
        failed: bool,
    ) -> None:
        # 模型名称来自 adapter 契约，不使用图片元数据作为聚合键。
        if model not in self._inference:
            self._inference[model] = {
                "batches": 0,
                "samples": 0,
                "errors": 0,
                **{name: _Timing() for name in _INFERENCE_TIMING_NAMES},
            }
        values = self._inference[model]
        values["batches"] += 1
        values["samples"] += batch_size
        values["errors"] += int(failed)
        for name, duration_ms in (
            ("batch_wait_ms", batch_wait_ms),
            ("executor_queue_wait_ms", executor_queue_wait_ms),
            ("rpc_ms", rpc_ms),
        ):
            if duration_ms is not None and math.isfinite(duration_ms) and duration_ms >= 0:
                values[name].add(duration_ms, failed=failed)

    @contextmanager
    def measure(self, operation: str) -> Iterator[None]:
        started = self._clock()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.record(operation, (self._clock() - started) * 1000, failed=failed)

    def message_completed(self, topic: str) -> None:
        self._messages[topic] = self._messages.get(topic, 0) + 1

    def image_completed(self, topic: str, *, uid: str = "", is_pano: bool = False) -> None:
        pipeline_metrics.increment("image_pipeline_completed")
        self._images[topic] = self._images.get(topic, 0) + 1
        traced = parse_replay_uid(uid)
        if traced is None:
            return
        now = self._clock()
        self._expire_replay_runs(now)
        run_id, sequence = traced
        run = self._replay_runs.get(run_id)
        if run is None:
            if len(self._replay_runs) >= REPLAY_MAX_RUNS:
                self._replay_runs.popitem(last=False)
                self._replay_evictions += 1
            self._replay_generation += 1
            run = _ReplayRun(topic, self._replay_generation, now, _ReplayFrames(), _ReplayFrames())
            self._replay_runs[run_id] = run
        elif run.topic != topic:
            run.conflicting_topic = True
            return
        run.touched_at = now
        self._replay_runs.move_to_end(run_id)
        frames = run.pano if is_pano else run.main
        frames.add(sequence, round(self._wall_clock() * 1000, 3))

    def _expire_replay_runs(self, now: float) -> None:
        while self._replay_runs:
            oldest = next(iter(self._replay_runs.values()))
            if now - oldest.touched_at < REPLAY_RUN_TTL_SECONDS:
                break
            self._replay_runs.popitem(last=False)
            self._replay_evictions += 1

    def snapshot(self, *, final: bool = False) -> dict:
        """重置周期计数；显式测试 run 累计值保留，不输出图片标识。"""
        now = self._clock()
        self._expire_replay_runs(now)
        now_unix_ms = int(self._wall_clock() * 1000)
        result = {
            "schema_version": 2,
            "pid": os.getpid(),
            "interval_seconds": round(max(0.0, now - self._started_at), 6),
            "interval_start_unix_ms": self._started_at_unix_ms,
            "interval_end_unix_ms": now_unix_ms,
            "final": final,
            "operations": {
                name: timing.fields() for name, timing in self._operations.items()
            },
            "messages_completed_by_topic": self._messages,
            "images_completed_by_topic": self._images,
            "inference_by_model": {
                model: {
                    name: value.fields() if isinstance(value, _Timing) else value
                    for name, value in values.items()
                }
                for model, values in self._inference.items()
            },
            "replay_runs": {run_id: values.fields() for run_id, values in self._replay_runs.items()},
            "replay_run_evictions": self._replay_evictions,
        }
        self._operations = {name: _Timing() for name in _OPERATION_NAMES}
        self._messages = {}
        self._images = {}
        self._inference = {}
        self._started_at = now
        self._started_at_unix_ms = now_unix_ms
        return result

    def _emit(self, *, final: bool = False) -> None:
        active_logger = self._logger
        if active_logger is None:
            from loguru import logger as active_logger
        active_logger.info(
            SUMMARY_PREFIX + json.dumps(self.snapshot(final=final), separators=(",", ":"))
        )

    async def _report(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            self._emit()

    async def _probe_event_loop(self, interval_seconds: float) -> None:
        while True:
            expected_at = self._clock() + interval_seconds
            await asyncio.sleep(interval_seconds)
            self.record("event_loop_lag", max(0.0, self._clock() - expected_at) * 1000)

    def start(self, *, interval_seconds: float = 5.0, probe_seconds: float = 0.1) -> None:
        if self._tasks:
            return
        for name, value in (("interval_seconds", interval_seconds), ("probe_seconds", probe_seconds)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        # 启动时丢弃导入阶段或上一次服务运行留下的计数。
        self._replay_runs.clear()
        self.snapshot()
        self._tasks = [
            asyncio.create_task(self._report(interval_seconds), name="alog-performance-report"),
            asyncio.create_task(self._probe_event_loop(probe_seconds), name="alog-event-loop-probe"),
        ]

    async def stop(self) -> None:
        if not self._tasks:
            return
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._emit(final=True)


runtime_performance = RuntimePerformance()
