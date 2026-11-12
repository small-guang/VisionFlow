"""事件循环内的阶段计数与独立 JSONL 日志；不读取输入内容，不查询外部服务。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import uuid


@dataclass
class _Stage:
    waiting: dict[int, float] = field(default_factory=dict)
    active: dict[int, float] = field(default_factory=dict)
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    waits: int = 0
    wait_ms: float = 0.0
    run_ms: float = 0.0


class WaitTicket:
    """即使取消或重复清理，也只扣减一次等待数。"""

    def __init__(self, stage: _Stage, key: int, clock: Callable[[], float]):
        self._stage = stage
        self._key = key
        self._clock = clock

    def close(self) -> None:
        started = self._stage.waiting.pop(self._key, None)
        if started is not None:
            self._stage.waits += 1
            self._stage.wait_ms += max(0.0, self._clock() - started) * 1000


class PipelineMetrics:
    """只在所属事件循环调用。阶段名与 provider 名必须来自代码，不使用业务 ID。"""

    def __init__(self, *, clock=time.perf_counter, wall_clock=time.time):
        self._clock = clock
        self._wall_clock = wall_clock
        self._stages: dict[str, _Stage] = {}
        self._counters: dict[str, int] = {}
        self._providers: dict[str, Callable[[], dict]] = {}
        self._previous_stages: dict[str, tuple] = {}
        self._previous_counters: dict[str, int] = {}
        self._sequence = 0
        self._sampled_at = clock()
        self._session = uuid.uuid4().hex
        self._task: asyncio.Task | None = None
        self._sink_id: int | None = None
        self._logger = None

    def begin_wait(self, name: str) -> WaitTicket:
        stage = self._stages.setdefault(name, _Stage())
        self._sequence += 1
        stage.waiting[self._sequence] = self._clock()
        return WaitTicket(stage, self._sequence, self._clock)

    @contextmanager
    def waiting(self, name: str):
        ticket = self.begin_wait(name)
        try:
            yield
        finally:
            ticket.close()

    @contextmanager
    def activity(self, name: str):
        stage = self._stages.setdefault(name, _Stage())
        self._sequence += 1
        key = self._sequence
        started = self._clock()
        stage.active[key] = started
        try:
            yield
        except asyncio.CancelledError:
            stage.cancelled += 1
            raise
        except BaseException:
            stage.failed += 1
            raise
        else:
            stage.completed += 1
        finally:
            stage.active.pop(key)
            stage.run_ms += max(0.0, self._clock() - started) * 1000

    @asynccontextmanager
    async def slot(self, name: str, limiter):
        # 等待取消时不释放从未取得的许可；持有期间保留原 async context 语义。
        ticket = self.begin_wait(name)
        try:
            async with limiter:
                ticket.close()
                with self.activity(name):
                    yield
        finally:
            ticket.close()

    def increment(self, name: str, amount: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + amount

    def register(self, name: str, provider: Callable[[], dict]) -> None:
        self._providers[name] = provider

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)

    def snapshot(self, *, final: bool = False) -> dict:
        now = self._clock()
        elapsed = max(0.0, now - self._sampled_at)
        stages = {}
        for name, stage in self._stages.items():
            totals = (stage.completed, stage.failed, stage.cancelled,
                      stage.waits, stage.wait_ms, stage.run_ms)
            previous = self._previous_stages.get(name, (0, 0, 0, 0, 0.0, 0.0))
            completed, failed, cancelled, waits, wait_ms, run_ms = (
                value - before for value, before in zip(totals, previous)
            )
            ended = completed + failed + cancelled
            stages[name] = {
                "waiting": len(stage.waiting), "active": len(stage.active),
                "oldest_wait_ms": round(max(0.0, now - min(stage.waiting.values())) * 1000, 3) if stage.waiting else 0.0,
                "oldest_active_ms": round(max(0.0, now - min(stage.active.values())) * 1000, 3) if stage.active else 0.0,
                "completed": completed, "failed": failed, "cancelled": cancelled,
                "completed_total": stage.completed,
                "completed_per_second": round(completed / elapsed, 3) if elapsed else 0.0,
                "waits_finished": waits,
                "avg_wait_ms": round(wait_ms / waits, 3) if waits else 0.0,
                "avg_active_ms": round(run_ms / ended, 3) if ended else 0.0,
            }
            self._previous_stages[name] = totals
        counters = {}
        for name, total in self._counters.items():
            delta = total - self._previous_counters.get(name, 0)
            counters[name] = {
                "count": delta, "total": total,
                "per_second": round(delta / elapsed, 3) if elapsed else 0.0,
            }
        self._previous_counters = self._counters.copy()
        queues = {}
        for name, provider in self._providers.items():
            try:
                values = provider()
                # 提前隔离不可序列化的 provider，不让坏快照影响其他端的记录。
                json.dumps(values, allow_nan=False)
                queues[name] = values
            except Exception as error:
                queues[name] = {"available": False, "error_type": type(error).__name__}
        self._sampled_at = now
        return {
            "schema_version": 1,
            "timestamp": datetime.fromtimestamp(self._wall_clock(), timezone.utc).isoformat(timespec="milliseconds"),
            "pid": os.getpid(), "session_id": self._session,
            "interval_seconds": round(elapsed, 6), "final": final,
            "stages": stages, "counters": counters, "queues": queues,
        }

    def start(self, *, path: str, interval_seconds: float = 5.0) -> None:
        if self._task is not None:
            return
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("pipeline metrics interval must be finite and positive")
        loop = asyncio.get_running_loop()
        from loguru import logger

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._sink_id = logger.add(
            str(target), format="{message}", level="INFO", enqueue=True,
            rotation="100 MB", retention="7 days", encoding="utf-8",
            filter=lambda record: record["extra"].get("pipeline_stats_session") == self._session,
        )
        try:
            self._logger = logger.bind(pipeline_stats_session=self._session)
            self._sampled_at = self._clock()
            self._emit()
            self._task = loop.create_task(self._report(interval_seconds), name="alog-pipeline-stats")
        except BaseException:
            self._close_sink()
            raise

    def _emit(self, *, final: bool = False) -> None:
        self._logger.info(json.dumps(self.snapshot(final=final), ensure_ascii=False, allow_nan=False))

    @staticmethod
    def _report_emit_failure(error: Exception) -> None:
        # 普通业务 logger 不绑定统计标签，避免告警混入 JSONL；不输出异常正文。
        from loguru import logger

        try:
            logger.warning("Pipeline statistics emission failed: {}", type(error).__name__)
        except Exception:
            # 业务日志也可能不可用，仍保留下一周期恢复统计的机会。
            pass

    def _close_sink(self) -> None:
        from loguru import logger

        sink_id, self._sink_id = self._sink_id, None
        self._logger = None
        if sink_id is not None:
            logger.remove(sink_id)

    async def _report(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                self._emit()
            except Exception as error:
                self._report_emit_failure(error)

    async def stop(self) -> None:
        if self._task is None:
            return
        task, self._task = self._task, None
        try:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            try:
                self._emit(final=True)
            except Exception as error:
                self._report_emit_failure(error)
            await self._logger.complete()
        finally:
            self._close_sink()


pipeline_metrics = PipelineMetrics()
