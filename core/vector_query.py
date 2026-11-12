"""向量查询微批调度；只复用既有搜索执行器，不持有身份或图库状态。"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, Counter
from copy import deepcopy
from dataclasses import dataclass
import math
import time
from typing import Any

from core.face_query_observability import current_locked_face_query_metrics


def validate_query_limits(batch_size: int, wait_ms: float, capacity: int) -> None:
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("query batch size must be an integer in [1, 32]")
    if type(capacity) is not int or capacity < batch_size:
        raise ValueError("query capacity must be an integer >= batch size")
    if isinstance(wait_ms, bool) or not isinstance(wait_ms, (int, float)) or not math.isfinite(wait_ms) or not 0 <= wait_ms <= 10:
        raise ValueError("query wait must be finite and in [0, 10] ms")


def _freeze(value: Any) -> tuple:
    """严格按类型、内容分组；未知 SDK 参数类型拒绝合批，不能忽略字段。"""
    if value is None or type(value) in (str, bool, int):
        return type(value).__name__, value
    if type(value) is float and math.isfinite(value):
        return "float", value
    if type(value) in (list, tuple):
        return type(value).__name__, tuple(_freeze(item) for item in value)
    if type(value) is dict and all(type(key) is str for key in value):
        return "dict", tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    raise ValueError(f"unsupported query parameter type: {type(value).__name__}")


@dataclass(eq=False)
class _Query:
    client: Any
    vector: list
    parameters: dict
    key: tuple
    result: asyncio.Future
    queued_at: float
    submitted: bool = False
    locked_metrics: Any = None
    sdk_finished_at: float | None = None


def _search(client, data, parameters):
    started = time.perf_counter()
    try:
        result = client.search(data=data, **parameters)
    except BaseException as error:
        return None, error, started, time.perf_counter()
    return result, None, started, time.perf_counter()


class VectorQueryCoordinator:
    """有界接纳兼容单向量查询，按原索引分发 Milvus 多 query 响应。

    capacity 同时包含已接纳排队和真实在途向量。取消排队查询会撤销它；
    取消已提交查询仅停止该调用者等待，SDK 退出前仍占容量。close 排空已
    接纳查询后返回，调用者应随后关闭执行器，最后关闭 MilvusClient。
    SDK timeout 仍只约束远程 RPC，不包含调度等待；端到端超时由调用方控制。
    """

    def __init__(self, executor, *, batch_size: int = 4, wait_ms: float = 1.0,
                 capacity: int = 128, metrics=None) -> None:
        validate_query_limits(batch_size, wait_ms, capacity)
        self._executor = executor
        self._batch_size = batch_size
        self._wait_seconds = wait_ms / 1000
        self._capacity = capacity
        self._metrics = metrics
        self._loop = asyncio.get_running_loop()
        self._groups: OrderedDict[tuple, list[_Query]] = OrderedDict()
        self._active: set[asyncio.Task] = set()
        self._changed = asyncio.Event()
        self._space = asyncio.Event()
        self._space.set()
        self._admitted = 0
        self._waiting = 0
        self._closed = False
        self._pump_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._counts: Counter = Counter({name: 0 for name in (
            "accepted_queries", "cancelled_queries", "completed_queries", "failed_queries",
            "finished_rpcs", "finished_rpc_queries",
        )})
        self._batch_histogram: Counter = Counter()
        self._admission_ms = 0.0
        self._batch_wait_ms = 0.0
        self._rpc_run_ms = 0.0

    def _record(self, name: str, duration_ms: float) -> None:
        if self._metrics is not None:
            try:
                self._metrics.record(name, duration_ms)
            except Exception as error:
                self._loop.call_exception_handler({"message": "vector query metric failed", "exception": error})

    def _record_locked(self, metrics, name: str, duration_ms: float, *, failed=False) -> None:
        if metrics is not None:
            try:
                metrics.record(name, duration_ms, failed=failed)
            except Exception as error:
                self._loop.call_exception_handler({"message": "locked face query metric failed", "exception": error})

    async def search(self, client, vector: list, /, **parameters) -> list[dict]:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("vector query coordinator belongs to a different event loop")
        if "data" in parameters:
            raise ValueError("query data must be supplied as a single vector")
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("query vector must be a nonempty list or tuple")
        # 调用方与泵任务不同，必须在接纳前捕获当前查询的身份锁归属。
        locked_metrics = current_locked_face_query_metrics()
        # 每个 SDK 参数均纳入 key；不同 client、维度、过滤和一致性不会合并。
        parameters = deepcopy(parameters)
        parameter_key = _freeze(parameters)
        started = time.perf_counter()
        self._waiting += 1
        try:
            while self._admitted >= self._capacity and not self._closed:
                self._space.clear()
                await self._space.wait()
            if self._closed:
                raise RuntimeError("vector query coordinator is closed")
        finally:
            self._waiting -= 1
        queued_at = time.perf_counter()
        admission_ms = (queued_at - started) * 1000
        self._admission_ms += admission_ms
        self._record("vector_query_admission_wait", admission_ms)
        self._record_locked(locked_metrics, "face_identity_locked_query_admission", admission_ms)
        # 接纳后才复制向量；参数在等待前冻结，避免 caller 修改后 key 失真。
        vector = list(vector)
        key = (id(client), len(vector), parameter_key)
        query = _Query(client, vector, parameters, key,
                       self._loop.create_future(), queued_at, locked_metrics=locked_metrics)
        self._admitted += 1
        self._counts["accepted_queries"] += 1
        self._groups.setdefault(key, []).append(query)
        self._changed.set()
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump(), name="alog-vector-query-pump")
        caller_cancelled = False
        try:
            return await asyncio.shield(query.result)
        except asyncio.CancelledError:
            caller_cancelled = True
            self._counts["cancelled_queries"] += 1
            if query.result.done() and not query.result.cancelled():
                query.result.exception()  # 消费取消与失败响应同时发生的异常。
            else:
                query.result.cancel()
            if not query.submitted:
                group = self._groups.get(key)
                if group is not None and query in group:
                    group.remove(query)
                    if not group:
                        del self._groups[key]
                    self._release(1)
                    self._changed.set()
            raise
        finally:
            if (not caller_cancelled and query.sdk_finished_at is not None
                    and not query.result.cancelled()):
                failed = query.result.done() and query.result.exception() is not None
                self._record_locked(
                    query.locked_metrics, "face_identity_locked_query_resume",
                    (time.perf_counter() - query.sdk_finished_at) * 1000,
                    failed=failed,
                )

    def _release(self, count: int) -> None:
        self._admitted -= count
        self._space.set()

    async def _pump(self) -> None:
        while True:
            self._changed.clear()
            if self._closed and not self._groups and not self._active:
                return
            delay = None
            # 不向原搜索池堆积额外无界任务；SBS 与本模块仍共享真实线程许可。
            while self._groups and len(self._active) < self._executor.workers:
                now = time.perf_counter()
                chosen = None
                for key, group in self._groups.items():
                    remaining = group[0].queued_at + self._wait_seconds - now
                    if len(group) >= self._batch_size or remaining <= 0 or self._closed:
                        chosen = key
                        break
                    delay = remaining if delay is None else min(delay, remaining)
                if chosen is None:
                    break
                group = self._groups[chosen]
                batch = group[:self._batch_size]
                del group[:len(batch)]
                if not group:
                    del self._groups[chosen]
                else:
                    self._groups.move_to_end(chosen)
                for query in batch:
                    query.submitted = True
                    wait_ms = (now - query.queued_at) * 1000
                    self._batch_wait_ms += wait_ms
                    self._record("vector_query_batch_wait", wait_ms)
                    self._record_locked(query.locked_metrics, "face_identity_locked_query_batch", wait_ms)
                task = asyncio.create_task(self._execute(batch, now), name="alog-vector-query-batch")
                self._active.add(task)
                task.add_done_callback(self._batch_done)
                delay = None
            if len(self._active) >= self._executor.workers:
                delay = None
            try:
                if delay is None:
                    await self._changed.wait()
                else:
                    await asyncio.wait_for(self._changed.wait(), timeout=max(0, delay))
            except TimeoutError:
                pass

    def _batch_done(self, task: asyncio.Task) -> None:
        self._active.discard(task)
        if not task.cancelled():
            task.exception()
        self._changed.set()

    async def _execute(self, batch: list[_Query], selected_at: float) -> None:
        first = batch[0]
        try:
            results, error, sdk_started_at, sdk_finished_at = await self._executor.run(
                _search, first.client, [query.vector for query in batch], first.parameters,
                drain_on_cancel=True,
            )
            rpc_ms = (sdk_finished_at - sdk_started_at) * 1000
            self._counts["finished_rpcs"] += 1
            self._counts["finished_rpc_queries"] += len(batch)
            self._batch_histogram[len(batch)] += 1
            self._rpc_run_ms += rpc_ms
            self._record("vector_query_rpc_run", rpc_ms)
            for query in batch:
                # 已取消的调用方可能已释放身份锁，不能计入后续实际 SDK 工作。
                if not query.result.done():
                    query.sdk_finished_at = sdk_finished_at
                    self._record_locked(
                        query.locked_metrics, "face_identity_locked_query_executor",
                        (sdk_started_at - selected_at) * 1000,
                    )
                    self._record_locked(
                        query.locked_metrics, "face_identity_locked_query_sdk", rpc_ms,
                        failed=error is not None,
                    )
            if error is not None:
                raise error
            # 不允许 zip 静默截短，所有 query 都必须有独立的结果槽位。
            if not isinstance(results, (list, tuple)) or len(results) != len(batch) or any(
                not isinstance(hits, (list, tuple)) or any(not isinstance(hit, dict) for hit in hits)
                for hits in results
            ):
                raise ValueError("Milvus search result shape does not match query batch")
            self._counts["completed_queries"] += len(batch)
            for query, hits in zip(batch, results):
                if not query.result.done():
                    query.result.set_result(list(hits))
        except BaseException as error:
            self._counts["failed_queries"] += len(batch)
            for query in batch:
                if not query.result.done():
                    query.result.set_exception(error)
        finally:
            # run 的取消路径等待真实 SDK 完成；任何情况下都不提前归还名额。
            self._release(len(batch))

    def queue_snapshot(self) -> dict:
        return {
            "closed": self._closed, "capacity_queries": self._capacity,
            "batch_size": self._batch_size, "wait_ms": self._wait_seconds * 1000,
            "admission_waiters": self._waiting, "admitted_queries": self._admitted,
            "queued_queries": sum(map(len, self._groups.values())),
            "inflight_batches": len(self._active),
            "compatible_groups": len(self._groups),
            "admission_wait_total_ms": self._admission_ms,
            "batch_wait_total_ms": self._batch_wait_ms,
            "rpc_run_total_ms": self._rpc_run_ms,
            "batch_histogram": dict(self._batch_histogram), **dict(self._counts),
        }

    async def close(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("vector query coordinator belongs to a different event loop")
        if self._close_task is None:
            self._closed = True
            self._space.set()
            self._changed.set()
            self._close_task = asyncio.create_task(self._finish_close(), name="alog-vector-query-close")
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        if self._pump_task is not None:
            await self._pump_task
