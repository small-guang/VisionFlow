import asyncio
import math
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from loguru import logger as default_logger

from core.infer.capabilities import ModelCapability
from core.infer.contracts import ModelContract
from core.infer.errors import InferenceQueueTimeout, InferenceUnavailable
from core.infer.metrics import (
    InferenceMetrics,
    active_inference_metrics,
    tensor_payload_bytes,
)
from core.performance import runtime_performance


@dataclass(frozen=True)
class NormalizedRequest:
    inputs: dict[str, np.ndarray]
    compatibility_key: tuple
    context: object | None = None


@dataclass
class PendingRequest:
    request_id: str
    normalized: NormalizedRequest
    future: asyncio.Future
    queued_at: float


@dataclass
class _PackedBatch:
    inputs: dict[str, np.ndarray]
    error: BaseException | None
    run_ms: float
    queue_wait_ms: float


class BatchAdapter(Protocol):
    contract: ModelContract

    def pack(
        self, requests: list[NormalizedRequest]
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def validate_outputs(
        self,
        outputs: dict[str, np.ndarray],
        batch_size: int,
    ) -> None:
        raise NotImplementedError

    def unpack_one(
        self,
        outputs: dict[str, np.ndarray],
        index: int,
        request: NormalizedRequest,
    ) -> object:
        raise NotImplementedError


class ModelBatchScheduler:
    def __init__(
        self,
        *,
        adapter: BatchAdapter,
        capability: ModelCapability,
        transport: object,
        executor: object,
        wait_ms: float,
        queue_timeout: float,
        logger: object | None = None,
        cpu_executor: object | None = None,
    ) -> None:
        if (
            isinstance(wait_ms, bool)
            or not isinstance(wait_ms, (int, float))
            or not math.isfinite(wait_ms)
            or wait_ms < 0
        ):
            raise ValueError("wait_ms must be a non-negative finite number")
        if (
            isinstance(queue_timeout, bool)
            or not isinstance(queue_timeout, (int, float))
            or not math.isfinite(queue_timeout)
            or queue_timeout <= 0
        ):
            raise ValueError("queue_timeout must be a positive finite number")
        if (
            type(capability.server_max_batch) is not int
            or capability.server_max_batch < 1
        ):
            raise ValueError("server_max_batch must be an integer at least 1")
        if (
            type(capability.target_batch) is not int
            or capability.target_batch < 1
            or capability.target_batch > capability.server_max_batch
        ):
            raise ValueError(
                "target_batch must be an integer between 1 and "
                "server_max_batch"
            )
        if capability.key != adapter.contract.key:
            raise ValueError(
                "adapter and capability must describe the same model key"
            )

        self.adapter = adapter
        self.capability = capability
        self.transport = transport
        self.executor = executor
        self.cpu_executor = cpu_executor
        self.wait_ms = float(wait_ms)
        self.queue_timeout = float(queue_timeout)
        self._logger = logger or default_logger
        self._lock = asyncio.Lock()
        self._buckets: dict[tuple, list[PendingRequest]] = {}
        self._bucket_tokens: dict[tuple, object] = {}
        self._batch_timers: dict[tuple, asyncio.TimerHandle] = {}
        self._expiration_timers: dict[str, asyncio.TimerHandle] = {}
        self._active_dispatches: set[asyncio.Task] = set()
        self._dispatch_pending: dict[
            asyncio.Task, tuple[PendingRequest, ...]
        ] = {}
        self._request_dispatches: dict[str, asyncio.Task] = {}
        self._closed = False
        self._close_task: asyncio.Task | None = None

    def queue_snapshot(self) -> dict[str, int | float | bool]:
        """在所属事件循环读取尚未派发的组批样本，不暴露请求内容。

        等待数按有效样本计数，与 executor 的批次数不可直接相加。
        已从 bucket 派发的请求由 executor 观测，不再计入待组批样本。
        """
        return {
            "closed": self._closed,
            "batch_waiting_samples": sum(
                not item.future.done()
                for bucket in self._buckets.values()
                for item in bucket
            ),
            "batch_target_samples": self.capability.target_batch,
            "server_max_batch_samples": self.capability.server_max_batch,
            "batch_wait_ms": self.wait_ms,
        }

    def _cancel_expiration_locked(self, request_id: str) -> None:
        timer = self._expiration_timers.pop(request_id, None)
        if timer is not None:
            timer.cancel()

    def _cancel_batch_timer_locked(self, compatibility_key: tuple) -> None:
        timer = self._batch_timers.pop(compatibility_key, None)
        if timer is not None:
            timer.cancel()

    def _detach_locked(
        self, compatibility_key: tuple
    ) -> list[PendingRequest]:
        pending = self._buckets.pop(compatibility_key, [])
        self._bucket_tokens.pop(compatibility_key, None)
        self._cancel_batch_timer_locked(compatibility_key)
        live: list[PendingRequest] = []
        for item in pending:
            self._cancel_expiration_locked(item.request_id)
            if not item.future.done():
                live.append(item)
        if live:
            task = asyncio.create_task(self._dispatch(live))
            self._active_dispatches.add(task)
            self._dispatch_pending[task] = tuple(live)
            for item in live:
                self._request_dispatches[item.request_id] = task
            task.add_done_callback(self._dispatch_done)
        return live

    def _dispatch_done(self, task: asyncio.Task) -> None:
        self._active_dispatches.discard(task)
        pending = self._dispatch_pending.pop(task, ())
        for item in pending:
            if self._request_dispatches.get(item.request_id) is task:
                self._request_dispatches.pop(item.request_id, None)
        if not task.cancelled():
            task.exception()

    def _schedule_flush(
        self, compatibility_key: tuple, bucket_token: object
    ) -> None:
        asyncio.create_task(self._flush(compatibility_key, bucket_token))

    def _schedule_expiration(
        self, compatibility_key: tuple, request_id: str
    ) -> None:
        asyncio.create_task(self._expire(compatibility_key, request_id))

    async def _flush(
        self, compatibility_key: tuple, bucket_token: object
    ) -> None:
        async with self._lock:
            if self._bucket_tokens.get(compatibility_key) is not bucket_token:
                return
            self._detach_locked(compatibility_key)

    async def _expire(
        self, compatibility_key: tuple, request_id: str
    ) -> None:
        expired: PendingRequest | None = None
        async with self._lock:
            bucket = self._buckets.get(compatibility_key)
            if bucket is None:
                return
            for index, item in enumerate(bucket):
                if item.request_id == request_id:
                    expired = bucket.pop(index)
                    break
            self._expiration_timers.pop(request_id, None)
            if not bucket:
                self._buckets.pop(compatibility_key, None)
                self._bucket_tokens.pop(compatibility_key, None)
                self._cancel_batch_timer_locked(compatibility_key)

        if expired is not None and not expired.future.done():
            expired.future.set_exception(
                InferenceQueueTimeout(
                    f"{self.adapter.contract.key}: request expired in batch queue"
                )
            )

    async def _remove_cancelled(
        self, compatibility_key: tuple, request_id: str
    ) -> None:
        async with self._lock:
            bucket = self._buckets.get(compatibility_key)
            if bucket is not None:
                for index, item in enumerate(bucket):
                    if item.request_id == request_id:
                        bucket.pop(index)
                        self._cancel_expiration_locked(request_id)
                        break
                if not bucket:
                    self._buckets.pop(compatibility_key, None)
                    self._bucket_tokens.pop(compatibility_key, None)
                    self._cancel_batch_timer_locked(compatibility_key)

            dispatch = self._request_dispatches.get(request_id)
            if dispatch is not None and not dispatch.done():
                pending = self._dispatch_pending.get(dispatch, ())
                if pending and all(item.future.done() for item in pending):
                    dispatch.cancel()

    @staticmethod
    def _fail_live(
        pending: list[PendingRequest], error: BaseException
    ) -> None:
        for item in pending:
            if not item.future.done():
                item.future.set_exception(error)

    def _log_batch(
        self,
        *,
        pending: list[PendingRequest],
        queue_wait_ms: float,
        packed_inputs: dict[str, np.ndarray],
        error_class: str | None,
        infer_ms: float,
        total_ms: float,
        metrics: InferenceMetrics | None = None,
    ) -> None:
        try:
            metric_fields = (metrics or InferenceMetrics()).log_fields(queue_wait_ms)
            runtime_performance.inference_completed(
                self.adapter.contract.key,
                batch_size=len(pending),
                batch_wait_ms=queue_wait_ms,
                executor_queue_wait_ms=metric_fields["executor_queue_wait_ms"],
                rpc_ms=metric_fields["rpc_ms"],
                failed=error_class is not None,
            )
            status = "ok" if error_class is None else "error"
            request_id = ",".join(item.request_id for item in pending)
            batch_logger = self._logger.bind(
                model=self.adapter.contract.key,
                model_name=self.adapter.contract.model_name,
                status=status,
                error_class=error_class,
                api_contract_version=self.capability.api_contract_version,
                request_ids=[item.request_id for item in pending],
                batch_size=len(pending),
                server_max_batch=self.capability.server_max_batch,
                target_batch=self.capability.target_batch,
                queue_wait_ms=round(queue_wait_ms, 3),
                input_shapes={
                    name: value.shape for name, value in packed_inputs.items()
                },
                infer_ms=round(infer_ms, 3),
                total_ms=round(total_ms, 3),
                end_to_end_ms=round(queue_wait_ms + total_ms, 3),
                **metric_fields,
            )
            # Default sinks print only {message}; keep the fields visible there
            # as well as in Loguru extras used by structured log collectors.
            summary = ", ".join(
                f"{name}={value}"
                for name, value in {
                    "model": self.adapter.contract.key,
                    "batch_size": len(pending),
                    **metric_fields,
                    "request_id": request_id,
                }.items()
            )
            if error_class is None:
                getattr(batch_logger, "debug", batch_logger.info)(
                    "Triton batch completed: status=ok, infer_ms={:.3f}, "
                    "total_ms={:.3f}, {}",
                    infer_ms,
                    total_ms,
                    summary,
                )
            else:
                getattr(batch_logger, "warning", batch_logger.info)(
                    "Triton batch completed: status=error, error_class={}, "
                    "infer_ms={:.3f}, total_ms={:.3f}, {}",
                    error_class,
                    infer_ms,
                    total_ms,
                    summary,
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            return

    @staticmethod
    def _pack_copy_is_large(requests: list[NormalizedRequest]) -> bool:
        # 连续 B=1 只增加 batch 维度；BYTES pack 只填指针，不复制图片载荷。
        # 只有至少 1 MiB 的真实 numeric 堆叠/连续化才值得一次线程往返。
        copying_bytes = 0
        for request in requests:
            for value in request.inputs.values():
                if (
                    isinstance(value, np.ndarray)
                    and value.dtype.kind in "biufc"
                    and (len(requests) > 1 or not value.flags.c_contiguous)
                ):
                    copying_bytes += value.nbytes
                    if copying_bytes >= 1024 * 1024:
                        return True
        return False

    def _pack_on_cpu(
        self, requests: list[NormalizedRequest], submitted_at: float,
    ) -> _PackedBatch:
        # 工作线程只返回计时值，聚合与请求指标都由事件循环写入。
        started = time.perf_counter()
        inputs: dict[str, np.ndarray] = {}
        error = None
        try:
            inputs = self.adapter.pack(requests)
        except BaseException as caught:
            error = caught
        return _PackedBatch(
            inputs, error, (time.perf_counter() - started) * 1000.0,
            (started - submitted_at) * 1000.0,
        )

    async def _pack_requests(
        self, requests: list[NormalizedRequest], metrics: InferenceMetrics,
    ) -> dict[str, np.ndarray]:
        if self.cpu_executor is not None and self._pack_copy_is_large(requests):
            metrics.pack_queue_wait_ms = None
            outcome = await self.cpu_executor.run(
                self._pack_on_cpu, requests, time.perf_counter(),
            )
            metrics.pack_ms = outcome.run_ms
            metrics.pack_queue_wait_ms = outcome.queue_wait_ms
            if outcome.error is not None:
                raise outcome.error
            return outcome.inputs
        started = time.perf_counter()
        try:
            return self.adapter.pack(requests)
        finally:
            metrics.pack_ms = (time.perf_counter() - started) * 1000.0

    async def _dispatch(self, pending: list[PendingRequest]) -> None:
        dispatch_started = time.perf_counter()
        oldest_queued_at = min(item.queued_at for item in pending)
        deadline = oldest_queued_at + self.queue_timeout
        queue_wait_ms = (dispatch_started - oldest_queued_at) * 1000.0
        packed_inputs: dict[str, np.ndarray] = {}
        infer_ms = 0.0
        metrics = InferenceMetrics()
        executor_submitted_at = time.perf_counter()
        operation_started = False
        operation_finished = False
        operation_error_class: str | None = None

        try:
            async def infer_once():
                nonlocal infer_ms, packed_inputs, operation_started, operation_finished
                nonlocal operation_error_class
                operation_started = True
                try:
                    pack_started = time.perf_counter()
                    metrics.executor_queue_wait_ms = (
                        pack_started - executor_submitted_at
                    ) * 1000.0
                    packed_inputs = await self._pack_requests(
                        [item.normalized for item in pending], metrics,
                    )
                    metrics.request_bytes = tensor_payload_bytes(packed_inputs)
                    infer_started = time.perf_counter()
                    token = active_inference_metrics.set(metrics)
                    try:
                        response = await self.transport.infer(
                            self.adapter.contract,
                            packed_inputs,
                            request_id=",".join(item.request_id for item in pending),
                        )
                        metrics.response_bytes = tensor_payload_bytes(response)
                        return response
                    finally:
                        active_inference_metrics.reset(token)
                        infer_ms = (time.perf_counter() - infer_started) * 1000.0
                except BaseException as error:
                    operation_error_class = type(error).__name__
                    raise
                finally:
                    operation_finished = True
                    if metrics.caller_cancelled:
                        # A real executor keeps an in-flight operation alive
                        # after its submitter is cancelled. It owns the one
                        # terminal record once all transfer measurements exist.
                        self._log_batch(
                            pending=pending, queue_wait_ms=queue_wait_ms,
                            packed_inputs=packed_inputs,
                            error_class=operation_error_class or "CancelledError",
                            infer_ms=infer_ms,
                            total_ms=(time.perf_counter() - dispatch_started) * 1000.0,
                            metrics=metrics,
                        )

            outputs = await self.executor.submit(
                self.adapter.contract.key,
                infer_once,
                deadline=deadline,
            )

            self.adapter.validate_outputs(outputs, len(pending))
        except asyncio.CancelledError:
            metrics.caller_cancelled = True
            for item in pending:
                if not item.future.done():
                    item.future.cancel()
            if operation_started and not operation_finished:
                raise
            total_ms = (time.perf_counter() - dispatch_started) * 1000.0
            if metrics.executor_queue_wait_ms is None:
                metrics.executor_queue_wait_ms = max(
                    0.0, total_ms - (executor_submitted_at - dispatch_started) * 1000.0
                )
            self._log_batch(
                pending=pending, queue_wait_ms=queue_wait_ms,
                packed_inputs=packed_inputs,
                error_class=operation_error_class or "CancelledError",
                infer_ms=infer_ms, total_ms=total_ms, metrics=metrics,
            )
            raise
        except BaseException as exc:
            total_ms = (time.perf_counter() - dispatch_started) * 1000.0
            if metrics.executor_queue_wait_ms is None:
                metrics.executor_queue_wait_ms = max(
                    0.0, total_ms - (executor_submitted_at - dispatch_started) * 1000.0
                )
            self._fail_live(pending, exc)
            self._log_batch(
                pending=pending,
                queue_wait_ms=queue_wait_ms,
                packed_inputs=packed_inputs,
                error_class=type(exc).__name__,
                infer_ms=infer_ms,
                total_ms=total_ms,
                metrics=metrics,
            )
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return

        unpack_error_class: str | None = None
        for index, item in enumerate(pending):
            if item.future.done():
                continue
            try:
                result = self.adapter.unpack_one(
                    outputs, index, item.normalized
                )
            except asyncio.CancelledError:
                if unpack_error_class is None:
                    unpack_error_class = "CancelledError"
                item.future.cancel()
            except (KeyboardInterrupt, SystemExit) as exc:
                if unpack_error_class is None:
                    unpack_error_class = type(exc).__name__
                self._fail_live(pending[index:], exc)
                total_ms = (
                    time.perf_counter() - dispatch_started
                ) * 1000.0
                self._log_batch(
                    pending=pending,
                    queue_wait_ms=queue_wait_ms,
                    packed_inputs=packed_inputs,
                    error_class=unpack_error_class,
                    infer_ms=infer_ms,
                    total_ms=total_ms,
                    metrics=metrics,
                )
                raise
            except BaseException as exc:
                if unpack_error_class is None:
                    unpack_error_class = type(exc).__name__
                item.future.set_exception(exc)
            else:
                item.future.set_result(result)

        total_ms = (time.perf_counter() - dispatch_started) * 1000.0
        self._log_batch(
            pending=pending,
            queue_wait_ms=queue_wait_ms,
            packed_inputs=packed_inputs,
            error_class=unpack_error_class,
            infer_ms=infer_ms,
            total_ms=total_ms,
            metrics=metrics,
        )

    async def submit(self, normalized: NormalizedRequest) -> object:
        loop = asyncio.get_running_loop()
        pending = PendingRequest(
            request_id=uuid.uuid4().hex,
            normalized=normalized,
            future=loop.create_future(),
            queued_at=time.perf_counter(),
        )
        compatibility_key = normalized.compatibility_key

        async with self._lock:
            if self._closed:
                raise InferenceUnavailable(
                    f"{self.adapter.contract.key}: batch scheduler is closed"
                )
            bucket = self._buckets.get(compatibility_key)
            if bucket is None:
                bucket = []
                self._buckets[compatibility_key] = bucket
                self._bucket_tokens[compatibility_key] = object()
            bucket.append(pending)
            self._expiration_timers[pending.request_id] = loop.call_later(
                self.queue_timeout,
                self._schedule_expiration,
                compatibility_key,
                pending.request_id,
            )
            if len(bucket) == 1:
                self._batch_timers[compatibility_key] = loop.call_later(
                    self.wait_ms / 1000.0,
                    self._schedule_flush,
                    compatibility_key,
                    self._bucket_tokens[compatibility_key],
                )
            if len(bucket) >= self.capability.target_batch:
                self._detach_locked(compatibility_key)

        try:
            return await pending.future
        except asyncio.CancelledError:
            pending.future.cancel()
            await asyncio.shield(
                self._remove_cancelled(
                    compatibility_key, pending.request_id
                )
            )
            raise

    async def _finish_close(
        self,
        pending: list[PendingRequest],
        active_dispatches: list[asyncio.Task],
    ) -> None:
        error = InferenceUnavailable(
            f"{self.adapter.contract.key}: batch scheduler is closed"
        )
        self._fail_live(pending, error)
        if active_dispatches:
            await asyncio.gather(*active_dispatches, return_exceptions=True)

    async def close(self) -> None:
        async with self._lock:
            if self._close_task is None:
                self._closed = True
                for timer in self._batch_timers.values():
                    timer.cancel()
                self._batch_timers.clear()
                for timer in self._expiration_timers.values():
                    timer.cancel()
                self._expiration_timers.clear()
                pending = [
                    item
                    for bucket in self._buckets.values()
                    for item in bucket
                ]
                self._buckets.clear()
                self._bucket_tokens.clear()
                active_dispatches = list(self._active_dispatches)
                self._close_task = asyncio.create_task(
                    self._finish_close(pending, active_dispatches)
                )
            close_task = self._close_task

        await asyncio.shield(close_task)
