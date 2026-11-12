import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Awaitable, TypeVar

import numpy as np
import grpc
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpcclient_aio
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from core.infer.contracts import ModelContract
from core.infer.metrics import active_inference_metrics
from core.infer.system_shared_memory import SystemSharedMemoryPool, SystemSharedMemorySlot
from core.performance import runtime_performance
from core.infer.errors import (
    InferenceInputError,
    InferenceQueueTimeout,
    InferenceResponseError,
    InferenceServerTimeout,
    InferenceUnavailable,
    TritonInferenceError,
)


T = TypeVar("T")


class AsyncClientPool:
    _NEW = "new"
    _STARTING = "starting"
    _OPEN = "open"
    _CLOSING = "closing"
    _CLOSED = "closed"

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        size: int,
        acquire_timeout: float,
    ) -> None:
        if type(size) is not int or size < 1:
            raise ValueError("client pool size must be at least 1")
        if (
            isinstance(acquire_timeout, bool)
            or not isinstance(acquire_timeout, (int, float))
            or not math.isfinite(acquire_timeout)
            or acquire_timeout <= 0
        ):
            raise ValueError(
                "client pool acquire timeout must be a positive finite number"
            )
        self._factory = factory
        self._size = size
        self._acquire_timeout = float(acquire_timeout)
        self._condition = asyncio.Condition()
        self._available: deque[Any] = deque()
        self._clients: list[Any] = []
        self._in_use: dict[int, Any] = {}
        self._state = self._NEW
        self._close_task: asyncio.Task[None] | None = None

    @staticmethod
    async def _close_clients(clients: list[Any]) -> list[BaseException]:
        errors: list[BaseException] = []
        for client in clients:
            try:
                await client.close()
            except BaseException as exc:
                errors.append(exc)
        return errors

    async def start(self) -> None:
        async with self._condition:
            if self._state == self._OPEN:
                return
            if self._state in (self._CLOSING, self._CLOSED):
                raise RuntimeError(f"client pool is {self._state}")

            self._state = self._STARTING
            created: list[Any] = []
            try:
                for _ in range(self._size):
                    created.append(self._factory())
            except BaseException:
                await self._close_clients(created)
                self._state = self._NEW
                self._condition.notify_all()
                raise

            self._clients = created
            self._available = deque(created)
            self._state = self._OPEN
            self._condition.notify_all()

    async def acquire(self) -> Any:
        await self.start()

        async def wait_for_client() -> Any:
            async with self._condition:
                while True:
                    if self._state != self._OPEN:
                        raise RuntimeError(f"client pool is {self._state}")
                    if self._available:
                        client = self._available.popleft()
                        self._in_use[id(client)] = client
                        return client
                    await self._condition.wait()

        try:
            return await asyncio.wait_for(
                wait_for_client(), timeout=self._acquire_timeout
            )
        except TimeoutError as exc:
            raise InferenceQueueTimeout(
                f"timed out waiting {self._acquire_timeout:g}s for a Triton client"
            ) from exc

    async def release(self, client: Any) -> None:
        async with self._condition:
            leased = self._in_use.get(id(client))
            if leased is not client:
                raise RuntimeError(
                    "client is not leased by this pool or was already released"
                )
            del self._in_use[id(client)]
            if self._state == self._OPEN:
                self._available.append(client)
            elif self._state != self._CLOSING:
                raise RuntimeError(f"client pool is {self._state}")
            self._condition.notify_all()

    @asynccontextmanager
    async def lease(self):
        client = await self.acquire()
        try:
            yield client
        finally:
            await self.release(client)

    async def _finish_close(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._in_use)
            clients = list(self._clients)

        errors = await self._close_clients(clients)

        async with self._condition:
            self._clients.clear()
            self._available.clear()
            self._state = self._CLOSED
            self._condition.notify_all()

        if errors:
            first_error = errors[0]
            raise RuntimeError(
                f"failed to close {len(errors)} Triton client(s): {first_error}"
            ) from first_error

    async def close(self) -> None:
        async with self._condition:
            if self._state == self._CLOSED:
                return
            if self._state == self._NEW:
                self._state = self._CLOSED
                self._condition.notify_all()
                return
            if self._state == self._OPEN:
                self._state = self._CLOSING
                self._available.clear()
                self._condition.notify_all()
                self._close_task = asyncio.create_task(self._finish_close())
            close_task = self._close_task

        if close_task is None:
            raise RuntimeError("client pool close state is inconsistent")
        await asyncio.shield(close_task)


class TritonTransport:
    def __init__(
        self,
        *,
        url: str,
        pool_size: int,
        acquire_timeout: float,
        infer_timeout: float,
        config_timeout: float,
        transient_retries: int,
        shm_slots: int = 0,
        shm_slot_bytes: int = 32 * 1024 * 1024,
        shm_copy_workers: int | None = None,
        client_factory: Callable[[str], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Triton URL must not be empty")
        for label, value in (
            ("infer_timeout", infer_timeout),
            ("config_timeout", config_timeout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{label} must be a positive finite number")
        if (
            isinstance(transient_retries, bool)
            or not isinstance(transient_retries, int)
            or transient_retries < 0
        ):
            raise ValueError("transient_retries must be a non-negative integer")

        if type(shm_slots) is not int or not 0 <= shm_slots <= 64:
            raise ValueError("shm_slots must be an integer between 0 and 64")
        if type(shm_slot_bytes) is not int or shm_slot_bytes < 1024 * 1024:
            raise ValueError("shm_slot_bytes must be an integer of at least 1 MiB")
        if shm_slots * shm_slot_bytes > 512 * 1024 * 1024:
            raise ValueError("SHM regions must use at most 512 MiB")
        if shm_copy_workers is not None and (
            type(shm_copy_workers) is not int
            or not 1 <= shm_copy_workers <= min(4, shm_slots)
        ):
            raise ValueError("shm_copy_workers must be between 1 and min(4, shm_slots)")
        self._shm_slots = shm_slots
        self._shm_slot_bytes = shm_slot_bytes
        self._shm_copy_workers = shm_copy_workers
        self._shm_pool: SystemSharedMemoryPool | None = None
        self._close_task: asyncio.Task[None] | None = None
        if client_factory is not None:
            factory = client_factory
        else:
            factory = lambda server_url: grpcclient_aio.InferenceServerClient(url=server_url)
        self._shm_client_factory = lambda: factory(url)
        self._pool = AsyncClientPool(
            lambda: factory(url),
            size=pool_size,
            acquire_timeout=acquire_timeout,
        )
        self._infer_timeout = float(infer_timeout)
        self._config_timeout = float(config_timeout)
        self._transient_retries = transient_retries
        self._logger = logger or logging.getLogger(__name__)

    async def start(self) -> None:
        await self._pool.start()
        if self._shm_slots:
            await self._get_shm_pool().start()

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._finish_close(), name="triton-transport-close"
            )
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        if self._shm_pool is not None:
            self._shm_pool.stop_borrowing()
        try:
            # 先等待真实 lease/RPC 排空并关闭推理 channel。
            await self._pool.close()
        finally:
            if self._shm_pool is not None:
                # 独立控制 channel 保留至注销；也等待尚在复制、未取得 client 的请求。
                await self._shm_pool.close()

    def _get_shm_pool(self) -> SystemSharedMemoryPool:
        if self._close_task is not None:
            raise RuntimeError("Triton transport is closing or closed")
        if self._shm_pool is None:
            self._shm_pool = SystemSharedMemoryPool(
                slots=self._shm_slots, slot_bytes=self._shm_slot_bytes,
                client_factory=self._shm_client_factory,
                config_timeout=self._config_timeout, logger=self._logger,
                copy_workers=self._shm_copy_workers,
            )
        return self._shm_pool

    def _can_share_raw_image(
        self, contract: ModelContract, inputs: dict[str, np.ndarray],
    ) -> bool:
        if contract.key not in {"face_det", "person_seg", "person_pose"}:
            return False
        if len(contract.inputs) != 1 or contract.inputs[0].name != "RAW_IMAGE":
            return False
        if set(inputs) != {"RAW_IMAGE"}:
            return False
        array = inputs["RAW_IMAGE"]
        if not isinstance(array, np.ndarray):
            return False
        # 不绕过 SDK 的 dtype 检查；不支持的输入继续由原 raw 路径报错。
        return (
            1024 * 1024 <= array.nbytes <= self._shm_slot_bytes
            and array.dtype.kind in "uif" and array.dtype.isnative
            and np_to_triton_dtype(array.dtype) == contract.inputs[0].datatype
        )

    def _log_client_wait(
        self,
        *,
        model: str | None,
        request_id: str,
        client_wait_ms: float,
        attempt: int,
    ) -> None:
        self._logger.debug(
            "triton_transport",
            extra={
                "event": "triton_transport",
                "model": model,
                "request_id": request_id,
                "client_wait_ms": client_wait_ms,
                "attempt": attempt,
            },
        )

    @staticmethod
    def _map_remote_error(exc: Exception) -> TritonInferenceError:
        if isinstance(exc, InferenceServerException):
            status = exc.status()
            if status == str(grpc.StatusCode.DEADLINE_EXCEEDED):
                return InferenceServerTimeout(str(exc))
            if status == str(grpc.StatusCode.UNAVAILABLE):
                return InferenceUnavailable(str(exc))
            return TritonInferenceError(str(exc))
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return InferenceServerTimeout(str(exc) or "Triton request timed out")
        return TritonInferenceError(str(exc))

    async def _invoke(
        self,
        operation: Callable[[Any], Awaitable[T]],
        *,
        model: str | None,
        request_id: str = "",
    ) -> T:
        metrics = active_inference_metrics.get()
        for attempt in range(1, self._transient_retries + 2):
            wait_started = time.perf_counter()
            try:
                try:
                    client = await self._pool.acquire()
                finally:
                    client_wait_ms = (time.perf_counter() - wait_started) * 1000.0
                    if metrics is not None:
                        metrics.client_wait_ms += client_wait_ms
            except InferenceQueueTimeout:
                self._log_client_wait(
                    model=model,
                    request_id=request_id,
                    client_wait_ms=client_wait_ms,
                    attempt=attempt,
                )
                raise

            self._log_client_wait(
                model=model,
                request_id=request_id,
                client_wait_ms=client_wait_ms,
                attempt=attempt,
            )
            rpc_started = time.perf_counter() if metrics is not None else None
            if metrics is not None:
                metrics.rpc_attempts += 1
            try:
                return await operation(client)
            except Exception as exc:
                mapped = self._map_remote_error(exc)
                if (
                    isinstance(mapped, InferenceUnavailable)
                    and attempt <= self._transient_retries
                ):
                    continue
                raise mapped from exc
            finally:
                if metrics is not None and rpc_started is not None:
                    metrics.rpc_ms = (metrics.rpc_ms or 0.0) + (
                        time.perf_counter() - rpc_started
                    ) * 1000.0
                await self._pool.release(client)
        raise AssertionError("unreachable")

    async def is_server_ready(self) -> bool:
        return await self._invoke(
            lambda client: client.is_server_ready(
                client_timeout=self._config_timeout
            ),
            model=None,
        )

    async def is_model_ready(self, model_name: str) -> bool:
        return await self._invoke(
            lambda client: client.is_model_ready(
                model_name=model_name,
                client_timeout=self._config_timeout,
            ),
            model=model_name,
        )

    async def get_model_config(self, model_name: str) -> dict[str, Any]:
        return await self._invoke(
            lambda client: client.get_model_config(
                model_name=model_name,
                as_json=True,
                client_timeout=self._config_timeout,
            ),
            model=model_name,
        )

    async def get_model_metadata(self, model_name: str) -> dict[str, Any]:
        return await self._invoke(
            lambda client: client.get_model_metadata(
                model_name=model_name,
                as_json=True,
                client_timeout=self._config_timeout,
            ),
            model=model_name,
        )

    def _build_inputs(
        self,
        contract: ModelContract,
        inputs: dict[str, np.ndarray],
        *,
        shared_memory: SystemSharedMemorySlot | None = None,
    ) -> list[Any]:
        expected_names = {tensor.name for tensor in contract.inputs}
        actual_names = set(inputs)
        differences: list[str] = []
        missing = expected_names - actual_names
        extra = actual_names - expected_names
        if missing:
            differences.append(f"missing input(s): {', '.join(sorted(missing))}")
        if extra:
            differences.append(f"extra input(s): {', '.join(sorted(extra))}")
        if differences:
            raise InferenceInputError(
                f"{contract.key}: " + "; ".join(differences)
            )

        infer_inputs = []
        for tensor in contract.inputs:
            array = inputs[tensor.name]
            if not isinstance(array, np.ndarray):
                raise InferenceInputError(
                    f"{contract.key}: input {tensor.name} must be a numpy array"
                )
            try:
                infer_input = grpcclient.InferInput(
                    tensor.name, array.shape, tensor.datatype
                )
                if shared_memory is not None and tensor.name == "RAW_IMAGE":
                    if (
                        array.dtype.kind not in "uif" or not array.dtype.isnative
                        or np_to_triton_dtype(array.dtype) != tensor.datatype
                        or array.nbytes > shared_memory.byte_size
                    ):
                        raise ValueError("RAW_IMAGE dtype or size is invalid for shared memory")
                    infer_input.set_shared_memory(shared_memory.name, array.nbytes)
                else:
                    infer_input.set_data_from_numpy(array)
            except Exception as exc:
                raise InferenceInputError(
                    f"{contract.key}: input {tensor.name} is invalid: {exc}"
                ) from exc
            infer_inputs.append(infer_input)
        return infer_inputs

    async def infer(
        self,
        contract: ModelContract,
        inputs: dict[str, np.ndarray],
        *,
        request_id: str,
    ) -> dict[str, np.ndarray]:
        metrics = active_inference_metrics.get()
        serialize_started = time.perf_counter() if metrics is not None else None
        slot: SystemSharedMemorySlot | None = None
        try:
            try:
                if self._shm_slots:
                    if self._can_share_raw_image(contract, inputs):
                        pool = self._get_shm_pool()
                        await pool.start()
                        slot = pool.borrow()
                        if slot is not None:
                            try:
                                await pool.copy(slot, inputs["RAW_IMAGE"])
                            except Exception as exc:
                                raise InferenceInputError(
                                    f"{contract.key}: input RAW_IMAGE could not be copied to shared memory"
                                ) from exc
                    if slot is None:
                        runtime_performance.record("triton_shm_fallback", 0)
                infer_inputs = self._build_inputs(contract, inputs, shared_memory=slot)
            finally:
                if metrics is not None and serialize_started is not None:
                    metrics.serialize_ms = (
                        time.perf_counter() - serialize_started
                    ) * 1000.0
            try:
                requested_outputs = [
                    grpcclient.InferRequestedOutput(tensor.name)
                    for tensor in contract.outputs
                ]
            except Exception as exc:
                raise InferenceInputError(
                    f"{contract.key}: output request construction failed: {exc}"
                ) from exc

            registration_repair_attempted = False

            async def send(client: Any) -> Any:
                nonlocal registration_repair_attempted
                attempt_inputs = infer_inputs
                if slot is not None and slot.poisoned:
                    # 前次失败不证明服务端停止读。重试发送 raw，旧 region 永不再写。
                    attempt_inputs = self._build_inputs(contract, inputs)
                    runtime_performance.record("triton_shm_fallback", 0)
                elif slot is not None:
                    runtime_performance.record("triton_shm_used", 0)
                try:
                    try:
                        return await client.infer(
                            model_name=contract.model_name, inputs=attempt_inputs,
                            outputs=requested_outputs, request_id=request_id,
                            client_timeout=self._infer_timeout,
                        )
                    except InferenceServerException as exc:
                        # 只认服务端在读取图片前对当前 own region 的明确拒绝。
                        # 超时、取消、其他 NOT_FOUND 和已隔离的槽都不能进入修复。
                        if (slot is None or slot.poisoned or registration_repair_attempted
                                or exc.status() != str(grpc.StatusCode.NOT_FOUND)
                                or exc.message() != f"Unable to find shared memory region: '{slot.name}'"):
                            raise
                        registration_repair_attempted = True
                        if not await pool.reregister_missing(slot):
                            raise
                        runtime_performance.record("triton_shm_used", 0)
                        if metrics is not None:
                            metrics.rpc_attempts += 1
                        # 保留原 tensor、name、字节内容和独占租约，有限重发一次。
                        return await client.infer(
                            model_name=contract.model_name, inputs=attempt_inputs,
                            outputs=requested_outputs, request_id=request_id,
                            client_timeout=self._infer_timeout,
                        )
                except BaseException:
                    if slot is not None:
                        pool.poison(slot)
                    raise

            result = await self._invoke(
                send, model=contract.model_name, request_id=request_id,
            )
        finally:
            if slot is not None:
                pool.release(slot)

        outputs: dict[str, np.ndarray] = {}
        for tensor in contract.outputs:
            try:
                value = result.as_numpy(tensor.name)
            except Exception as exc:
                raise InferenceResponseError(
                    f"{contract.key}: output {tensor.name} could not be read: {exc}"
                ) from exc
            if value is None:
                raise InferenceResponseError(
                    f"{contract.key}: output {tensor.name} is missing"
                )
            if not isinstance(value, np.ndarray):
                raise InferenceResponseError(
                    f"{contract.key}: output {tensor.name} must be a numpy array"
                )
            outputs[tensor.name] = value
        return outputs
