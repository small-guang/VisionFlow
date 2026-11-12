from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from loguru import logger

from core.infer.adapters import ModelAdapter, build_adapters
from core.infer.batching import ModelBatchScheduler
from core.infer.capabilities import (
    ModelCapability,
    discover_all_capabilities,
)
from core.infer.contracts import MODEL_CONTRACTS
from core.infer.errors import InferenceUnavailable, TritonContractError
from core.infer.executor import FairInferenceExecutor
from core.infer.results import (
    FaceDetection,
    FaceQualityResult,
    FeatureResult,
    PersonPose,
    PersonSegment,
)
from core.infer.transport import TritonTransport


class UnifiedTritonClient:
    """Business-facing facade over Triton's transport and tensor contracts."""

    _NEW = "new"
    _STARTING = "starting"
    _OPEN = "open"
    _CLOSING = "closing"
    _CLOSED = "closed"

    def __init__(
        self,
        *,
        settings: Any | None = None,
        transport: Any | None = None,
        executor: Any | None = None,
        cpu_executor: Any | None = None,
    ) -> None:
        if settings is None:
            from core.config import settings as runtime_settings

            settings = runtime_settings

        self.settings = settings
        if transport is None:
            shm_options = {}
            copy_workers = getattr(settings, "TRITON_SYSTEM_SHM_COPY_WORKERS", None)
            if copy_workers is not None:
                shm_options["shm_copy_workers"] = copy_workers
            transport = TritonTransport(
                url=settings.TRITON_SERVER_URL,
                pool_size=settings.TRITON_CLIENT_POOL_SIZE,
                acquire_timeout=(
                    settings.TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS
                ),
                infer_timeout=settings.TRITON_INFER_TIMEOUT_SECONDS,
                config_timeout=settings.TRITON_CONFIG_TIMEOUT_SECONDS,
                transient_retries=settings.TRITON_TRANSIENT_RETRIES,
                shm_slots=getattr(settings, "TRITON_SYSTEM_SHM_SLOTS", 0),
                shm_slot_bytes=getattr(settings, "TRITON_SYSTEM_SHM_SLOT_BYTES", 33554432),
                **shm_options,
            )
        self.transport = transport
        if executor is None:
            executor = FairInferenceExecutor(
                concurrency=settings.TRITON_EXECUTOR_CONCURRENCY,
                queue_capacity=settings.TRITON_EXECUTOR_QUEUE_CAPACITY,
            )
        self.executor = executor
        # 借用应用的图片 CPU 资源；其生命周期由应用持有。
        self.cpu_executor = cpu_executor
        self._capabilities: dict[str, ModelCapability] = {}
        self._schedulers: dict[str, ModelBatchScheduler] = {}
        self._capabilities_view: Mapping[str, ModelCapability] = (
            MappingProxyType(self._capabilities)
        )
        self._schedulers_view: Mapping[str, ModelBatchScheduler] = (
            MappingProxyType(self._schedulers)
        )
        self._adapters: dict[str, ModelAdapter] = {}
        self._state = self._NEW
        self._lifecycle_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._close_error: RuntimeError | None = None

    @property
    def capabilities(self) -> Mapping[str, ModelCapability]:
        return self._capabilities_view

    @property
    def schedulers(self) -> Mapping[str, ModelBatchScheduler]:
        return self._schedulers_view

    def queue_snapshot(self) -> dict[str, Any]:
        """返回本地推理队列快照；须在所属事件循环同步调用。

        不查询 Triton 服务端。executor active 包含本地资源等待，不等同于
        GPU 运行批次。注入的 executor 无此接口时用 None 表示无法观测。
        """
        snapshot_provider = getattr(self.executor, "queue_snapshot", None)
        executor = snapshot_provider() if callable(snapshot_provider) else None
        executor_models = executor["models"] if executor is not None else {}
        models = {}
        for key, scheduler in self._schedulers.items():
            model_snapshot = scheduler.queue_snapshot()
            executor_model = executor_models.get(key, {})
            models[key] = {
                **model_snapshot,
                "executor_queued_batches": (
                    executor_model.get("queued_batches", 0)
                    if executor is not None else None
                ),
                "executor_active_batches": (
                    executor_model.get("active_batches", 0)
                    if executor is not None else None
                ),
            }
        return {
            "state": self._state,
            "batch_waiting_samples": sum(
                model["batch_waiting_samples"] for model in models.values()
            ),
            "executor": executor,
            "models": models,
        }

    def _batch_targets(self) -> dict[str, int]:
        return {
            key: self.settings.TRITON_BATCH_TARGETS.get(
                key, self.settings.TRITON_BATCH_TARGET_DEFAULT
            )
            for key in MODEL_CONTRACTS
        }

    def _batch_waits(self) -> dict[str, float]:
        return {
            key: self.settings.TRITON_BATCH_WAIT_MS.get(
                key, self.settings.TRITON_BATCH_WAIT_MS_DEFAULT
            )
            for key in MODEL_CONTRACTS
        }

    @staticmethod
    def _validate_capability_keys(
        capabilities: Mapping[str, ModelCapability],
    ) -> None:
        expected = set(MODEL_CONTRACTS)
        actual = set(capabilities)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if not missing and not extra:
            return
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise TritonContractError(
            "Triton discovery registry does not match model contracts: "
            + ", ".join(details)
        )

    @staticmethod
    async def _close_resources(
        schedulers: list[ModelBatchScheduler],
        executor: Any,
        transport: Any,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        for scheduler in schedulers:
            try:
                await scheduler.close()
            except BaseException as error:
                errors.append(error)
        try:
            await executor.close()
        except BaseException as error:
            errors.append(error)
        try:
            await transport.close()
        except BaseException as error:
            errors.append(error)
        return errors

    @staticmethod
    async def _wait_for_task_uninterruptibly(
        task: asyncio.Task[list[BaseException]],
    ) -> list[BaseException]:
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    def _log_startup_summary(self) -> None:
        try:
            summary = [
                {
                    "model": key,
                    "api_contract_version": capability.api_contract_version,
                    "server_max_batch": capability.server_max_batch,
                    "target_batch": capability.target_batch,
                    "wait_ms": self._schedulers[key].wait_ms,
                }
                for key, capability in self._capabilities.items()
            ]
            logger.bind(models=summary).info(
                "Triton inference facade initialized: model_count={}",
                len(summary),
            )
        except Exception:
            return

    async def initialize(self) -> None:
        async with self._lifecycle_lock:
            if self._state == self._OPEN:
                return
            if self._state != self._NEW:
                raise InferenceUnavailable(
                    f"Triton inference facade is {self._state}"
                )

            self._state = self._STARTING
            local_schedulers: dict[str, ModelBatchScheduler] = {}
            try:
                await self.transport.start()
                if not await self.transport.is_server_ready():
                    raise InferenceUnavailable("Triton server is not ready")

                targets = self._batch_targets()
                waits = self._batch_waits()
                capabilities = await discover_all_capabilities(
                    self.transport,
                    MODEL_CONTRACTS,
                    targets,
                )
                self._validate_capability_keys(capabilities)
                adapters = build_adapters()
                await self.executor.start()
                for key in MODEL_CONTRACTS:
                    local_schedulers[key] = ModelBatchScheduler(
                        adapter=adapters[key],
                        capability=capabilities[key],
                        transport=self.transport,
                        executor=self.executor,
                        wait_ms=waits[key],
                        queue_timeout=self.settings.TRITON_QUEUE_TIMEOUT_SECONDS,
                        cpu_executor=self.cpu_executor,
                    )
            except BaseException:
                cleanup_task = asyncio.create_task(
                    self._close_resources(
                        list(local_schedulers.values()),
                        self.executor,
                        self.transport,
                    ),
                    name="triton-facade-startup-cleanup",
                )
                cleanup_errors = await self._wait_for_task_uninterruptibly(
                    cleanup_task
                )
                self._state = self._CLOSED
                if cleanup_errors:
                    try:
                        logger.bind(
                            error_classes=[
                                type(error).__name__ for error in cleanup_errors
                            ]
                        ).error(
                            "Triton facade startup cleanup failed: error_count={}",
                            len(cleanup_errors),
                        )
                    except Exception:
                        pass
                raise

            self._adapters = adapters
            self._capabilities.update(capabilities)
            self._schedulers.update(local_schedulers)
            self._state = self._OPEN
            self._log_startup_summary()

    async def _finish_close(
        self, schedulers: list[ModelBatchScheduler]
    ) -> None:
        errors = await self._close_resources(
            schedulers,
            self.executor,
            self.transport,
        )
        close_error: RuntimeError | None = None
        if errors:
            first_error = errors[0]
            close_error = RuntimeError(
                f"failed to close {len(errors)} Triton facade resource(s): "
                f"{first_error}"
            )
            close_error.__cause__ = first_error
            try:
                logger.bind(
                    error_classes=[
                        type(error).__name__ for error in errors
                    ],
                    error_count=len(errors),
                ).error("Triton inference facade close failed")
            except Exception:
                pass
        async with self._lifecycle_lock:
            self._close_error = close_error
            self._state = self._CLOSED
        if close_error is not None:
            raise close_error

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._state == self._CLOSED:
                if self._close_error is not None:
                    raise self._close_error
                return
            if self._close_task is None:
                self._state = self._CLOSING
                schedulers = list(self._schedulers.values())
                self._capabilities.clear()
                self._schedulers.clear()
                self._adapters = {}
                self._close_task = asyncio.create_task(
                    self._finish_close(schedulers)
                )
                self._close_task.add_done_callback(
                    self._consume_task_exception
                )
            close_task = self._close_task

        await asyncio.shield(close_task)

    async def _submit(self, key: str, *args: object) -> object:
        # Keep input owners alive through packing and the RPC. Callers must
        # leave numeric input arrays unchanged until this await completes.
        if self._state != self._OPEN:
            raise InferenceUnavailable(
                f"Triton inference facade is not initialized ({self._state})"
            )
        adapter = self._adapters[key]
        scheduler = self._schedulers[key]
        normalized = adapter.normalize(*args)
        return await scheduler.submit(normalized)

    async def extract_image_features(self, image_data: bytes) -> FeatureResult:
        return cast(
            FeatureResult, await self._submit("clip_image", image_data)
        )

    async def extract_face_features(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> FeatureResult:
        return cast(
            FeatureResult,
            await self._submit("face_rec", image, landmarks),
        )

    async def extract_tbps_image_features(
        self, image_data: bytes | np.ndarray
    ) -> FeatureResult:
        return cast(
            FeatureResult, await self._submit("tbps_image", image_data)
        )

    async def extract_sbs_features(
        self, image_data: bytes | np.ndarray
    ) -> FeatureResult:
        return cast(
            FeatureResult, await self._submit("sbs_image", image_data)
        )

    async def extract_face_quality(
        self, face_image: np.ndarray, landmarks: np.ndarray
    ) -> FaceQualityResult:
        return cast(
            FaceQualityResult,
            await self._submit("face_qa", face_image, landmarks),
        )

    async def extract_face_detections(
        self, image: np.ndarray
    ) -> list[FaceDetection]:
        return cast(
            list[FaceDetection], await self._submit("face_det", image)
        )

    async def extract_face_keypoints(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        return cast(
            np.ndarray, await self._submit("face_kps", image, bbox)
        )

    async def extract_person_segments(
        self, image: np.ndarray
    ) -> list[PersonSegment]:
        return cast(
            list[PersonSegment], await self._submit("person_seg", image)
        )

    async def extract_person_poses(
        self, image: np.ndarray
    ) -> list[PersonPose]:
        return cast(
            list[PersonPose], await self._submit("person_pose", image)
        )


__all__ = ["UnifiedTritonClient"]
