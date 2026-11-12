"""应用装配与资源生命周期；导入本模块不连接外部服务。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from core.admission import AdmissionController, FrameBudget


class OutputNamespacedStorage:
    """历史原图仍读取原key，测试裁图可写入独立命名空间。"""

    def __init__(self, storage, prefix: str):
        self.storage = storage
        self.prefix = prefix.strip("/")

    async def get_object(self, object_name, thumbnail_params=None):
        return await self.storage.get_object(object_name, thumbnail_params)

    async def upload_object(self, image_data, object_name, headers=None):
        key = f"{self.prefix}/{object_name.lstrip('/')}" if self.prefix else object_name
        return await self.storage.upload_object(image_data, key, headers)


def _resources(config):
    from core.execution import ExecutionResources
    from core.performance import runtime_performance

    return ExecutionResources(
        image_workers=config.IMAGE_CPU_WORKERS, storage_workers=config.STORAGE_IO_WORKERS,
        storage_download_reserved=config.STORAGE_DOWNLOAD_RESERVED_WORKERS,
        search_workers=config.MILVUS_SEARCH_WORKERS, write_workers=config.MILVUS_WRITE_WORKERS,
        query_batch_size=config.MILVUS_QUERY_BATCH_SIZE,
        query_wait_ms=config.MILVUS_QUERY_WAIT_MS,
        query_capacity=config.MILVUS_QUERY_CAPACITY,
        metrics=runtime_performance,
    )


def _triton(config, resources):
    from core.infer import UnifiedTritonClient
    return UnifiedTritonClient(settings=config, cpu_executor=resources.image_cpu)


def _databases(config, resources):
    from db.zilliz import DatabaseServices
    return DatabaseServices.create(config, resources)


def _storage(config, resources):
    from utils.storage.factory import create_storage_client
    return create_storage_client(config, execution=resources.storage_io)


def _output(config, databases):
    from aiokafka import AIOKafkaProducer
    from utils.buffers.buffer import FeatureBuffer

    return FeatureBuffer(
        face_db=databases.face, picture_db=databases.picture,
        sbs_db=databases.sbs, tbps_db=databases.tbps,
        backlog_high_watermark=config.OUTPUT_BACKLOG_HIGH_WATERMARK,
        backlog_low_watermark=config.OUTPUT_BACKLOG_LOW_WATERMARK,
        producer_factory=lambda: AIOKafkaProducer(**config.kafka_client_config),
        topics={
            "face_detection": config.KAFKA_FACE_DETECTION_RECORD_TOPIC,
            "face_recognition": config.KAFKA_FACE_RECOGNITION_RECORD_TOPIC,
            "person_detection": config.KAFKA_PERSON_DETECTION_RECORD_TOPIC,
            "person_recognition": config.KAFKA_PERSON_RECOGNITION_RECORD_TOPIC,
            "vector_record": config.KAFKA_VECTOR_RECORD_TOPIC,
        },
    )


def _processor(runtime):
    from loguru import logger
    from core.identity import FaceIdentityService
    from core.keyframe_processor import KeyframeProcessor
    from utils.reid.person_identity import PersonIdentityService
    from utils.reid.person_search import PersonReidSearchService

    config, output, databases = runtime.config, runtime.output, runtime.databases
    face_identity = FaceIdentityService(
        face_gallery=databases.face, pending_faces=output.face_pending_view,
        face_features=output.face_features, records=output.records, logger=logger,
        prefetch_enabled=config.FACE_SEARCH_PREFETCH_ENABLED,
    )
    person_search = PersonReidSearchService(person_sbs_embedding_db=databases.sbs, sbs_buffer=output.sbs)
    person_identity = PersonIdentityService(
        identity_lock=asyncio.Lock(), search_persons=person_search.search,
        search_service=person_search, sbs_buffer=output.sbs, logger=logger,
    )
    return KeyframeProcessor(
        triton_client=runtime.triton,
        face_identity_service=face_identity, person_identity_service=person_identity,
        picture_features=output.picture_features, records=output.records,
        sbs_features=output.sbs, person_clip_features=output.tbps,
        image_executor=runtime.resources.image_cpu, frame_budget=runtime.frames,
        face_frame_concurrency=config.FACE_FRAME_CONCURRENCY,
        face_global_concurrency=config.FACE_GLOBAL_CONCURRENCY,
        face_prepare_lookahead=config.FACE_PREPARE_LOOKAHEAD,
        person_frame_concurrency=config.PERSON_FRAME_CONCURRENCY,
        person_global_concurrency=config.PERSON_GLOBAL_CONCURRENCY,
        person_prepare_lookahead=config.PERSON_PREPARE_LOOKAHEAD,
        person_face_concurrency=config.PERSON_FACE_CONCURRENCY or config.PERSON_GLOBAL_CONCURRENCY,
        person_upright_device_sns=config.person_upright_device_sns,
        person_face_reuse_mode=config.person_face_reuse_mode,
    )


@dataclass(frozen=True)
class RuntimeFactories:
    resources: Callable = _resources
    triton: Callable = _triton
    databases: Callable = _databases
    storage: Callable = _storage
    output: Callable = _output
    processor: Callable = _processor


class ApplicationRuntime:
    """唯一资源所有者；关闭等待真实线程与输出结束，启动失败逆序清理。"""

    def __init__(self, config, *, factories: RuntimeFactories | None = None, storage=None):
        self.config = config
        self.factories = factories or RuntimeFactories()
        self.resources = self.triton = self.databases = self.output = self.processor = None
        self.storage = storage
        self.admission = None
        self._started = False
        self._close_task = None
        self._start_task = None
        capacity = config.ALOG_MAX_INFLIGHT_MESSAGES or config.ALOG_WORKER_COUNT
        self.frames = FrameBudget(
            config.ALOG_MAX_INFLIGHT_FRAMES or capacity,
            memory_bytes=config.FRAME_MEMORY_BUDGET_BYTES,
            reservation_bytes=config.FRAME_RESERVATION_BYTES,
        )

    async def start(self):
        if self._close_task is not None:
            raise RuntimeError("application runtime is closing or closed")
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._start_resources(), name="alog-runtime-start")
        try:
            return await asyncio.shield(self._start_task)
        except BaseException:
            try:
                await self.close()
            except BaseException:
                pass
            raise

    async def _start_resources(self):
        self.resources = self.factories.resources(self.config)
        self.triton = self.factories.triton(self.config, self.resources)
        await self.triton.initialize()
        # 构造函数可能建立连接/检查schema，保留真实结果直到可以登记所有权。
        creation = asyncio.create_task(self.resources.vector_write.run(
            self.factories.databases, self.config, self.resources,
        ))
        try:
            self.databases = await asyncio.shield(creation)
        except asyncio.CancelledError:
            while not creation.done():
                try:
                    await asyncio.shield(creation)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not creation.cancelled() and creation.exception() is None:
                self.databases = creation.result()
            raise
        if self.storage is None:
            self.storage = self.factories.storage(self.config, self.resources)
        if self.config.STORAGE_OUTPUT_PREFIX:
            self.storage = OutputNamespacedStorage(self.storage, self.config.STORAGE_OUTPUT_PREFIX)
        self.output = self.factories.output(self.config, self.databases)
        await self.output.start()
        self.admission = AdmissionController(self.output)
        self.processor = self.factories.processor(self)
        self._started = True
        return self

    async def process_message(self, message):
        if not self._started:
            raise RuntimeError("application runtime is not started")
        await self.processor.process_message(message, self.storage)

    def metric_providers(self) -> dict[str, Callable]:
        return {
            "resources": self.resources.queue_snapshot,
            "frames": self.frames.queue_snapshot,
            "triton": self.triton.queue_snapshot,
            "output": self.output.queue_snapshot,
        }

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close(), name="alog-runtime-close")
        await asyncio.shield(self._close_task)

    async def _finish_close(self):
        if self._start_task is not None and not self._start_task.done():
            self._start_task.cancel()
            await asyncio.gather(self._start_task, return_exceptions=True)
        failures = []
        for resource, method in (
            (self.output, "stop"), (self.triton, "close"), (self.resources, "close"),
        ):
            if resource is not None:
                try:
                    await getattr(resource, method)()
                except BaseException as error:
                    failures.append(error)
        # 线程池已排空，数据库连接不再被取消后的后台线程使用。
        if self.databases is not None:
            try:
                self.databases.close()
            except BaseException as error:
                failures.append(error)
        self._started = False
        if failures:
            raise RuntimeError("application resource cleanup failed") from failures[0]

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            await self.close()
        except BaseException:
            if exc is None:
                raise
