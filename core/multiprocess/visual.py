"""视觉进程装配：图像与推理留在本进程，身份和输出通过窄端口提交。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from core.application import OutputNamespacedStorage, _storage, _triton
from .assembly import visual_resources
from .frames import RemoteFrameBudget


def _configure_process(config):
    import cv2
    from loguru import logger
    from utils.identifiers.id_generator import id_generator

    # spawn 创建全新解释器；在任何图像工作/ID生成前设置本进程的唯一编号。
    index, count = config.MP_WORKER_INDEX, config.MP_WORKER_COUNT
    if (type(index) is not int or type(count) is not int
            or not 0 <= index < count <= 4):
        raise ValueError("invalid visual worker identity")
    id_generator.worker_id = index + 1
    cv2.setNumThreads(1)
    logger.remove()
    logger.add(sys.stderr, level="WARNING", enqueue=False)


def _processor(runtime):
    from core.keyframe_processor import KeyframeProcessor
    from .ports import make_visual_ports

    config = runtime.config
    runtime.identity_ports = make_visual_ports(runtime.peer)
    return KeyframeProcessor(
        triton_client=runtime.triton,
        **runtime.identity_ports,
        image_executor=runtime.resources.image_cpu,
        frame_budget=runtime.frames,
        face_frame_concurrency=config.FACE_FRAME_CONCURRENCY,
        face_global_concurrency=config.FACE_GLOBAL_CONCURRENCY,
        face_prepare_lookahead=config.FACE_PREPARE_LOOKAHEAD,
        person_frame_concurrency=config.PERSON_FRAME_CONCURRENCY,
        person_global_concurrency=config.PERSON_GLOBAL_CONCURRENCY,
        person_prepare_lookahead=config.PERSON_PREPARE_LOOKAHEAD,
        person_face_concurrency=config.PERSON_FACE_CONCURRENCY,
        person_upright_device_sns=config.person_upright_device_sns,
        person_face_reuse_mode=config.person_face_reuse_mode,
        strict_errors=True,
        image_completed_callback=runtime.image_completed,
    )


@dataclass(frozen=True)
class VisualRuntimeFactories:
    configure_process: Callable = _configure_process
    resources: Callable = visual_resources
    triton: Callable = _triton
    storage: Callable = _storage
    processor: Callable = _processor


class VisualRuntime:
    """无数据库、producer或身份状态的视觉生命周期。"""

    def __init__(
        self, config, peer, *, factories: VisualRuntimeFactories | None = None,
        observability: bool = True,
    ):
        self.config = config
        self.peer = peer
        self.factories = factories or VisualRuntimeFactories()
        self.frames = RemoteFrameBudget(peer)
        self.resources = self.triton = self.storage = self.processor = None
        self.identity_ports = None
        self._started = False
        self._start_task = self._close_task = None
        self._active_tasks: set[asyncio.Task] = set()
        self._observability = observability
        self._metrics_started = False
        self._performance_sink = None
        self._previous_performance_logger = None

    async def start(self):
        if self._close_task is not None:
            raise RuntimeError("visual runtime is closing or closed")
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._start_resources(), name="alog-visual-start")
        try:
            return await asyncio.shield(self._start_task)
        except BaseException:
            try:
                await self.close()
            except BaseException:
                pass
            raise

    async def _start_resources(self):
        self.factories.configure_process(self.config)
        self.resources = self.factories.resources(self.config)
        self.triton = self.factories.triton(self.config, self.resources)
        await self.triton.initialize()
        self.storage = self.factories.storage(self.config, self.resources)
        if self.config.STORAGE_OUTPUT_PREFIX:
            self.storage = OutputNamespacedStorage(self.storage, self.config.STORAGE_OUTPUT_PREFIX)
        self.processor = self.factories.processor(self)
        if self._observability:
            self._start_metrics()
        self._started = True
        return self

    def _start_metrics(self):
        from loguru import logger
        from core.performance import runtime_performance
        from core.queue_metrics import pipeline_metrics

        worker = self.config.MP_WORKER_INDEX + 1
        target = Path(self.config.PIPELINE_STATS_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._previous_performance_logger = runtime_performance._logger
        self._performance_sink = logger.add(
            str(target.with_name(f"visual-{worker}.log")), level="INFO", format="{message}",
            enqueue=True, rotation="100 MB", retention="7 days", encoding="utf-8",
            filter=lambda record: record["extra"].get("visual_performance_worker") == worker,
        )
        runtime_performance._logger = logger.bind(visual_performance_worker=worker)
        self._metrics_started = True
        for name, provider in self.metric_providers().items():
            pipeline_metrics.register(name, provider)
        if self.config.PIPELINE_STATS_ENABLED:
            pipeline_metrics.start(
                path=str(target.with_name(f"{target.stem}.visual-{worker}{target.suffix}")),
                interval_seconds=self.config.PIPELINE_STATS_INTERVAL_SECONDS,
            )
        runtime_performance.start(interval_seconds=self.config.PIPELINE_STATS_INTERVAL_SECONDS)

    async def process_message(self, message):
        from .transport import current_work_id

        if not self._started or self._close_task is not None:
            raise RuntimeError("visual runtime is not started")
        work_id = current_work_id.get()
        task = asyncio.current_task()
        self._active_tasks.add(task)
        try:
            if isinstance(message, Mapping):
                value, topic, offset = message["value"], message.get("topic", ""), message.get("offset", "")
            else:
                value, topic, offset = message.value, getattr(message, "topic", ""), getattr(message, "offset", "")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            data = value if isinstance(value, dict) else json.loads(value)
            if not isinstance(data, dict):
                raise ValueError("Kafka message payload must be an object")
            await self.processor.process_payload(
                data, oss_client=self.storage, topic=topic, offset=offset,
            )
            from core.performance import runtime_performance
            runtime_performance.message_completed(topic)
            return {"topic": topic, "message_completed": 1}
        finally:
            self._active_tasks.discard(task)
            if self.identity_ports is not None:
                self.identity_ports.close_work(work_id)

    async def image_completed(self, topic: str, uid: str, is_pano: bool):
        await self.peer.call("frame.complete", {"topic": topic, "uid": uid, "is_pano": is_pano})

    def metric_providers(self) -> dict[str, Callable]:
        return {
            "resources": self.resources.queue_snapshot,
            "frames": self.frames.queue_snapshot,
            "triton": self.triton.queue_snapshot,
        }

    def metrics(self) -> dict[str, dict]:
        """只读队列快照，不消费周期性能汇总的增量计数。"""
        if self.resources is None or self.triton is None:
            return {}
        return {name: provider() for name, provider in self.metric_providers().items()}

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close(), name="alog-visual-close")
        await asyncio.shield(self._close_task)

    async def _finish_close(self):
        self._started = False
        if self._start_task is not None and not self._start_task.done():
            self._start_task.cancel()
            await asyncio.gather(self._start_task, return_exceptions=True)
        # supervisor 先停止分发；尚在执行的帧必须离开本地资源后才能关闭池。
        if self._active_tasks:
            await asyncio.gather(*tuple(self._active_tasks), return_exceptions=True)
        failures = []
        for resource in (self.triton, self.resources):
            if resource is not None:
                try:
                    await resource.close()
                except BaseException as error:
                    failures.append(error)
        if self._metrics_started:
            try:
                await self._stop_metrics()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError("visual resource cleanup failed") from failures[0]

    async def _stop_metrics(self):
        from loguru import logger
        from core.performance import runtime_performance
        from core.queue_metrics import pipeline_metrics

        try:
            try:
                await runtime_performance.stop()
            finally:
                await pipeline_metrics.stop()
            await logger.complete()
        finally:
            for name in self.metric_providers():
                pipeline_metrics.unregister(name)
            runtime_performance._logger = self._previous_performance_logger
            if self._performance_sink is not None:
                logger.remove(self._performance_sink)
                self._performance_sink = None
            self._metrics_started = False


def visual_runtime_factory(config, peer):
    return VisualRuntime(config, peer)
