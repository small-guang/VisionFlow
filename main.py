"""ALog入口：配置日志、装配应用、运行Kafka服务。导入不建立外部连接。"""
from __future__ import annotations

import asyncio
import signal
import sys

from loguru import logger

from core.config import settings
from core.keyframe_processor import KeyframeProcessor
from core.message_scheduler import MessageScheduler
from core.performance import runtime_performance
from core.queue_metrics import pipeline_metrics
from utils.buffers.buffer import feature_buffer
from utils.storage.base import ObjectStorageClient
from utils.storage.factory import create_storage_client

# 仅供旧离线工具显式注入；生产入口由ApplicationRuntime持有依赖。
KEYFRAME_PROCESSOR = None
TRITON_CLIENT = None
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"


def configure_logging():
    logger.remove()
    options = dict(level=settings.LOG_LEVEL.upper(), format=LOG_FORMAT, enqueue=True,
                   filter=lambda record: "pipeline_stats_session" not in record["extra"])
    logger.add(sys.stderr, **options)
    logger.add("logs/ALog.log", rotation="100 MB", retention="7 days", **options)


async def init_triton_client():
    global TRITON_CLIENT
    if TRITON_CLIENT is None:
        from core.infer import UnifiedTritonClient
        TRITON_CLIENT = UnifiedTritonClient()
    await TRITON_CLIENT.initialize()


async def check_db_connection():
    from db.pgsql.cam_device import SessionLocal
    from sqlalchemy.future import select
    async with SessionLocal() as session:
        await session.execute(select(1))


async def worker_process_message(msg, oss_client: ObjectStorageClient):
    if KEYFRAME_PROCESSOR is None:
        raise RuntimeError("Use ApplicationRuntime, or explicitly inject KEYFRAME_PROCESSOR")
    await KEYFRAME_PROCESSOR.process_message(msg, oss_client)


async def processing_worker(
    name: str,
    queue: asyncio.Queue,
    oss_client: ObjectStorageClient,
    stop_event: asyncio.Event,
    scheduler: MessageScheduler | None = None,
):
    logger.info(f"Worker {name} started.")
    while not stop_event.is_set() or not queue.empty():
        try:
            with pipeline_metrics.waiting("output_backpressure"):
                await feature_buffer.wait_for_capacity()
            msg = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logger.info(f"Worker {name} was cancelled.")
            break
        transferred = False
        try:
            if scheduler is None:
                with pipeline_metrics.activity("message_processing"):
                    await worker_process_message(msg, oss_client)
            else:
                await scheduler.submit(msg, on_done=queue.task_done)
                transferred = True
        except asyncio.CancelledError:
            logger.info(f"Worker {name} was cancelled.")
            raise
        except Exception as e:
            logger.error(f"Worker {name} encountered an unhandled exception: {e}")
        finally:
            if not transferred:
                queue.task_done()

    logger.info(f"Worker {name} stopped.")


async def consume_and_dispatch(consumer, queue, stop_event):
    from core.worker_service import consume_messages
    from utils.keyframe.detection_helpers import parse_keyframe_context
    return await consume_messages(consumer, queue, stop_event,
                                  parse_context=parse_keyframe_context, logger=logger)


async def run_with_triton(client, runner):
    await client.initialize()
    runner_error = None
    try:
        return await runner()
    except BaseException as error:
        runner_error = error
        raise
    finally:
        try:
            await client.close()
        except BaseException as close_error:
            if runner_error is None:
                raise
            logger.error(
                "Triton close failed while propagating service failure: {}",
                type(close_error).__name__,
            )


async def run_worker_service(runtime=None, stop_event=None):
    from core.multiprocess.runtime import create_runtime
    if runtime is None:
        # 旧调用入口仍通过所选存储工厂，正式main传入已装配runtime。
        storage = None if getattr(settings, "VISUAL_WORKER_PROCESSES", 0) else create_storage_client()
        async with create_runtime(settings, storage=storage) as owned:
            return await run_worker_service(owned, stop_event)
    from aiokafka import AIOKafkaConsumer
    from core.worker_service import WorkerService
    from utils.keyframe.detection_helpers import parse_keyframe_context
    stop_event = stop_event if stop_event is not None else asyncio.Event()
    consumer = AIOKafkaConsumer(
        *runtime.config.KAFKA_INPUT_TOPICS,
        group_id=runtime.config.KAFKA_GROUP_ID,
        auto_offset_reset=runtime.config.KAFKA_AUTO_OFFSET_RESET,
        enable_auto_commit=False, max_poll_records=50, fetch_max_wait_ms=200,
        **runtime.config.kafka_client_config,
    )
    service = WorkerService(
        consumer=consumer, handler=runtime.process_message, admission=runtime.admission,
        worker_count=runtime.config.ALOG_WORKER_COUNT,
        message_capacity=runtime.config.ALOG_MAX_INFLIGHT_MESSAGES or runtime.config.ALOG_WORKER_COUNT,
        parse_context=parse_keyframe_context, logger=logger,
    )
    if not callable(getattr(runtime, "wait_failure", None)):
        return await service.run(stop_event)
    running = asyncio.create_task(service.run(stop_event), name="alog-service")
    failure = asyncio.create_task(runtime.wait_failure(), name="alog-process-health")
    try:
        done, _ = await asyncio.wait([running, failure], return_when=asyncio.FIRST_COMPLETED)
        if failure in done:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            await failure
        return await running
    finally:
        running.cancel()
        failure.cancel()
        await asyncio.gather(running, failure, return_exceptions=True)


async def main():
    import cv2
    from core.multiprocess.runtime import create_runtime
    cv2.setNumThreads(1)
    configure_logging()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)
    providers = {}
    runtime_performance.start()
    try:
        async with create_runtime(settings) as runtime:
            providers = runtime.metric_providers()
            for name, provider in providers.items():
                pipeline_metrics.register(name, provider)
            if settings.PIPELINE_STATS_ENABLED:
                pipeline_metrics.start(path=settings.PIPELINE_STATS_PATH,
                                       interval_seconds=settings.PIPELINE_STATS_INTERVAL_SECONDS)
            await run_worker_service(runtime, stop_event)
    finally:
        try:
            await runtime_performance.stop()
        finally:
            try:
                await pipeline_metrics.stop()
            finally:
                for name in providers:
                    pipeline_metrics.unregister(name)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(sig)
                await logger.complete()


if __name__ == "__main__":
    import uvloop
    uvloop.install()
    asyncio.run(main())
