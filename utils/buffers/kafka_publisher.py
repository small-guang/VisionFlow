"""各输出 Topic 的有序发布端口；只依赖 producer 协议与写队列。"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from typing import Any

from loguru import logger
from .background_writer import WriteQueue


class KafkaPublisher:
    """每个 Topic 一个 flush owner，ACK 后完成计数，失败只重排未 ACK 部分。"""

    def __init__(self, *, topics: dict[str, str], producer_factory: Callable[[], Any], max_batch_size: int):
        self.topics = dict(topics)
        self.queues = {name: WriteQueue(unit="kafka_record") for name in topics}
        self.producer_factory = producer_factory
        self.producer = None
        self.max_batch_size = max_batch_size
        self.request_flush: Callable[[str], None] | None = None

    async def start(self) -> None:
        if self.producer is None:
            producer = self.producer_factory()
            # start 可能在建立部分连接后失败或被取消，回收者必须仍能找到它。
            self.producer = producer
            await producer.start()

    async def stop(self) -> None:
        if self.producer is not None:
            await self.producer.stop()
            self.producer = None

    async def add(self, name: str, record: dict[str, Any]) -> None:
        self.queues[name].put_nowait(record)
        await self.flush_if_full(name)

    async def flush_if_full(self, name: str) -> None:
        if self.queues[name].qsize() >= self.max_batch_size:
            if self.request_flush is not None:
                self.request_flush(name)
            else:
                await self.flush(name)

    async def flush(self, name: str) -> None:
        await self.flush_queue(self.queues[name], self.topics[name], name)

    async def flush_queue(
        self,
        record_buffer: WriteQueue,
        topic: str,
        record_name: str,
    ):
        """逐条发送业务记录 Kafka 消息，失败时只重排未发送记录。"""
        async with record_buffer.flush_lock:
            if record_buffer.empty():
                return

            batch = []
            while not record_buffer.empty() and len(batch) < self.max_batch_size:
                batch.append(await record_buffer.get())

            if not batch:
                return

            sent_count = 0
            try:
                kafka_producer = self.producer
                if kafka_producer is None:
                    raise RuntimeError(f"Kafka producer is not initialized for {record_name}")

                for record in batch:
                    await kafka_producer.send_and_wait(
                        topic,
                        json.dumps(record, ensure_ascii=False).encode('utf-8')
                    )
                    sent_count += 1
                    record_buffer.complete()

                logger.info(f"成功发送 {sent_count} 条{record_name}到 Kafka topic {topic}")
            except asyncio.CancelledError:
                record_buffer.requeue(batch[sent_count:])
                raise
            except Exception as e:
                logger.error(f"发送{record_name}到 Kafka 失败: {str(e)}")
                record_buffer.requeue(batch[sent_count:])

    async def add_face_detection_record(self, record: dict[str, Any]) -> None:
        await self.add("face_detection", record)

    async def add_face_recognition_record(self, record: dict[str, Any]) -> None:
        await self.add("face_recognition", record)

    async def add_person_detection_record(self, record: dict[str, Any]) -> None:
        await self.add("person_detection", record)

    async def add_person_recognition_record(self, record: dict[str, Any]) -> None:
        await self.add("person_recognition", record)

    async def add_vector_record(self, record: dict[str, Any]) -> None:
        await self.add("vector_record", record)
