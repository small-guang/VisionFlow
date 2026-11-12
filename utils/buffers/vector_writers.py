"""整图与人脸向量写入端口；队列、pending、回执各有独立所有者。"""
from __future__ import annotations

import asyncio
import math
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger
from core.algorithm_config import algorithm_config
from .background_writer import WriteQueue
from .pending import FacePendingIndex
from .persistence_ticket import PersistenceTicket
from .vector_records import VectorRecordEmitter
from .kafka_publisher import KafkaPublisher


class FaceVectorWriter:
    def __init__(self, *, vector_db: Any, pending: FacePendingIndex,
                 emitter: VectorRecordEmitter, records: KafkaPublisher, max_batch_size: int):
        self.vector_db = vector_db
        self.pending = pending
        self.emitter = emitter
        self.records = records
        self.max_batch_size = max_batch_size
        self.queue = WriteQueue(unit="vector_item")
        self.request_flush = None

    async def _accept_face_recognition(self, record: dict[str, Any]) -> None:
        # 接纳与 ticket.released 之间无可取消 I/O；flush 留在确认之后。
        self.records.queues["face_recognition"].put_nowait(record)

    async def _flush_if_full(self) -> None:
        if self.queue.qsize() >= self.max_batch_size:
            if self.request_flush is not None:
                self.request_flush()
            else:
                await self.flush()

    async def flush(self) -> None:
        async with self.queue.flush_lock:
            await self._flush_face_batch()

    async def add_face_feature(
        self,
        face_id: str,
        embedding: np.ndarray,
        device_id: str,
        keyframe_id: str,
        picture_vector_id: int,
        keyframe_url: str,
        face_rect: Dict[str, int],
        face_key_points: List[Dict[str, float]],
        face_crop_path: str,
        timestamp: int,
        ai_id: str,
        quality_score: float,
        is_pano_keyframe: bool = False,
        new_face_recognition_record: Optional[Dict[str, Any]] = None,
    ):
        """添加人脸特征到缓冲区"""
        quality_score = float(quality_score)
        if not math.isfinite(quality_score) or not algorithm_config.face.quality.enroll_min_score <= quality_score <= 10.0:
            raise ValueError(f"quality_score must be in [{algorithm_config.face.quality.enroll_min_score}, 10]")
        item = {
            'sampleId': str(uuid.uuid4()),
            'faceId': face_id,
            'embedding': embedding,
            'deviceId': device_id,
            'keyframeId': keyframe_id,
            'pictureVectorId': picture_vector_id,
            'keyframeUrl': keyframe_url,
            'faceRect': face_rect,
            'faceKeyPoints': face_key_points,
            'faceCropPath': face_crop_path,
            'timestamp': timestamp,
            'aiId': ai_id,
            'qualityScore': quality_score,
            'isPanoKeyframe': bool(is_pano_keyframe),
        }
        if new_face_recognition_record is not None:
            item['__newFaceRecognitionRecord'] = new_face_recognition_record
            item['__persistenceTicket'] = PersistenceTicket(
                record=new_face_recognition_record, on_ready=self._accept_face_recognition,
            )
        await self.pending.add(item)
        await self.queue.put(item)

        await self._flush_if_full()

    async def _flush_face_batch(self):
        if self.queue.empty():
            return

        batch = []
        while not self.queue.empty() and len(batch) < self.max_batch_size:
            batch.append(await self.queue.get())

        skipped_count = sum(1 for item in batch if not item.get('faceCropPath'))
        if skipped_count:
            logger.warning(f"跳过 {skipped_count} 条缺少 faceCropPath 的人脸特征，未写入 Milvus")
        if skipped_count:
            try:
                await self.pending.remove(
                    [item for item in batch if not item.get('faceCropPath')],
                )
            except asyncio.CancelledError:
                self.queue.requeue(batch)
                raise
            self.queue.complete(skipped_count)
        batch = [item for item in batch if item.get('faceCropPath')]
        if not batch:
            return

        if batch:
            try:
                face_ids = [item['faceId'] for item in batch]
                embeddings = [item['embedding'] for item in batch]
                device_ids = [item['deviceId'] for item in batch]
                track_ids = [0 for _ in batch]
                keyframe_ids = [item['pictureVectorId'] for item in batch]
                face_rects = [item['faceRect'] for item in batch]
                face_key_points = [item['faceKeyPoints'] for item in batch]
                face_crop_paths = [item['faceCropPath'] for item in batch]
                timestamps = [item['timestamp'] for item in batch]


                insert_result = await self.vector_db.ainsert_embeddings(
                    faceId=face_ids,
                    embedding=embeddings,
                    deviceId=device_ids,
                    trackId=track_ids,
                    keyframeId=keyframe_ids,
                    faceRect=face_rects,
                    faceKeyPoints=face_key_points,
                    faceCropPath=face_crop_paths,
                    timestamp=timestamps,
                    qualityScore=[item['qualityScore'] for item in batch],
                    sampleId=[item['sampleId'] for item in batch],
                )
                outcomes = insert_result.get("outcomes", [])
                if len(outcomes) != len(batch) or any(
                    not outcome.get("persisted") for outcome in outcomes
                ):
                    raise RuntimeError("Milvus did not confirm all face gallery outcomes")
                written = [
                    (item, outcome["primary_key"])
                    for item, outcome in zip(batch, outcomes)
                    if outcome.get("primary_key") is not None
                ]
                await self.emitter._queue_vector_records(
                    [item for item, _ in written],
                    [primary_key for _, primary_key in written],
                    "face",
                )
                for item in batch:
                    ticket = item.get("__persistenceTicket")
                    if ticket is not None:
                        await ticket.mark_persisted()
                        item["__newFaceRecordQueued"] = ticket.released
                    await self.records.flush_if_full("face_recognition")
                await self.pending.remove(batch)
                self.queue.complete(len(batch))
                logger.info(
                    f"人脸图库更新完成：保留/更新 {len(written)} 条，质量策略跳过 {len(batch) - len(written)} 条"
                )
            except asyncio.CancelledError:
                self.queue.requeue(batch)
                raise
            except Exception as e:
                logger.error(f"写入人脸特征失败: {str(e)}")
                self.queue.requeue(batch)


class PictureVectorWriter:
    def __init__(self, *, vector_db: Any, emitter: VectorRecordEmitter, max_batch_size: int):
        self.vector_db = vector_db
        self.emitter = emitter
        self.max_batch_size = max_batch_size
        self.queue = WriteQueue(unit="vector_item")
        self.request_flush = None

    async def _flush_if_full(self) -> None:
        if self.queue.qsize() >= self.max_batch_size:
            if self.request_flush is not None:
                self.request_flush()
            else:
                await self.flush()

    async def add_picture_feature(
        self,
        keyframe_id: str,
        picture_vector_id: int,
        embedding: np.ndarray,
        device_id: str,
        image_path: str,
        timestamp: int,
        is_pano_keyframe: bool = False,
    ):
        """添加关键帧特征到缓冲区"""
        await self.queue.put({
            'keyframeId': keyframe_id,
            'pictureVectorId': picture_vector_id,
            'embedding': embedding,
            'deviceId': device_id,
            'imagePath': image_path,
            'timestamp': timestamp,
            'keyframeUrl': image_path,
            'aiId': "",
            'isPanoKeyframe': bool(is_pano_keyframe),
        })

        await self._flush_if_full()

    async def flush(self):
        """批量写入关键帧特征"""
        async with self.queue.flush_lock:
            if self.queue.empty():
                return

            batch = []
            while not self.queue.empty() and len(batch) < self.max_batch_size:
                batch.append(await self.queue.get())

            if batch:
                try:
                    keyframe_ids = [item['pictureVectorId'] for item in batch]
                    embeddings = [item['embedding'] for item in batch]
                    device_ids = [item['deviceId'] for item in batch]
                    image_paths = [item['imagePath'] for item in batch]
                    timestamps = [item['timestamp'] for item in batch]

                    await self.vector_db.ainsert_embeddings(
                        keyframeId=keyframe_ids,
                        embedding=embeddings,
                        deviceId=device_ids,
                        imagePath=image_paths,
                        timestamp=timestamps
                    )
                    logger.info(f"成功写入 {len(batch)} 条关键帧特征")
                    await self.emitter._queue_vector_records(batch, keyframe_ids, "picture")
                    self.queue.complete(len(batch))
                except asyncio.CancelledError:
                    self.queue.requeue(batch)
                    raise
                except Exception as e:
                    logger.error(f"写入关键帧特征失败: {str(e)}")
                    self.queue.requeue(batch)

