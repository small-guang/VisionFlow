from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Dict

import numpy as np
from loguru import logger

from core.face_observation import FrameFaceObservation
from core.face_keyframe_processor import FaceKeyframeProcessor
from core.identity import FaceIdentityService
from core.identity.ports import PictureFeatureSink
from core.config import settings
from utils.keyframe.keyframe_flow_helpers import normalize_numpy_vector
from core.person_keyframe_processor import PersonKeyframeProcessor
from core.performance import runtime_performance
from utils.identifiers.id_generator import id_generator
from utils.keyframe.detection_helpers import (
    KeyframeContext,
    parse_keyframe_context,
    parse_pano_keyframe_context,
)
from utils.reid.person_identity import PersonIdentityService
from utils.reid.person_search import PersonReidSearchService
from utils.storage.image_util import get_image_by_oss_name
from utils.storage.base import ObjectStorageClient


class KeyframeProcessor:
    def __init__(
        self,
        *,
        triton_client: Any,
        feature_buffer: Any = None,
        face_embedding_db: Any = None,
        person_sbs_embedding_db: Any = None,
        keyframe_id_generator: Any = id_generator,
        image_loader: Callable[[str, ObjectStorageClient], Any] = get_image_by_oss_name,
        person_upright_device_sns: frozenset[str] | None = None,
        person_face_reuse_mode: str | None = None,
        face_identity_service: FaceIdentityService | None = None,
        person_identity_service: PersonIdentityService | None = None,
        picture_features: PictureFeatureSink | None = None,
        records: Any = None,
        sbs_features: Any = None,
        person_clip_features: Any = None,
        image_executor: Any,
        person_face_concurrency: int | None = None,
        face_frame_concurrency: int | None = None,
        face_global_concurrency: int | None = None,
        face_prepare_lookahead: int | None = None,
        person_frame_concurrency: int | None = None,
        person_global_concurrency: int | None = None,
        person_prepare_lookahead: int | None = None,
        frame_budget: Any = None,
        face_processor: Any = None,
        person_processor: Any = None,
        strict_errors: bool = False,
        image_completed_callback: Any = None,
    ):
        self.strict_errors = strict_errors
        self.image_completed_callback = image_completed_callback
        self.triton_client = triton_client
        self.feature_buffer = feature_buffer  # 旧工具构造的兼容入口。
        self.face_embedding_db = face_embedding_db
        self.person_sbs_embedding_db = person_sbs_embedding_db
        self.keyframe_id_generator = keyframe_id_generator
        self.image_executor = image_executor
        self.frame_budget = frame_budget
        self.image_loader = (
            partial(image_loader, image_executor=image_executor)
            if image_loader is get_image_by_oss_name
            else image_loader
        )
        self.picture_features = picture_features if picture_features is not None else feature_buffer
        records = records if records is not None else getattr(feature_buffer, "records", feature_buffer)
        sbs_features = sbs_features if sbs_features is not None else getattr(feature_buffer, "sbs", None)
        person_clip_features = (
            person_clip_features if person_clip_features is not None
            else getattr(feature_buffer, "tbps", None)
        )
        if face_identity_service is None and (face_processor is None or person_processor is None):
            face_features = getattr(feature_buffer, "face_features", None)
            if not callable(getattr(face_features, "add_face_feature", None)):
                face_features = feature_buffer
            face_identity_service = FaceIdentityService(
                face_gallery=face_embedding_db,
                pending_faces=getattr(feature_buffer, "face_pending_view", feature_buffer),
                face_features=face_features,
                records=records,
                logger=logger,
                prefetch_enabled=settings.FACE_SEARCH_PREFETCH_ENABLED,
            )
        self.face_identity_service = face_identity_service
        # 兼容旧工具的诊断属性；共享锁与搜索实际由身份服务拥有。
        self._face_identity_lock = getattr(face_identity_service, "identity_lock", None)
        self._face_search_service = getattr(face_identity_service, "search_service", None)
        if person_identity_service is None and person_processor is None:
            person_search = PersonReidSearchService(
                person_sbs_embedding_db=person_sbs_embedding_db, sbs_buffer=sbs_features,
            )
            person_identity_service = PersonIdentityService(
                identity_lock=asyncio.Lock(), search_persons=person_search.search,
                search_service=person_search, sbs_buffer=sbs_features, logger=logger,
            )
        self._person_identity_service = person_identity_service
        self._person_identity_lock = getattr(person_identity_service, "identity_lock", None)
        self._person_reid_search = getattr(person_identity_service, "search_service", None)
        self.face_processor = face_processor if face_processor is not None else FaceKeyframeProcessor(
            triton_client=triton_client,
            identity_service=face_identity_service,
            records=records,
            image_executor=image_executor,
            face_frame_concurrency=face_frame_concurrency,
            face_global_concurrency=face_global_concurrency,
            prepare_lookahead=(
                settings.FACE_PREPARE_LOOKAHEAD
                if face_prepare_lookahead is None else face_prepare_lookahead
            ),
            logger=logger,
        )
        self.person_processor = person_processor if person_processor is not None else PersonKeyframeProcessor(
            triton_client=triton_client,
            records=records,
            sbs_features=sbs_features,
            clip_features=person_clip_features,
            identity_service=person_identity_service,
            search_faces_with_pending_locked=face_identity_service.search,
            logger=logger,
            upright_device_sns=person_upright_device_sns,
            person_face_reuse_mode=person_face_reuse_mode,
            person_face_concurrency=person_face_concurrency,
            person_frame_concurrency=person_frame_concurrency,
            person_global_concurrency=person_global_concurrency,
            prepare_lookahead=(
                settings.PERSON_PREPARE_LOOKAHEAD
                if person_prepare_lookahead is None else person_prepare_lookahead
            ),
            image_executor=image_executor,
        )

    def _face_service(self) -> FaceIdentityService:
        service = getattr(self, "face_identity_service", None)
        if service is None:
            # 兼容只注入旧字段的独立工具；正常运行永远使用应用装配的服务。
            service = FaceIdentityService(
                face_gallery=self.face_embedding_db, pending_faces=self.feature_buffer,
                face_features=self.feature_buffer, records=self.feature_buffer,
                identity_lock=self._face_identity_lock,
                search_service=getattr(self, "_face_search_service", None), logger=logger,
            )
            self.face_identity_service = service
        return service

    async def _search_faces_with_pending(self, embedding, threshold, device_ids):
        return await self._face_service().search_faces(
            embedding, threshold=threshold, device_ids=device_ids,
        )

    async def _search_faces_with_pending_locked(self, embedding, threshold, device_ids):
        return await self._face_service().search(embedding, threshold, device_ids)

    async def process_message(self, msg: Any, oss_client: ObjectStorageClient):
        topic = getattr(msg, "topic", "")
        offset = getattr(msg, "offset", "")
        try:
            value = msg.value
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            data = json.loads(value)
            await self.process_payload(
                data,
                oss_client=oss_client,
                topic=topic,
                offset=offset,
            )
            runtime_performance.message_completed(topic)
            logger.debug(
                f"Successfully processed message: Topic={topic}, Offset={offset}, "
                f"uId={data.get('uId', '')}"
            )
        except Exception as e:
            logger.exception(
                f"Error processing message in worker. Topic={topic}, Offset={offset}. Error: {e}"
            )

    async def process_payload(
        self,
        data: Dict[str, Any],
        *,
        oss_client: ObjectStorageClient,
        topic: str = "",
        offset: Any = "",
    ):
        ctx = parse_keyframe_context(data)
        main_processed = await self._process_keyframe_context(
            ctx,
            oss_client=oss_client,
            topic=topic,
            offset=offset,
        )
        if not main_processed:
            return

        pano_object_name = ""
        try:
            pano_ctx = parse_pano_keyframe_context(data, now_ms=ctx.timestamp)
            if pano_ctx is None or pano_ctx.object_name == ctx.object_name:
                return
            pano_object_name = pano_ctx.object_name
            await self._process_keyframe_context(
                pano_ctx,
                oss_client=oss_client,
                topic=topic,
                offset=offset,
            )
        except Exception as exc:
            if getattr(self, "strict_errors", False):
                raise
            logger.exception(
                "Failed to process pano keyframe. "
                f"Topic={topic}, Offset={offset}, "
                f"ObjectName={pano_object_name}, Error={exc}"
            )

    async def _process_keyframe_context(
        self,
        ctx: KeyframeContext,
        *,
        oss_client: ObjectStorageClient,
        topic: str = "",
        offset: Any = "",
    ) -> bool:
        budget = getattr(self, "frame_budget", None)
        if budget is None:
            return await self._run_frame(ctx, oss_client=oss_client, topic=topic, offset=offset)
        async with budget.frame() as lease:
            return await self._run_frame(
                ctx, oss_client=oss_client, topic=topic, offset=offset, lease=lease,
            )

    async def _run_frame(
        self, ctx: KeyframeContext, *, oss_client: ObjectStorageClient,
        topic: str = "", offset: Any = "", lease: Any = None,
    ) -> bool:
        key_frame_object_name = ctx.object_name

        key_frame, key_frame_byte = await self.image_loader(
            key_frame_object_name, oss_client
        )
        if key_frame is None:
            if getattr(self, "strict_errors", False):
                raise RuntimeError("keyframe image download failed")
            logger.error(
                f"Failed to get keyframe from object storage. Topic={topic}, Offset={offset}"
            )
            return False

        if lease is not None:
            lease.observe_image(key_frame, key_frame_byte)
        key_frame_id = ctx.keyframe_id
        picture_vector_id = self.keyframe_id_generator.generate()
        await self.store_picture_feature(
            ctx, key_frame_id, picture_vector_id, key_frame_object_name, key_frame_byte
        )
        frame_faces = await self.process_face_detection(
            ctx,
            key_frame,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            oss_client,
        )
        await self.process_person_detection(
            ctx,
            key_frame,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            oss_client,
            frame_faces=frame_faces,
        )
        # 业务处理完成，缓冲区的 Milvus/Kafka 写入可能仍在等待；独立全景图另计一张。
        runtime_performance.image_completed(
            topic, uid=ctx.keyframe_id, is_pano=ctx.is_pano_keyframe
        )
        callback = getattr(self, "image_completed_callback", None)
        if callback is not None:
            await callback(topic, ctx.keyframe_id, ctx.is_pano_keyframe)
        return True

    async def store_picture_feature(
        self,
        ctx: KeyframeContext,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        key_frame_byte: bytes,
    ):
        picture_features = await self.triton_client.extract_image_features(key_frame_byte)
        raw_features = picture_features.vector
        if self.image_executor is not None:
            normalized_features = await self.image_executor.run(normalize_numpy_vector, raw_features)
        else:
            normalized_features = await asyncio.get_running_loop().run_in_executor(
                None, normalize_numpy_vector, raw_features,
            )
        await self.picture_features.add_picture_feature(
            keyframe_id=key_frame_id,
            picture_vector_id=picture_vector_id,
            embedding=normalized_features,
            device_id=ctx.sn,
            image_path=key_frame_object_name,
            timestamp=ctx.timestamp,
            is_pano_keyframe=ctx.is_pano_keyframe,
        )

    async def process_face_detection(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        oss_client: ObjectStorageClient,
    ) -> Sequence[FrameFaceObservation]:
        return await self.face_processor.process_detection(
            ctx,
            key_frame,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            oss_client,
        )

    async def process_person_detection(
        self,
        ctx: KeyframeContext,
        key_frame: np.ndarray,
        key_frame_id: str,
        picture_vector_id: int,
        key_frame_object_name: str,
        oss_client: ObjectStorageClient,
        *,
        frame_faces: Sequence[FrameFaceObservation] = (),
    ) -> None:
        await self.person_processor.process_detection(
            ctx,
            key_frame,
            key_frame_id,
            picture_vector_id,
            key_frame_object_name,
            oss_client,
            frame_faces=frame_faces,
        )
