"""SBS queue, pending search, and Milvus flush lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, field
import uuid
from typing import Any, Optional

from loguru import logger
import numpy as np


from utils.reid.person_body import (
    gallery_body_value,
)
from .pending import SbsPendingIndex
from .persistence_ticket import PersistenceTicket
from .person_feature_buffer import (
    PersonFeatureBufferBase,
    PersonVectorInsertedCallback,
)
from utils.reid.reid_models import (
    normalize_face_id,
)
from utils.reid.reid_enrollment_state import (
    REID_ENROLLMENT_STATE_FORMAL_PENDING,
    REID_ENROLLMENT_STATE_REJECTED,
)
from utils.reid.reid_state import (
    REID_STATE_ACTIVE,
    REID_STATE_UNASSIGNED,
    require_assigned_reid_state,
)


PersonRecognitionReadyCallback = Callable[
    [dict[str, Any]],
    Awaitable[None],
]


# 保留公开名称，Face 与 SBS 共用首次持久化回执协议。
SbsRecognitionRelease = PersistenceTicket


@dataclass(frozen=True)
class SbsEnrollmentOutcome:
    enrolled: bool
    reason: str
    reid_state: str = REID_STATE_UNASSIGNED
    recognition_release: SbsRecognitionRelease | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def enrollment_state(self) -> str:
        """Return this observation's stage independently of its ReID lifecycle."""

        return (
            REID_ENROLLMENT_STATE_FORMAL_PENDING
            if self.enrolled
            else REID_ENROLLMENT_STATE_REJECTED
        )


class SbsFeatureBuffer(PersonFeatureBufferBase):
    TARGET_TYPE = "person_sbs"

    def __init__(
        self,
        *,
        vector_db: Any,
        max_batch_size: int,
        on_vectors_inserted: PersonVectorInsertedCallback,
        pending_index: SbsPendingIndex | None = None,
    ):
        super().__init__(
            vector_db=vector_db,
            max_batch_size=max_batch_size,
            on_vectors_inserted=on_vectors_inserted,
        )
        self.pending_index = pending_index if pending_index is not None else SbsPendingIndex()
        self.pending = self.pending_index.records
        self._pending_lock = self.pending_index.lock
        self._lifecycle_lock = asyncio.Lock()
        self._canonical_face_by_reid: dict[str, str] = {}
        if not callable(getattr(vector_db, "aapply_prototype_updates", None)):
            raise TypeError(
                "SBS vector database must provide aapply_prototype_updates"
            )

    _as_vector = staticmethod(SbsPendingIndex.as_vector)
    _score_vectors = staticmethod(SbsPendingIndex.score_vectors)
    _device_allowed = staticmethod(SbsPendingIndex.device_allowed)

    async def _add_pending(self, item: dict[str, Any]) -> None:
        await self.pending_index.add(item)

    def pending_removal_revision(self) -> int:
        return self.pending_index.removal_revision()

    async def _remove_pending(self, items: list[dict[str, Any]]) -> None:
        await self.pending_index.remove(items)

    async def _pending_snapshot(self) -> list[dict[str, Any]]:
        return await self.pending_index.snapshot()

    async def _enqueue_formal_feature(
        self,
        *,
        face_id: str,
        re_id: str,
        body_completeness: str,
        embedding: np.ndarray,
        device_id: str,
        keyframe_id: str,
        picture_vector_id: int,
        keyframe_url: str,
        crop_url: str,
        person_rect: dict[str, int],
        timestamp: int,
        ai_id: str,
        quality_score: float,
        is_pano_keyframe: bool = False,
        recognition_release: SbsRecognitionRelease | None = None,
    ) -> None:
        body_completeness = gallery_body_value(body_completeness)
        normalized_reid = str(re_id or "").strip()
        normalized_device = str(device_id or "")
        if quality_score is None:
            raise ValueError("quality_score is required")
        face_id = self._canonical_face_by_reid.get(
            normalized_reid,
            str(face_id or "0"),
        )
        item = {
            "prototypeId": str(uuid.uuid4()),
            "faceId": str(face_id or "0"),
            "reId": normalized_reid,
            "bodyCompleteness": body_completeness,
            "cropUrl": str(crop_url or ""),
            "embedding": embedding,
            "deviceId": normalized_device,
            "keyframeId": keyframe_id,
            "pictureVectorId": picture_vector_id,
            "keyframeUrl": keyframe_url,
            "personRect": person_rect,
            "timestamp": timestamp,
            "aiId": ai_id,
            "qualityScore": float(quality_score),
            "isPanoKeyframe": bool(is_pano_keyframe),
        }
        if recognition_release is not None:
            item["__newReidRecognitionRelease"] = recognition_release
        await self._add_pending(item)
        await self.queue.put(item)
        await self._flush_if_full()

    async def offer_feature(
        self,
        *,
        re_id: str,
        identity_state: str,
        body_completeness: str,
        embedding: np.ndarray,
        device_id: str,
        keyframe_id: str,
        picture_vector_id: int,
        keyframe_url: str,
        person_rect: dict[str, int],
        timestamp: int,
        ai_id: str,
        quality_score: float = 0.0,
        crop_url: str = "",
        observed_face_id: str | None = None,
        canonical_face_id: str | None = None,
        face_id: str | None = None,
        is_pano_keyframe: bool = False,
        defer_new_reid_recognition: bool = False,
    ) -> SbsEnrollmentOutcome:
        """Offer a face-confirmed observation to the formal write queue."""

        recognition_release = (
            SbsRecognitionRelease() if defer_new_reid_recognition else None
        )
        observed_face = normalize_face_id(
            observed_face_id if observed_face_id is not None else face_id
        )
        canonical_face = normalize_face_id(canonical_face_id or observed_face)
        normalized_reid = str(re_id or "").strip()
        require_assigned_reid_state(identity_state)
        normalized_body = gallery_body_value(body_completeness)
        if not normalized_reid:
            return SbsEnrollmentOutcome(
                enrolled=False,
                reason="empty_reid",
                recognition_release=recognition_release,
            )
        if not observed_face:
            return SbsEnrollmentOutcome(
                enrolled=False,
                reason="face_required",
                recognition_release=recognition_release,
            )
        mapped_face = normalize_face_id(
            self._canonical_face_by_reid.get(normalized_reid)
        )
        if mapped_face and canonical_face and mapped_face != canonical_face:
            return SbsEnrollmentOutcome(
                enrolled=False,
                reason="canonical_face_conflict",
                reid_state=REID_STATE_ACTIVE,
                recognition_release=recognition_release,
            )

        canonical_face = canonical_face or observed_face
        self._canonical_face_by_reid[normalized_reid] = canonical_face
        await self._enqueue_formal_feature(
            face_id=canonical_face,
            re_id=normalized_reid,
            body_completeness=normalized_body,
            embedding=embedding,
            device_id=device_id,
            keyframe_id=keyframe_id,
            picture_vector_id=picture_vector_id,
            keyframe_url=keyframe_url,
            crop_url=crop_url,
            person_rect=person_rect,
            timestamp=timestamp,
            ai_id=ai_id,
            quality_score=quality_score,
            is_pano_keyframe=is_pano_keyframe,
            recognition_release=recognition_release,
        )
        return SbsEnrollmentOutcome(
            enrolled=True,
            reason="face_confirmed",
            reid_state=REID_STATE_ACTIVE,
            recognition_release=recognition_release,
        )

    async def attach_new_reid_recognition(
        self,
        release: SbsRecognitionRelease,
        record: dict[str, Any],
        on_ready: PersonRecognitionReadyCallback,
    ) -> None:
        """Attach the record; publish immediately if SBS already persisted."""

        await release.attach(record, on_ready)

    async def _mark_new_reid_persisted(self, item: dict[str, Any]) -> None:
        release = item.get("__newReidRecognitionRelease")
        if release is not None:
            await release.mark_persisted()

    async def asearch_candidates_with_hit_info(
        self, embedding: np.ndarray, query_body_completeness: str,
        device_ids: Optional[list[str]] = None, face_id: Optional[str] = None,
        excluded_reids: Optional[Collection[str]] = None,
        query_timestamp: Optional[int] = None,
    ):
        return await self.pending_index.asearch_candidates_with_hit_info(
            embedding, query_body_completeness, device_ids, face_id,
            excluded_reids, query_timestamp,
        )

    async def flush(self) -> None:
        async with self.queue.flush_lock:
            batch = await self._drain_batch()
            if not batch:
                return
            for item in batch:
                canonical_face = self._canonical_face_by_reid.get(
                    str(item.get("reId") or "")
                )
                if canonical_face:
                    item["faceId"] = canonical_face
            try:
                async with self._lifecycle_lock:
                    outcomes = await self.vector_db.aapply_prototype_updates(batch)
                if len(outcomes) != len(batch):
                    raise RuntimeError(
                        "SBS prototype outcome count does not match batch size"
                    )
                inserted_count = 0
                action_counts: dict[str, int] = {}
                completed_items = []
                retry_items = []
                for outcome in outcomes:
                    outcome_item = outcome.get("item") or {}
                    action = str(outcome.get("action") or "unknown")
                    action_counts[action] = action_counts.get(action, 0) + 1
                    if outcome.get("retry"):
                        retry_items.append(outcome_item)
                        continue
                    if "persisted" not in outcome:
                        raise RuntimeError(
                            "SBS prototype outcome is missing persisted state"
                        )
                    completed_items.append(outcome_item)
                    insert_result = outcome.get("insert_result")
                    persisted = outcome.get("persisted") is True
                    if persisted:
                        await self._mark_new_reid_persisted(outcome_item)
                    if (
                        insert_result is not None
                        and not outcome_item.get("__vectorMetadataNotified", False)
                    ):
                        inserted_count += 1
                        await self._notify_inserted(
                            [outcome_item],
                            insert_result,
                        )
                        outcome_item["__vectorMetadataNotified"] = True
                await self._remove_pending(completed_items)
                await self._requeue(retry_items)
                self.queue.complete(len(completed_items))
                logger.info(
                    "Applied SBS prototype policies: inserted={}, retry={}, "
                    "actions={}",
                    inserted_count,
                    len(retry_items),
                    action_counts,
                )
            except asyncio.CancelledError:
                await self._requeue(batch)
                raise
            except Exception as error:
                logger.error(f"Failed to write SBS person vectors: {error}")
                await self._requeue(batch)
