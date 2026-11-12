"""身份与帧处理依赖的窄接口，不拥有线程池或外部客户端生命周期。"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from utils.reid.face_matching import FaceSampleMatch


class FaceGallery(Protocol):
    async def asearch_face_samples(
        self, embedding: np.ndarray, device_ids: list | None = None,
    ) -> list[FaceSampleMatch]: ...

    def candidate_revision(self) -> tuple[int, int]: ...


class FacePendingView(Protocol):
    async def asearch_pending_face_samples(
        self, embedding: np.ndarray, device_ids: list | None = None,
    ) -> list[FaceSampleMatch]: ...

    def pending_face_removal_revision(self) -> int: ...


class FaceFeatureSink(Protocol):
    async def add_face_feature(self, **feature: Any) -> None: ...


class FaceRecordSink(Protocol):
    async def add_face_detection_record(self, record: dict[str, Any]) -> None: ...

    async def add_face_recognition_record(self, record: dict[str, Any]) -> None: ...


class PersonRecordSink(Protocol):
    async def add_person_detection_record(self, record: dict[str, Any]) -> None: ...

    async def add_person_recognition_record(self, record: dict[str, Any]) -> None: ...


class PictureFeatureSink(Protocol):
    async def add_picture_feature(self, **feature: Any) -> None: ...


class PersonClipSink(Protocol):
    async def add_feature(self, **feature: Any) -> None: ...
