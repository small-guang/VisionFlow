"""视觉进程使用的窄端口；跨进程只发送字段、向量和不透明回执。"""
from __future__ import annotations

from dataclasses import fields
from collections.abc import Mapping

import numpy as np

from core.identity.face import FaceIdentityOutcome
from utils.reid.person_identity import IdentityOutcome, ReidDiagnostics

from .transport import current_work_id


def wire_data(value):
    """显式转换 DTO，拒绝回调、锁、ticket、图像对象等意外依赖。"""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, np.generic):
        return wire_data(value.item())
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size > 4096:
            raise ValueError("only bounded embedding vectors may cross identity IPC")
        return value.tolist()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("IPC dictionary keys must be strings")
        return {key: wire_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [wire_data(item) for item in value]
    raise TypeError(f"unsupported identity IPC value: {type(value).__name__}")


def request_data(request):
    return {item.name: wire_data(getattr(request, item.name)) for item in fields(request)}


class _StagePort:
    remote_identity = True

    def __init__(self, peer, stage):
        self.peer, self.stage = peer, stage

    async def begin_stage(self, picture_vector_id, count):
        await self.peer.call("stage.begin", {
            "picture_vector_id": picture_vector_id, "stage": self.stage, "count": count,
        })

    async def complete_detection(self, picture_vector_id, ordinal):
        await self.peer.call("stage.complete", {
            "picture_vector_id": picture_vector_id, "stage": self.stage, "ordinal": ordinal,
        })

    async def end_stage(self, picture_vector_id):
        await self.peer.call("stage.end", {
            "picture_vector_id": picture_vector_id, "stage": self.stage,
        })


class RemoteFaceIdentity(_StagePort):
    def __init__(self, peer):
        super().__init__(peer, "face")

    async def search(self, embedding, threshold, device_ids=None):
        result = await self.peer.call("identity.face.search", wire_data({
            "embedding": embedding, "threshold": threshold, "device_ids": device_ids,
        }))
        if result["query_failed"]:
            raise RuntimeError("owner face gallery search failed")
        return tuple(result["result"])

    async def commit_record(self, request, record_context, *, ordinal):
        result = await self.peer.call("identity.face.commit", {
            "request": request_data(request), "record_context": wire_data(record_context),
            "picture_vector_id": request.feature_metadata["picture_vector_id"],
            "ordinal": ordinal,
        })
        return FaceIdentityOutcome(**result)


class RemotePersonIdentity(_StagePort):
    def __init__(self, peer):
        super().__init__(peer, "person")

    async def assign_remote(self, request, *, ordinal, duplicate_with_ordinals):
        result = await self.peer.call("identity.person.assign", {
            "request": request_data(request), "picture_vector_id": request.picture_vector_id,
            "ordinal": ordinal, "duplicate_with_ordinals": list(duplicate_with_ordinals),
        })
        result = dict(result)
        result["diagnostics"] = ReidDiagnostics(**result["diagnostics"])
        return IdentityOutcome(**result)


class RemoteRecords:
    def __init__(self, peer):
        self.peer = peer

    async def _add(self, kind, record):
        await self.peer.call("record.add", {
            "kind": kind, "record": wire_data(record),
            "picture_vector_id": record["pictureVectorId"],
        })

    async def add_face_detection_record(self, record):
        await self._add("face_detection", record)

    async def add_face_recognition_record(self, record):
        await self._add("face_recognition", record)

    async def add_person_detection_record(self, record):
        await self._add("person_detection", record)

    async def add_person_recognition_record(self, record):
        await self._add("person_recognition", record)


class RemotePictureFeatures:
    def __init__(self, peer):
        self.peer = peer

    async def add_picture_feature(self, **feature):
        await self.peer.call("picture.add", wire_data(feature))


class RemoteClipFeatures:
    def __init__(self, peer):
        self.peer = peer

    async def add_feature(self, **feature):
        await self.peer.call("clip.add", wire_data(feature))


class RemoteSbsFeatures:
    def __init__(self, peer):
        self.peer = peer

    async def attach_new_reid_recognition(self, release, record, on_ready):
        # on_ready 是本进程端口回调；owner 使用自己的 publisher，不传闭包。
        if not isinstance(release, str) or not release:
            raise TypeError("remote ReID recognition requires an opaque receipt")
        await self.peer.call("recognition.attach", {
            "receipt": release, "record": wire_data(record),
            "picture_vector_id": record["pictureVectorId"],
        })


class _RemoteChannel:
    """即使业务层捕获了搜索异常，传输失败也不能变成成功帧。"""

    def __init__(self, peer):
        self.peer = peer
        self.failed = {}

    async def call(self, method, payload):
        work_id = current_work_id.get()
        if work_id in self.failed:
            raise RuntimeError(f"identity RPC previously failed: {self.failed[work_id]}")
        try:
            return await self.peer.call(method, payload)
        except Exception as error:
            # query_failed 业务结果由 Face proxy 转为可降级异常，不经过此分支。
            self.failed[work_id] = type(error).__name__
            raise


class VisualPorts(dict):
    def __init__(self, channel, **ports):
        super().__init__(ports)
        self.channel = channel

    def close_work(self, work_id):
        self.channel.failed.pop(work_id, None)


def make_visual_ports(peer):
    peer = _RemoteChannel(peer)
    return VisualPorts(
        peer,
        face_identity_service=RemoteFaceIdentity(peer),
        person_identity_service=RemotePersonIdentity(peer),
        picture_features=RemotePictureFeatures(peer),
        records=RemoteRecords(peer),
        sbs_features=RemoteSbsFeatures(peer),
        person_clip_features=RemoteClipFeatures(peer),
    )
