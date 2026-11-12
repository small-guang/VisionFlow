"""唯一 owner 的身份与输出 RPC；不持有视觉数组或跨进程 Future。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields, asdict
import hashlib
import json
import uuid

import numpy as np
from loguru import logger

from core.identity.face import FaceIdentityRequest, FaceIdentityService
from utils.records.recognition_record_payloads import build_face_recognition_record
from utils.reid.person_identity import IdentityRequest, PersonIdentityService
from utils.reid.person_search import PersonReidSearchService

from .ports import wire_data


@dataclass
class _Frame:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stage: str = "new"
    count: int = 0
    ordinal: int = 0
    ai_id: str = ""
    identity: dict | None = None
    recognition: bool = False
    assignments: dict[str, list[int]] = field(default_factory=dict)
    frame_key: tuple[str, bool] | None = None
    reported: bool = False


@dataclass
class _Work:
    frames: dict[int, _Frame] = field(default_factory=dict)
    operations: dict[tuple, tuple[str, asyncio.Task]] = field(default_factory=dict)
    receipts: dict[str, tuple] = field(default_factory=dict)
    searches: set[asyncio.Task] = field(default_factory=set)
    closing: bool = False


class OwnerIdentityHandler:
    """每帧有序，跨帧并发；真正的身份锁和版本复核仍由原服务负责。

    调度器在调用前验证 worker/work 所属关系，close_work 后拒绝迟到请求。
    本类去重仅覆盖当前 work 的本地 RPC，不提供 Kafka 崩溃重放承诺。
    """

    def __init__(self, config, databases, output):
        self.output = output
        self.face = FaceIdentityService(
            face_gallery=databases.face, pending_faces=output.face_pending_view,
            face_features=output.face_features, records=output.records, logger=logger,
            prefetch_enabled=config.FACE_SEARCH_PREFETCH_ENABLED,
        )
        search = PersonReidSearchService(person_sbs_embedding_db=databases.sbs, sbs_buffer=output.sbs)
        self.person = PersonIdentityService(
            identity_lock=asyncio.Lock(), search_persons=search.search,
            search_service=search, sbs_buffer=output.sbs, logger=logger,
        )
        self._works = {}

    async def handle(self, worker_id, work_id, method, payload):
        if not work_id:
            raise ValueError("identity RPC requires a work id")
        work = self._works.setdefault((worker_id, work_id), _Work())
        if work.closing:
            raise RuntimeError("work is closing")
        if method == "identity.face.search":
            vector = self._embedding(payload["embedding"], 512)
            task = asyncio.create_task(self._search_face(vector, payload))
            work.searches.add(task)
            task.add_done_callback(work.searches.discard)
            task.add_done_callback(self._consume_error)
            return await asyncio.shield(task)
        key = self._operation_key(method, payload)
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        previous = work.operations.get(key)
        if previous is not None:
            if previous[0] != digest:
                raise ValueError("conflicting duplicate identity operation")
            return await asyncio.shield(previous[1])
        task = asyncio.create_task(self._dispatch(work, method, payload))
        # 保存执行任务而不是只保存成功结果：取消等待者不得重复入队副作用。
        work.operations[key] = (digest, task)
        task.add_done_callback(self._consume_error)
        return await asyncio.shield(task)

    @staticmethod
    def _consume_error(task):
        if not task.cancelled():
            task.exception()

    async def _search_face(self, vector, payload):
        try:
            result = await self.face.search(vector, payload["threshold"], payload.get("device_ids"))
        except Exception:
            return {"query_failed": True, "result": None}
        return {"query_failed": False, "result": wire_data(result)}

    @staticmethod
    def _operation_key(method, payload):
        frame_id = payload.get("picture_vector_id")
        if type(frame_id) is not int or frame_id <= 0:
            raise ValueError("invalid picture_vector_id")
        if method in {"picture.add", "clip.add"}:
            return method, frame_id, payload.get("ai_id", "")
        if method in {"stage.begin", "stage.end", "stage.complete"}:
            return method, frame_id, payload["stage"], payload.get("ordinal")
        if method in {"identity.face.commit", "identity.person.assign"}:
            return method, frame_id, payload["ordinal"]
        if method == "record.add":
            return method, frame_id, payload["kind"], payload["record"]["aiId"]
        if method == "recognition.attach":
            return method, frame_id, payload["receipt"]
        raise ValueError(f"unsupported owner method: {method}")

    @staticmethod
    def _embedding(value, size):
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError("invalid identity embedding")
        return vector

    async def _dispatch(self, work, method, payload):
        frame_id = payload["picture_vector_id"]
        if method == "picture.add":
            frame = work.frames.setdefault(frame_id, _Frame())
        else:
            frame = work.frames.get(frame_id)
            if frame is None:
                raise ValueError("picture has not been accepted")
        async with frame.lock:
            if method == "picture.add":
                if frame.stage != "new":
                    raise ValueError("picture already accepted")
                frame_key = str(payload["keyframe_id"]), bool(payload["is_pano_keyframe"])
                if any(other is not frame and other.frame_key == frame_key for other in work.frames.values()):
                    raise ValueError("duplicate main or pano picture within work")
                frame.frame_key = frame_key
                feature = dict(payload)
                feature["embedding"] = self._embedding(feature["embedding"], 512)
                await self.output.picture_features.add_picture_feature(**feature)
                frame.stage = "picture"
            elif method.startswith("stage."):
                self._stage(frame, method, payload)
            elif method == "record.add":
                await self._record(frame, payload)
            elif method == "identity.face.commit":
                self._current(frame, "face", payload["ordinal"])
                return await self._commit_face(frame, payload)
            elif method == "identity.person.assign":
                self._current(frame, "person", payload["ordinal"])
                return await self._assign_person(work, frame, payload)
            elif method == "clip.add":
                self._current(frame, "person")
                if not frame.ai_id or payload["ai_id"] != frame.ai_id:
                    raise ValueError("CLIP feature does not match current detection")
                feature = dict(payload)
                feature["embedding"] = self._embedding(feature["embedding"], 512)
                await self.output.tbps.add_feature(**feature)
            elif method == "recognition.attach":
                await self._attach(work, frame, payload)
        return None

    @staticmethod
    def _current(frame, stage, ordinal=None):
        if frame.stage != stage or frame.ordinal >= frame.count:
            raise ValueError("invalid identity stage")
        if ordinal is not None and (type(ordinal) is not int or ordinal != frame.ordinal):
            raise ValueError("identity detection arrived out of order")

    def _stage(self, frame, method, payload):
        stage = payload["stage"]
        if stage not in {"face", "person"}:
            raise ValueError("invalid stage")
        if method == "stage.begin":
            expected = "picture" if stage == "face" else "face_done"
            count = payload["count"]
            if frame.stage != expected or type(count) is not int or count < 0:
                raise ValueError("stage began before its predecessor completed")
            frame.stage, frame.count, frame.ordinal = stage, count, 0
        elif method == "stage.complete":
            self._current(frame, stage, payload["ordinal"])
            if not frame.ai_id:
                raise ValueError("detection must be recorded before completion")
            if frame.identity is not None and not frame.recognition:
                raise ValueError("identity recognition has not been attached")
            frame.ordinal += 1
            frame.ai_id, frame.identity, frame.recognition = "", None, False
        elif method == "stage.end":
            if frame.stage != stage or frame.ordinal != frame.count:
                raise ValueError("stage has unfinished detections")
            frame.stage = f"{stage}_done"

    async def _record(self, frame, payload):
        kind, record = payload["kind"], payload["record"]
        if kind not in {"face_detection", "person_detection", "person_recognition"}:
            raise ValueError("face recognition must be committed by owner")
        stage = kind.split("_")[0]
        self._current(frame, stage)
        if record["pictureVectorId"] != payload["picture_vector_id"]:
            raise ValueError("record frame mismatch")
        if kind.endswith("detection"):
            if frame.ai_id or not record["aiId"]:
                raise ValueError("duplicate detection ordinal")
            await getattr(self.output.records, f"add_{kind}_record")(record)
            frame.ai_id = record["aiId"]
        else:
            self._validate_recognition(frame, record, allow_new=False)
            await self.output.records.add_person_recognition_record(record)
            frame.recognition = True

    @staticmethod
    def _validate_recognition(frame, record, *, allow_new):
        if not frame.ai_id or record["aiId"] != frame.ai_id:
            raise ValueError("recognition detection mismatch")
        if frame.identity is not None:
            if str(record["reid"]) != str(frame.identity["reid"]):
                raise ValueError("recognition identity mismatch")
            if frame.identity["is_new"] and not allow_new:
                raise ValueError("new ReID recognition requires a persistence receipt")
        elif record.get("isNewReid"):
            raise ValueError("unassigned recognition cannot create a ReID")

    async def _commit_face(self, frame, payload):
        data = dict(payload["request"])
        metadata = data["feature_metadata"]
        if (not frame.ai_id or metadata["picture_vector_id"] != payload["picture_vector_id"]
                or metadata["ai_id"] != frame.ai_id or frame.identity is not None):
            raise ValueError("face request detection mismatch")
        data["embedding"] = self._embedding(data["embedding"], 512)
        context = dict(payload["record_context"])
        if context["picture_vector_id"] != payload["picture_vector_id"] or context["ai_id"] != frame.ai_id:
            raise ValueError("face record context mismatch")

        def build_record(outcome):
            return build_face_recognition_record(
                **context, face_id=outcome.face_id, score=outcome.score,
                vector_hit_face_id=outcome.vector_hit_face_id,
                face_status=outcome.status, info=outcome.info,
            )

        outcome = await self.face.commit(FaceIdentityRequest(**data), build_record)
        frame.identity = wire_data(asdict(outcome))
        frame.recognition = True  # 新生记录已与向量在 owner 原子绑定。
        return frame.identity

    async def _assign_person(self, work, frame, payload):
        data = dict(payload["request"])
        if (not frame.ai_id or data["picture_vector_id"] != payload["picture_vector_id"]
                or data["ai_id"] != frame.ai_id or frame.identity is not None):
            raise ValueError("person request detection mismatch")
        ordinal = payload["ordinal"]
        duplicates = payload["duplicate_with_ordinals"]
        if any(type(value) is not int or not 0 <= value < ordinal for value in duplicates):
            raise ValueError("invalid duplicate detection relation")
        duplicates = set(duplicates)
        data["embedding"] = self._embedding(data["embedding"], 2048)
        # 排除集合的权威状态归 owner，视觉进程不能直接覆盖它。
        if data.get("excluded_reids"):
            raise ValueError("worker cannot supply authoritative ReID exclusions")

        def exclusions():
            return {reid for reid, prior in frame.assignments.items()
                    if any(value not in duplicates for value in prior)}

        def register(reid):
            frame.assignments.setdefault(str(reid), []).append(ordinal)

        outcome = await self.person.assign(
            IdentityRequest(**data), register_assignment=register,
            excluded_reids_factory=exclusions,
        )
        result = {item.name: (asdict(outcome.diagnostics) if item.name == "diagnostics"
                             else getattr(outcome, item.name))
                  for item in fields(outcome) if item.name != "new_reid_recognition_release"}
        receipt = None
        if outcome.new_reid_recognition_release is not None:
            receipt = uuid.uuid4().hex
            work.receipts[receipt] = (
                payload["picture_vector_id"], frame.ai_id,
                outcome.new_reid_recognition_release,
            )
        result["new_reid_recognition_release"] = receipt
        frame.identity = wire_data(result)
        return frame.identity

    async def _attach(self, work, frame, payload):
        self._current(frame, "person")
        receipt = work.receipts.get(payload["receipt"])
        if receipt is None or receipt[:2] != (payload["picture_vector_id"], frame.ai_id):
            raise ValueError("receipt does not belong to this work and detection")
        self._validate_recognition(frame, payload["record"], allow_new=True)
        await self.output.sbs.attach_new_reid_recognition(
            receipt[2], payload["record"], self.output.records.add_person_recognition_record,
        )
        frame.recognition = True

    async def validate_frame_complete(self, worker_id, work_id, keyframe_id, is_pano):
        work = self._works.get((worker_id, work_id))
        if work is None or work.closing:
            raise ValueError("unknown or closing work")
        frame_key = str(keyframe_id), bool(is_pano)
        matches = [frame for frame in work.frames.values() if frame.frame_key == frame_key]
        if len(matches) != 1:
            raise ValueError("unknown or ambiguous frame completion")
        frame = matches[0]
        async with frame.lock:
            if frame.stage != "person_done" or frame.reported:
                raise ValueError("frame incomplete or already reported")
            frame.reported = True

    async def close_work(self, worker_id, work_id):
        key = worker_id, work_id
        work = self._works.get(key)
        if work is None:
            return
        work.closing = True
        tasks = [task for _, task in work.operations.values()] + list(work.searches)
        await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)
        # writer 仍拥有 ticket；仅释放 RPC 缓存，不撤销已入队的持久化或输出。
        self._works.pop(key, None)
