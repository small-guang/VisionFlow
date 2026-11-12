"""单 owner 的身份一致性、阶段协议与持久化门控；不连接外部服务。"""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import numpy as np

from core.identity.face import FaceIdentityRequest, FaceIdentityService
from core.multiprocess.identity import OwnerIdentityHandler
from core.multiprocess.ports import make_visual_ports, wire_data
from core.multiprocess.transport import PeerClosed, current_work_id
from core.person_pipeline.pipeline import duplicate_person_ordinals
from utils.buffers.persistence_ticket import PersistenceTicket
from utils.keyframe.detection_helpers import same_frame_reid_exclusions
from utils.reid.person_identity import IdentityRequest, PersonIdentityService
from utils.reid.reid_models import PersonReidSearchResult


class LoopbackPeer:
    def __init__(self, owner, worker=0, work="work"):
        self.owner, self.worker, self.work = owner, worker, work
        self.requests = []

    async def call(self, method, payload):
        self.requests.append((method, payload))
        self.assert_plain(payload)
        return await self.owner.handle(self.worker, self.work, method, payload)

    @classmethod
    def assert_plain(cls, value):
        if isinstance(value, dict):
            for item in value.values():
                cls.assert_plain(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                cls.assert_plain(item)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise AssertionError(f"non-data IPC object: {type(value)}")


class OwnerIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.records = SimpleNamespace(**{
            f"add_{kind}_record": AsyncMock() for kind in (
                "face_detection", "face_recognition", "person_detection", "person_recognition",
            )
        })

        async def attach(ticket, record, on_ready):
            await ticket.attach(record, on_ready)

        self.output = SimpleNamespace(
            records=self.records, face_pending_view=object(),
            face_features=SimpleNamespace(add_face_feature=AsyncMock()),
            picture_features=SimpleNamespace(add_picture_feature=AsyncMock()),
            tbps=SimpleNamespace(add_feature=AsyncMock()),
            sbs=SimpleNamespace(attach_new_reid_recognition=attach),
        )
        self.owner = OwnerIdentityHandler(
            SimpleNamespace(FACE_SEARCH_PREFETCH_ENABLED=True),
            SimpleNamespace(face=object(), sbs=object()), self.output,
        )
        self.peer = LoopbackPeer(self.owner)
        self.ports = make_visual_ports(self.peer)
        self.addAsyncCleanup(self.owner.close_work, 0, "work")

    async def picture(self, frame=1, *, ports=None):
        ports = ports or self.ports
        await ports["picture_features"].add_picture_feature(
            picture_vector_id=frame, keyframe_id=f"frame-{frame}",
            embedding=np.ones(512, dtype=np.float32), device_id="camera",
            image_path="frame.jpg", timestamp=100, is_pano_keyframe=False,
        )

    async def people(self, count=1, *, frame=1, ports=None):
        ports = ports or self.ports
        await self.picture(frame, ports=ports)
        await ports["face_identity_service"].begin_stage(frame, 0)
        await ports["face_identity_service"].end_stage(frame)
        await ports["person_identity_service"].begin_stage(frame, count)

    async def detection(self, ordinal=0, *, frame=1, kind="person", ports=None):
        record = {"pictureVectorId": frame, "aiId": f"ai-{ordinal}"}
        await getattr((ports or self.ports)["records"], f"add_{kind}_detection_record")(record)

    def person_request(self, ordinal=0, *, frame=1, **kwargs):
        return IdentityRequest(
            embedding=np.ones(2048, dtype=np.float32), observed_face_id="face",
            person_quality_level=2, person_quality_score=8, body_completeness="full_body",
            device_id="camera", timestamp=100, keyframe_id=f"frame-{frame}",
            picture_vector_id=frame, keyframe_url="frame.jpg", person_rect={"x": 0, "y": 0, "w": 20, "h": 40},
            ai_id=f"ai-{ordinal}", segment_index=ordinal, **kwargs,
        )

    def install_person_service(self, *, persisted=False):
        self.ticket = PersistenceTicket(persisted=persisted)

        async def search(*_args, **_kwargs):
            return PersonReidSearchResult(reid=None)

        async def offer(**_kwargs):
            return SimpleNamespace(
                enrollment_state="formal_pending", reid_state="active",
                reason="queued", recognition_release=self.ticket,
            )

        self.owner.person = PersonIdentityService(
            identity_lock=asyncio.Lock(), search_persons=search,
            sbs_buffer=SimpleNamespace(offer_feature=offer), logger=SimpleNamespace(debug=lambda *_: None),
            id_factory=lambda: "new-reid",
        )

    async def test_shared_face_owner_prevents_two_workers_creating_two_ids(self):
        pending, tickets = [], []

        async def search(*_args, **_kwargs):
            return (pending[0], .99, pending[0]) if pending else (None, 0., "")

        async def store(**feature):
            pending.append(feature["face_id"])
            if "new_face_recognition_record" in feature:
                tickets.append(PersistenceTicket(
                    record=feature["new_face_recognition_record"],
                    on_ready=self.records.add_face_recognition_record,
                ))

        self.owner.face = FaceIdentityService(
            face_features=SimpleNamespace(add_face_feature=store), records=self.records,
            logger=SimpleNamespace(error=lambda *_: None), search_faces=search,
            id_factory=lambda: "same-face",
        )

        async def submit(worker):
            ports = make_visual_ports(LoopbackPeer(self.owner, worker, f"work-{worker}"))
            self.addAsyncCleanup(self.owner.close_work, worker, f"work-{worker}")
            await self.picture(worker + 1, ports=ports)
            face = ports["face_identity_service"]
            await face.begin_stage(worker + 1, 1)
            await self.detection(kind="face", frame=worker + 1, ports=ports)
            metadata = {"picture_vector_id": worker + 1, "ai_id": "ai-0"}
            request = FaceIdentityRequest(np.ones(512, dtype=np.float32), 2, metadata)
            context = {"dev_sn": "camera", "keyframe_id": "frame", "picture_vector_id": worker + 1,
                       "face_url": "face.jpg", "timestamp": 100, "ai_id": "ai-0"}
            outcome = await face.commit_record(request, context, ordinal=0)
            await face.complete_detection(worker + 1, 0)
            await face.end_stage(worker + 1)
            return outcome

        outcomes = await asyncio.gather(submit(1), submit(2))
        self.assertEqual([value.face_id for value in outcomes], ["same-face", "same-face"])
        self.assertEqual(sorted(value.status for value in outcomes), [1, 2])
        self.assertEqual(len(tickets), 1)
        self.assertEqual(self.records.add_face_recognition_record.await_count, 1)
        await tickets[0].mark_persisted()
        self.assertEqual(self.records.add_face_recognition_record.await_count, 2)

    async def test_reid_receipt_holds_record_until_persisted_even_after_work_closes(self):
        self.install_person_service()
        await self.people()
        await self.detection()
        person = self.ports["person_identity_service"]
        outcome = await person.assign_remote(self.person_request(), ordinal=0, duplicate_with_ordinals=())
        self.assertIsInstance(outcome.new_reid_recognition_release, str)
        record = {"pictureVectorId": 1, "aiId": "ai-0", "reid": "new-reid", "isNewReid": True}
        await self.ports["sbs_features"].attach_new_reid_recognition(
            outcome.new_reid_recognition_release, record, lambda _: None,
        )
        await person.complete_detection(1, 0)
        await person.end_stage(1)
        await self.owner.close_work(0, "work")
        self.assertEqual(self.owner._works, {})
        self.records.add_person_recognition_record.assert_not_awaited()
        await self.ticket.mark_persisted()
        await self.ticket.mark_persisted()
        self.records.add_person_recognition_record.assert_awaited_once_with(record)

    async def test_persisted_first_receipt_and_duplicate_attach_release_once(self):
        self.install_person_service(persisted=True)
        await self.people()
        await self.detection()
        outcome = await self.ports["person_identity_service"].assign_remote(
            self.person_request(), ordinal=0, duplicate_with_ordinals=(),
        )
        record = {"pictureVectorId": 1, "aiId": "ai-0", "reid": "new-reid"}
        await self.ports["sbs_features"].attach_new_reid_recognition(outcome.new_reid_recognition_release, record, None)
        await self.ports["sbs_features"].attach_new_reid_recognition(outcome.new_reid_recognition_release, record, None)
        self.records.add_person_recognition_record.assert_awaited_once_with(record)

    async def test_receipt_cannot_be_reused_by_another_work(self):
        self.install_person_service()
        await self.people()
        await self.detection()
        outcome = await self.ports["person_identity_service"].assign_remote(
            self.person_request(), ordinal=0, duplicate_with_ordinals=(),
        )
        other = make_visual_ports(LoopbackPeer(self.owner, 1, "other"))
        self.addAsyncCleanup(self.owner.close_work, 1, "other")
        await self.people(ports=other)
        await self.detection(ports=other)
        with self.assertRaises(ValueError):
            await other["sbs_features"].attach_new_reid_recognition(
                outcome.new_reid_recognition_release,
                {"pictureVectorId": 1, "aiId": "ai-0", "reid": "new-reid"}, None,
            )
        self.records.add_person_recognition_record.assert_not_awaited()

    async def test_owner_preserves_nontransitive_same_frame_exclusions(self):
        exclusions_seen = []

        async def search(*_args, **kwargs):
            excluded = set(kwargs.get("excluded_reids", ()))
            exclusions_seen.append(excluded)
            return PersonReidSearchResult(reid="r" if "r" not in excluded else None)

        self.owner.person = PersonIdentityService(
            identity_lock=asyncio.Lock(), search_persons=search, sbs_buffer=object(),
            logger=SimpleNamespace(debug=lambda *_: None),
        )
        await self.people(3)
        person = self.ports["person_identity_service"]
        results = []
        for ordinal, duplicates in enumerate([(), (0,), (1,)]):
            await self.detection(ordinal)
            request = replace(self.person_request(ordinal), observed_face_id="0", person_quality_level=1)
            result = await person.assign_remote(request, ordinal=ordinal, duplicate_with_ordinals=duplicates)
            results.append(result.reid)
            await self.ports["records"].add_person_recognition_record({
                "pictureVectorId": 1, "aiId": f"ai-{ordinal}", "reid": result.reid,
            })
            await person.complete_detection(1, ordinal)
        await person.end_stage(1)
        self.assertEqual(exclusions_seen, [set(), set(), {"r"}])
        self.assertEqual(results, ["r", "r", "0"])

    async def test_stage_order_and_explicit_skipped_detections(self):
        await self.picture()
        with self.assertRaises(ValueError):
            await self.owner.handle(0, "work", "stage.begin", {"picture_vector_id": 1, "stage": "person", "count": 0})
        face = self.ports["face_identity_service"]
        await face.begin_stage(1, 2)
        for ordinal in range(2):
            await self.detection(ordinal, kind="face")
            await face.complete_detection(1, ordinal)
        await face.end_stage(1)
        self.assertEqual(self.owner._works[(0, "work")].frames[1].stage, "face_done")

    async def test_out_of_order_ordinal_is_rejected(self):
        await self.people(2)
        await self.detection()
        with self.assertRaises(ValueError):
            await self.ports["person_identity_service"].complete_detection(1, 1)

    async def test_frame_completion_requires_all_ordinals_and_cannot_repeat(self):
        await self.people(1)
        with self.assertRaises(ValueError):
            await self.owner.validate_frame_complete(0, "work", "frame-1", False)
        await self.detection()
        person = self.ports["person_identity_service"]
        await person.complete_detection(1, 0)
        with self.assertRaises(ValueError):
            await self.owner.validate_frame_complete(0, "work", "frame-1", False)
        await person.end_stage(1)
        await self.owner.validate_frame_complete(0, "work", "frame-1", False)
        with self.assertRaises(ValueError):
            await self.owner.validate_frame_complete(0, "work", "frame-1", False)

    async def test_main_and_pano_with_same_uid_have_independent_stages(self):
        await self.people(0)
        await self.ports["person_identity_service"].end_stage(1)
        await self.ports["picture_features"].add_picture_feature(
            picture_vector_id=2, keyframe_id="frame-1", embedding=np.ones(512),
            device_id="camera", image_path="pano.jpg", timestamp=100, is_pano_keyframe=True,
        )
        face, person = self.ports["face_identity_service"], self.ports["person_identity_service"]
        await face.begin_stage(2, 0)
        await face.end_stage(2)
        await person.begin_stage(2, 0)
        await person.end_stage(2)
        await self.owner.validate_frame_complete(0, "work", "frame-1", False)
        await self.owner.validate_frame_complete(0, "work", "frame-1", True)

    async def test_duplicate_picture_uid_and_kind_is_rejected(self):
        await self.picture()
        method, payload = self.peer.requests[-1]
        with self.assertRaises(ValueError):
            await self.owner.handle(0, "work", method, {**payload, "picture_vector_id": 2})
        self.output.picture_features.add_picture_feature.assert_awaited_once()

    async def test_new_reid_cannot_bypass_receipt_via_direct_records(self):
        self.install_person_service()
        await self.people()
        await self.detection()
        await self.ports["person_identity_service"].assign_remote(self.person_request(), ordinal=0, duplicate_with_ordinals=())
        with self.assertRaises(ValueError):
            await self.ports["records"].add_person_recognition_record({"pictureVectorId": 1, "aiId": "ai-0", "reid": "new-reid"})
        self.records.add_person_recognition_record.assert_not_awaited()

    async def test_cancelled_waiter_does_not_cancel_or_repeat_accepted_output(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def store(**_feature):
            entered.set()
            await release.wait()

        self.output.picture_features.add_picture_feature.side_effect = store
        task = asyncio.create_task(self.picture())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        retry = asyncio.create_task(self.picture())
        await asyncio.sleep(0)
        self.assertFalse(retry.done())
        release.set()
        await retry
        self.output.picture_features.add_picture_feature.assert_awaited_once()

    async def test_slow_frame_does_not_block_another_frames_output(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def store(**feature):
            if feature["picture_vector_id"] == 1:
                entered.set()
                await release.wait()

        self.output.picture_features.add_picture_feature.side_effect = store
        task = asyncio.create_task(self.picture(1))
        await entered.wait()
        await asyncio.wait_for(self.picture(2), 1)
        release.set()
        await task

    async def test_conflicting_duplicate_is_rejected(self):
        await self.picture()
        method, payload = self.peer.requests[-1]
        with self.assertRaises(ValueError):
            await self.owner.handle(0, "work", method, {**payload, "timestamp": 101})
        self.output.picture_features.add_picture_feature.assert_awaited_once()

    async def test_business_query_failure_does_not_poison_work(self):
        self.owner.face.search = AsyncMock(side_effect=RuntimeError("query unavailable"))
        with self.assertRaises(RuntimeError):
            await self.ports["face_identity_service"].search(np.ones(512), .5)
        await self.picture()

    async def test_person_pipeline_matches_local_identity_and_quality_with_fixed_visual_input(self):
        from tests.test_person_pipeline_concurrency import PersonPipelineConcurrencyTests
        helper = PersonPipelineConcurrencyTests()
        local, remote = helper.make_flow(3), helper.make_flow(3)
        await helper.run_flow(local)
        self.owner.person = remote.pipeline.identity_service
        remote.processor.records = self.ports["records"]
        remote.processor.sbs_features = self.ports["sbs_features"]
        remote.pipeline.identity_service = self.ports["person_identity_service"]
        remote.pipeline.features.clip_features = self.ports["person_clip_features"]
        await self.picture()
        await self.ports["face_identity_service"].begin_stage(1, 0)
        await self.ports["face_identity_service"].end_stage(1)
        await helper.run_flow(remote)
        await self.owner.validate_frame_complete(0, "work", "frame-1", False)
        emitted = [call.args[0] for call in self.records.add_person_recognition_record.await_args_list]
        keys = ("reid", "reidStatus", "reidDecision", "reidDecisionReason", "sameFrameExcludedReidCount",
                "personQualityLevel", "personQualityScore", "bodyCompleteness", "reidEnrollmentState")
        self.assertEqual([{key: item.get(key) for key in keys} for item in emitted],
                         [{key: item.get(key) for key in keys} for item in local.buffer.recognitions])
        self.assertEqual(len(emitted), 3)
        self.assertEqual([feature["re_id"] for feature in remote.buffer.features],
                         [feature["re_id"] for feature in local.buffer.features])
        self.assertEqual(self.output.tbps.add_feature.await_count, len(local.buffer.clip_features))

    async def test_remote_face_pipeline_reports_quality_skip_as_completed_ordinal(self):
        from core.algorithm_config import algorithm_config
        from core.face_keyframe_processor import FaceKeyframeProcessor
        from tests.test_keyframe_flow import FaceOnlyTriton, FakeOSS, DummyLogger
        from utils.keyframe.detection_helpers import KeyframeContext
        await self.picture()
        processor = FaceKeyframeProcessor(
            triton_client=FaceOnlyTriton(algorithm_config.face.quality.recognition_min_score - .01),
            identity_service=self.ports["face_identity_service"], records=self.ports["records"],
            logger=DummyLogger(), face_frame_concurrency=1, face_global_concurrency=1,
        )
        ctx = KeyframeContext(sn="camera", timestamp=100, object_name="frame.jpg", keyframe_id="frame-1")
        result = await processor.process_detection(ctx, np.zeros((120, 120, 3), dtype=np.uint8),
                                                  "frame-1", 1, "frame.jpg", FakeOSS())
        self.assertEqual(len(result), 1)
        frame = self.owner._works[(0, "work")].frames[1]
        self.assertEqual((frame.stage, frame.ordinal), ("face_done", 1))
        self.records.add_face_recognition_record.assert_not_awaited()

    async def test_swallowed_transport_failure_prevents_stage_success(self):
        peer = SimpleNamespace(call=AsyncMock(side_effect=PeerClosed("lost")))
        ports = make_visual_ports(peer)
        token = current_work_id.set("failed-work")
        try:
            with self.assertRaises(PeerClosed):
                await ports["face_identity_service"].search(np.ones(512), .5)
            peer.call.side_effect = None
            with self.assertRaises(RuntimeError):
                await ports["person_identity_service"].end_stage(1)
            self.assertEqual(peer.call.await_count, 1)
            self.assertEqual(ports.channel.failed, {"failed-work": "PeerClosed"})
            ports.close_work("failed-work")
            self.assertEqual(ports.channel.failed, {})
        finally:
            current_work_id.reset(token)


class IdentityWireBoundaryTests(unittest.TestCase):
    def test_large_images_callbacks_and_ticket_never_cross_wire(self):
        for value in (np.zeros((32, 32, 3)), lambda: None, PersistenceTicket(), object()):
            with self.assertRaises((TypeError, ValueError)):
                wire_data({"bad": value})

    def test_duplicate_relations_match_original_mask_rule(self):
        segments = []
        for start in (0, 1, 2, 20):
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[:, start:start + 12] = 1
            segments.append({"rect": {"x": start, "y": 0, "w": 12, "h": 32}, "mask": mask})
        relationships = duplicate_person_ordinals(segments)
        for ordinal in range(len(segments)):
            for previous in range(ordinal):
                excluded = same_frame_reid_exclusions({"r": [segments[previous]]}, segments[ordinal])
                self.assertEqual(previous in relationships[ordinal], "r" not in excluded)
