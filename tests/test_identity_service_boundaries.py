"""身份服务可独立装配；Face 等待不占人物视觉准备容量。"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.image_execution_test_support import InlineImageExecutor

from core.identity import FaceIdentityRequest, FaceIdentityService
from core.keyframe_processor import KeyframeProcessor
from tests.test_face_candidate_prefetch import ControlledGallery, ControlledPending
from tests.test_keyframe_flow import DummyLogger
from tests import test_person_pipeline_concurrency as person_fixtures
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_ENROLL
from utils.keyframe.detection_helpers import KeyframeContext
from utils.reid.face_matching import FaceSampleMatch


class IdentityServiceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_stage_capacities_do_not_read_global_settings(self):
        class NoGlobalSettings:
            def __getattr__(self, name):
                raise AssertionError(f"unexpected global configuration read: {name}")

        with (
            patch("core.keyframe_processor.settings", NoGlobalSettings()),
            patch("core.face_keyframe_processor.settings", NoGlobalSettings()),
            patch("core.person_keyframe_processor.settings", NoGlobalSettings()),
        ):
            flow = KeyframeProcessor(
                image_executor=InlineImageExecutor(),
                triton_client=object(),
                face_identity_service=SimpleNamespace(search=object()),
                person_identity_service=SimpleNamespace(),
                picture_features=object(), records=object(), sbs_features=object(),
                person_clip_features=object(), person_upright_device_sns=frozenset(),
                person_face_reuse_mode="off", person_face_concurrency=7,
                face_frame_concurrency=2, face_global_concurrency=3,
                face_prepare_lookahead=0, person_frame_concurrency=4,
                person_global_concurrency=5, person_prepare_lookahead=6,
            )
        self.assertEqual(flow.face_processor.frame_concurrency, 2)
        self.assertEqual(flow.face_processor.prepare_lookahead, 0)
        self.assertEqual(flow.face_processor._preparation_slots._value, 3)
        person = flow.person_processor._pipeline
        self.assertEqual(person.frame_concurrency, 4)
        self.assertEqual(person.prepare_lookahead, 6)
        self.assertEqual(person._preparation_slots._value, 5)
        self.assertEqual(person._face_resolution_slots._value, 7)

    async def test_frame_flow_accepts_independent_processors_and_holds_admission_for_whole_frame(self):
        events, observed = [], []
        faces = [object()]
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        class Budget:
            active = False

            @asynccontextmanager
            async def frame(self):
                self.active = True
                events.append("admit")
                try:
                    yield SimpleNamespace(observe_image=lambda *args: observed.append(args))
                finally:
                    self.active = False
                    events.append("release")

        budget = Budget()

        async def load(*args):
            self.assertTrue(budget.active)
            events.append("load")
            return image, b"encoded"

        async def image_feature(*args):
            return SimpleNamespace(vector=np.array([1., 0.]))

        async def store_picture(**kwargs):
            self.assertTrue(budget.active)
            events.append("picture")

        async def detect_faces(*args):
            self.assertTrue(budget.active)
            events.append("face")
            return faces

        async def detect_people(*args, frame_faces):
            self.assertTrue(budget.active)
            self.assertIs(frame_faces, faces)
            events.append("person")

        flow = KeyframeProcessor(
            image_executor=InlineImageExecutor(),
            triton_client=SimpleNamespace(extract_image_features=image_feature),
            picture_features=SimpleNamespace(add_picture_feature=store_picture),
            face_processor=SimpleNamespace(process_detection=detect_faces),
            person_processor=SimpleNamespace(process_detection=detect_people),
            image_loader=load, frame_budget=budget,
            keyframe_id_generator=SimpleNamespace(generate=lambda: 1),
        )
        ctx = KeyframeContext(sn="camera", timestamp=1, object_name="image.jpg", keyframe_id="frame")
        self.assertTrue(await flow._process_keyframe_context(ctx, oss_client=object()))
        self.assertEqual(events, ["admit", "load", "picture", "face", "person", "release"])
        self.assertIs(observed[0][0], image)
        self.assertEqual(observed[0][1], b"encoded")

    async def test_independent_ports_share_atomic_face_registration_and_newborn_gate(self):
        gallery, pending = ControlledGallery(), ControlledPending()
        inserted, release_insert = asyncio.Event(), asyncio.Event()
        writes, records = [], []

        async def add_feature(**feature):
            self.assertTrue(service.identity_lock.locked())
            inserted.set()
            await release_insert.wait()
            writes.append(feature)
            pending.samples.append(FaceSampleMatch(
                feature["face_id"], .99, "sample-1", "crop.jpg",
            ))

        async def publish(record):
            self.assertFalse(service.identity_lock.locked())
            records.append(record)

        service = FaceIdentityService(
            face_gallery=gallery, pending_faces=pending,
            face_features=SimpleNamespace(add_face_feature=add_feature),
            records=SimpleNamespace(add_face_recognition_record=publish),
            logger=DummyLogger(), id_factory=lambda: "created-face",
        )
        vector = np.array([1., 0.], dtype=np.float32)
        request = FaceIdentityRequest(vector, QUALITY_LEVEL_ENROLL, {
            "device_id": "camera", "keyframe_id": "frame",
        })
        committing = asyncio.create_task(service.commit(
            request, lambda result: {"face_id": result.face_id, "status": result.status},
        ))
        tasks = [committing]
        try:
            await asyncio.wait_for(inserted.wait(), 1)

            # 共享查询首先等待同一身份协调锁，不能越过尚未登记的 pending。
            reading = asyncio.create_task(service.search(vector, .45))
            tasks.append(reading)
            await asyncio.sleep(0)
            self.assertFalse(reading.done())
            self.assertEqual(records, [])
            release_insert.set()
            created, found = await asyncio.gather(*tasks)
            self.assertEqual(created.face_id, "created-face")
            self.assertEqual(found, ("created-face", .99, "created-face"))
            self.assertEqual(records, [])
            self.assertEqual(writes[0]["new_face_recognition_record"], {
                "face_id": "created-face", "status": 1,
            })
        finally:
            release_insert.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_face_wait_does_not_hold_visual_capacity_and_face_parallelism_is_bounded(self):
        helpers = person_fixtures.PersonPipelineConcurrencyTests()
        flow = helpers.make_flow(
            1, person_global_concurrency=1, person_face_concurrency=1,
        )
        face_entered, release_face, both_embeddings = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(),
        )
        original_face = flow.pipeline.features.match_face
        original_embeddings = flow.pipeline.features.extract_embeddings
        face_active = face_peak = embeddings_done = 0

        async def blocked_face(*args, **kwargs):
            nonlocal face_active, face_peak
            face_active += 1
            face_peak = max(face_peak, face_active)
            face_entered.set()
            try:
                await release_face.wait()
                return await original_face(*args, **kwargs)
            finally:
                face_active -= 1

        async def embeddings(*args, **kwargs):
            nonlocal embeddings_done
            result = await original_embeddings(*args, **kwargs)
            embeddings_done += 1
            if embeddings_done == 2:
                both_embeddings.set()
            return result

        flow.pipeline.features.match_face = blocked_face
        flow.pipeline.features.extract_embeddings = embeddings
        tasks = [asyncio.create_task(helpers.run_flow(flow, "first-frame"))]
        try:
            await asyncio.wait_for(face_entered.wait(), 2)
            tasks.append(asyncio.create_task(helpers.run_flow(flow, "second-frame")))
            # 两帧都能完成 embedding；第一帧的 Face 查询尚未返回。
            await asyncio.wait_for(both_embeddings.wait(), 2)
            self.assertEqual(flow.buffer.recognitions, [])
            self.assertFalse(flow.pipeline._preparation_slots.locked())
            self.assertEqual(face_peak, 1)
            release_face.set()
            await asyncio.gather(*tasks)
            self.assertEqual(len(flow.buffer.recognitions), 2)
            self.assertEqual(face_peak, 1)
        finally:
            release_face.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_cancelling_a_waiting_face_branch_releases_both_resource_domains(self):
        helpers = person_fixtures.PersonPipelineConcurrencyTests()
        flow = helpers.make_flow(
            1, person_global_concurrency=1, person_face_concurrency=1,
        )
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def blocked_face(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        flow.pipeline.features.match_face = blocked_face
        tasks = [asyncio.create_task(helpers.run_flow(flow, "first-frame"))]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            tasks.append(asyncio.create_task(helpers.run_flow(flow, "second-frame")))
            await asyncio.sleep(0)
            for task in tasks:
                task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            self.assertTrue(all(isinstance(value, asyncio.CancelledError) for value in outcomes))
            self.assertTrue(stopped.is_set())
            self.assertFalse(flow.pipeline._face_resolution_slots.locked())
            self.assertFalse(flow.pipeline._preparation_slots.locked())
            self.assertEqual(flow.buffer.recognitions, [])
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
