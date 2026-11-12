"""身份锁计时必须区分排队与持有，并保留锁范围、异常和取消行为。"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.face_keyframe_processor import FaceKeyframeProcessor, _PreparedFace
from core.face_observation import FrameFaceObservation
from core.performance import RuntimePerformance
from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_ENROLL
from utils.reid.person_identity import IdentityRequest, PersonIdentityService
from utils.reid.reid_models import PersonReidSearchResult
from utils.reid.reid_state import REID_STATE_ACTIVE


class AdvancingLock:
    """可控等待时间，保留 async context manager 的 acquire/release 边界。"""

    def __init__(self, clock):
        self.clock = clock
        self.active = False

    async def __aenter__(self):
        self.clock.now += .012
        self.active = True

    async def __aexit__(self, *_args):
        self.active = False

    def locked(self):
        return self.active


class IdentityLockTimingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = SimpleNamespace(now=10.0)
        self.metrics = RuntimePerformance(clock=lambda: self.clock.now)
        for target, value in (
            ("core.face_keyframe_processor.runtime_performance", self.metrics),
            ("utils.reid.person_identity.runtime_performance", self.metrics),
            ("core.face_keyframe_processor.time.perf_counter", lambda: self.clock.now),
        ):
            mocking = patch(target, value)
            mocking.start()
            self.addCleanup(mocking.stop)

    def person_request(self):
        return IdentityRequest(
            embedding=np.asarray([1.0], dtype=np.float32), observed_face_id="face-1",
            person_quality_level=QUALITY_LEVEL_ENROLL, person_quality_score=8.0,
            body_completeness="full_body", device_id="camera", timestamp=1,
            keyframe_id="frame", picture_vector_id=1, keyframe_url="camera/frame.jpg",
            person_rect={"x": 0, "y": 0, "w": 20, "h": 40}, ai_id="ai", segment_index=0,
        )

    def person_service(self, lock, search):
        async def offer_feature(**_kwargs):
            self.assertTrue(lock.locked())
            self.clock.now += .007
            return SimpleNamespace(
                enrollment_state="formal_pending", reid_state=REID_STATE_ACTIVE,
                reason="queued", recognition_release=None,
            )

        return PersonIdentityService(
            identity_lock=lock, search_persons=search,
            sbs_buffer=SimpleNamespace(offer_feature=offer_feature),
            logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        )

    def face_processor(self, lock, search):
        # 本测试只观察 commit 边界，不运行图片/推理准备阶段。
        processor = FaceKeyframeProcessor.__new__(FaceKeyframeProcessor)
        processor.face_identity_lock = lock
        processor.search_faces_with_pending = search
        processor.logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)

        async def detection(*_args, **_kwargs):
            self.assertFalse(lock.locked())
            self.clock.now += .050

        async def feature(**_kwargs):
            self.assertTrue(lock.locked())
            self.clock.now += .007

        async def recognition(_record):
            self.assertFalse(lock.locked())
            self.clock.now += .030

        def build_record(*_args, **_kwargs):
            self.assertTrue(lock.locked())
            self.clock.now += .003
            return {}

        processor.queue_face_detection_record = detection
        processor._build_face_recognition_record = build_record
        processor.feature_buffer = SimpleNamespace(
            add_face_feature=feature, add_face_recognition_record=recognition,
        )
        return processor

    async def commit_face(self, processor):
        ctx = KeyframeContext(sn="camera", timestamp=1,
                              object_name="camera/frame.jpg", keyframe_id="frame")
        prepared = _PreparedFace(
            ai_id="ai", detect_time=1, confidence=.9,
            face_rect={"x": 0, "y": 0, "w": 20, "h": 20}, detection_info="",
            quality_score=8.0, quality_level=QUALITY_LEVEL_ENROLL,
            observation=FrameFaceObservation((0, 0, 20, 20), .9),
            recognition_embedding=np.asarray([1.0], dtype=np.float32),
        )
        return await processor._commit_face(ctx, "frame", 1, ctx.object_name, prepared)

    async def test_face_wait_excludes_detection_and_hold_excludes_recognition_enqueue(self):
        lock = AdvancingLock(self.clock)

        async def search(*_args, **_kwargs):
            self.assertTrue(lock.locked())
            self.clock.now += .020
            return "face-1", .99, "face-1"

        result = await self.commit_face(self.face_processor(lock, search))
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(result.assigned_face_id, "face-1")
        self.assertEqual(values["face_identity_lock_wait"]["total_ms"], 12)
        self.assertEqual(values["face_identity_lock_hold"]["total_ms"], 30)
        self.assertEqual(values["face_identity_lock_hold"]["errors"], 0)
        self.assertFalse(lock.locked())

    async def test_person_hold_covers_search_registration_and_enrollment(self):
        lock = AdvancingLock(self.clock)

        async def search(*_args, **_kwargs):
            self.assertTrue(lock.locked())
            self.clock.now += .020
            return PersonReidSearchResult(reid="person-1", score=.9,
                                          state=REID_STATE_ACTIVE, reason="accepted")

        def register(_reid):
            self.assertTrue(lock.locked())
            self.clock.now += .003

        result = await self.person_service(lock, search).assign(
            self.person_request(), register_assignment=register)
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(result.reid, "person-1")
        self.assertEqual(values["person_identity_lock_wait"]["total_ms"], 12)
        self.assertEqual(values["person_identity_lock_hold"]["total_ms"], 30)
        self.assertFalse(lock.locked())

    async def test_person_without_embedding_does_not_acquire_or_record_lock(self):
        lock = AdvancingLock(self.clock)
        service = self.person_service(lock, None)
        await service.assign(replace(self.person_request(), embedding=None),
                             register_assignment=lambda _reid: None)
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(values["person_identity_lock_wait"]["count"], 0)
        self.assertEqual(values["person_identity_lock_hold"]["count"], 0)
        self.assertEqual(self.clock.now, 10)

    async def test_person_search_exception_records_hold_and_releases_lock(self):
        lock = AdvancingLock(self.clock)

        async def fail(*_args, **_kwargs):
            self.clock.now += .020
            raise ValueError("search failed")

        with self.assertRaisesRegex(ValueError, "search failed"):
            await self.person_service(lock, fail).assign(
                self.person_request(), register_assignment=lambda _reid: None)
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(values["person_identity_lock_hold"]["total_ms"], 20)
        self.assertEqual(values["person_identity_lock_hold"]["errors"], 1)
        self.assertFalse(lock.locked())

    async def test_cancellation_while_waiting_does_not_record_acquired_lock_samples(self):
        for kind in ("face", "person"):
            with self.subTest(kind=kind):
                lock = asyncio.Lock()
                await lock.acquire()
                if kind == "face":
                    # detection 输出原本在锁外，测试中当前锁由竞争方持有。
                    processor = self.face_processor(lock, None)
                    async def noop(*_args, **_kwargs):
                        pass
                    processor.queue_face_detection_record = noop
                    work = self.commit_face(processor)
                else:
                    work = self.person_service(lock, None).assign(
                        self.person_request(), register_assignment=lambda _reid: None)
                task = asyncio.create_task(work)
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(lock.locked())
                lock.release()
                values = self.metrics.snapshot()["operations"]
                self.assertEqual(values[f"{kind}_identity_lock_wait"]["count"], 0)
                self.assertEqual(values[f"{kind}_identity_lock_hold"]["count"], 0)

    async def test_cancellation_while_holding_records_error_and_releases_lock(self):
        for kind in ("face", "person"):
            with self.subTest(kind=kind):
                lock = asyncio.Lock()
                entered = asyncio.Event()

                async def wait(*_args, **_kwargs):
                    self.assertTrue(lock.locked())
                    self.clock.now += .020
                    entered.set()
                    await asyncio.Event().wait()

                work = (self.commit_face(self.face_processor(lock, wait)) if kind == "face"
                        else self.person_service(lock, wait).assign(
                            self.person_request(), register_assignment=lambda _reid: None))
                task = asyncio.create_task(work)
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(lock.locked())
                values = self.metrics.snapshot()["operations"]
                self.assertEqual(values[f"{kind}_identity_lock_wait"]["count"], 1)
                self.assertEqual(values[f"{kind}_identity_lock_hold"]["count"], 1)
                self.assertEqual(values[f"{kind}_identity_lock_hold"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
