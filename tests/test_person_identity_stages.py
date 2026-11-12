"""身份临界路径分段必须保留查询快照顺序、锁范围和失败边界。"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.performance import RuntimePerformance
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_ENROLL
from utils.reid.person_identity import IdentityRequest, PersonIdentityService
from utils.reid.person_search import PersonReidSearchService
from utils.reid.reid_models import (
    REID_SEARCH_MODE_SAME_FACE,
    PersonReidSearchResult,
    ReidCandidate,
    ReidClassResolution,
)
from utils.reid.reid_state import REID_STATE_ACTIVE


class PersonIdentityStageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = SimpleNamespace(now=10.0)
        self.metrics = RuntimePerformance(clock=lambda: self.clock.now)
        self.lock = asyncio.Lock()
        self.events = []
        for target, value in (
            ("utils.reid.person_identity.runtime_performance", self.metrics),
            ("utils.reid.person_search.runtime_performance", self.metrics),
            ("utils.reid.person_identity.time.perf_counter", lambda: self.clock.now),
        ):
            mocking = patch(target, value)
            mocking.start()
            self.addCleanup(mocking.stop)

    def advance(self, event, seconds):
        self.assertTrue(self.lock.locked())
        self.events.append(event)
        self.clock.now += seconds

    def request(self, **changes):
        request = IdentityRequest(
            embedding=np.asarray([1.0], dtype=np.float32), observed_face_id="face-1",
            person_quality_level=QUALITY_LEVEL_ENROLL, person_quality_score=8.0,
            body_completeness="full_body", device_id="camera", timestamp=100,
            keyframe_id="frame", picture_vector_id=1, keyframe_url="frame.jpg",
            person_rect={"x": 0, "y": 0, "w": 20, "h": 40}, ai_id="ai", segment_index=0,
        )
        return replace(request, **changes)

    def service(self, search, *, offer=None):
        async def default_offer(**kwargs):
            self.advance("offer", .007)
            return SimpleNamespace(
                enrollment_state="formal_pending", reid_state=REID_STATE_ACTIVE,
                reason="queued", recognition_release=None,
            )
        return PersonIdentityService(
            identity_lock=self.lock, search_persons=search,
            sbs_buffer=SimpleNamespace(offer_feature=offer or default_offer),
            logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        )

    def register(self, reid):
        self.assertEqual(reid, "person-1")
        self.advance("register", .003)

    async def matched_search(self, *_args, **_kwargs):
        self.advance("search", .020)
        return PersonReidSearchResult(reid="person-1", score=.9,
                                      state=REID_STATE_ACTIVE, reason="accepted")

    async def test_search_decision_and_nested_enrollment_partition_the_lock(self):
        result = await self.service(self.matched_search).assign(
            self.request(), register_assignment=self.register)
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(result.reid, "person-1")
        self.assertEqual(self.events, ["search", "register", "offer"])
        for name, expected in (
            ("person_identity_lock_hold", 30),
            ("person_identity_search", 20),
            ("person_identity_decision", 3),
            ("person_identity_enroll", 7),
            ("person_identity_offer_feature", 7),
        ):
            self.assertEqual(values[name]["total_ms"], expected)
            self.assertEqual(values[name]["count"], 1)
            self.assertEqual(values[name]["errors"], 0)
        self.assertFalse(self.lock.locked())

    async def test_anonymous_observation_does_not_enroll_inherited_face_identity(self):
        async def search(*_args, **_kwargs):
            return replace(await self.matched_search(), canonical_face_id="inherited-face")
        result = await self.service(search).assign(
            self.request(observed_face_id="0"), register_assignment=self.register)
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(result.canonical_face_id, "inherited-face")
        self.assertEqual(self.events, ["search", "register"])
        self.assertEqual(values["person_identity_enroll"]["count"], 0)
        self.assertEqual(values["person_identity_offer_feature"]["count"], 0)

    async def test_offer_failure_keeps_exception_and_registration_order(self):
        failure = ValueError("offer failed")
        async def offer(**_kwargs):
            self.advance("offer", .007)
            raise failure
        with self.assertRaises(ValueError) as raised:
            await self.service(self.matched_search, offer=offer).assign(
                self.request(), register_assignment=self.register)
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.events, ["search", "register", "offer"])
        values = self.metrics.snapshot()["operations"]
        for name in ("person_identity_enroll", "person_identity_offer_feature"):
            self.assertEqual(values[name]["total_ms"], 7)
            self.assertEqual(values[name]["errors"], 1)
        self.assertEqual(values["person_identity_search"]["errors"], 0)
        self.assertFalse(self.lock.locked())

    async def test_cancellation_during_search_marks_only_entered_stages(self):
        entered = asyncio.Event()
        async def search(*_args, **_kwargs):
            self.advance("search", .020)
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(self.service(search).assign(
            self.request(), register_assignment=self.register))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(values["person_identity_search"]["total_ms"], 20)
        self.assertEqual(values["person_identity_search"]["errors"], 1)
        self.assertEqual(values["person_identity_decision"]["count"], 0)
        self.assertEqual(values["person_identity_enroll"]["count"], 0)
        self.assertFalse(self.lock.locked())

    def search_service(self, *, pending_failure=None, resolver_failure=None):
        first = ReidCandidate("milvus-person", .8, source="milvus")
        second = ReidCandidate("pending-person", .9, source="buffer")
        async def milvus(embedding, **kwargs):
            self.advance("milvus", .011)
            self.assertEqual(kwargs["excluded_reids"], frozenset({"occupied"}))
            self.assertEqual(kwargs["face_id"], "face-1")
            self.assertEqual(kwargs["query_timestamp"], 100)
            return [first], .8, first.reid, first.source
        async def pending(embedding, **kwargs):
            self.advance("pending", .004)
            self.assertEqual(self.events, ["milvus", "pending"])
            if pending_failure is not None:
                raise pending_failure
            return [second], .9, second.reid, second.source
        def resolve(candidates, **kwargs):
            self.advance("resolve", .003)
            self.assertEqual(candidates, (first, second))
            self.assertEqual(kwargs, {"query_face_id": "face-1", "query_device_id": "camera"})
            if resolver_failure is not None:
                raise resolver_failure
            return ReidClassResolution(reid="pending-person", score=.9, source="buffer",
                                       state=REID_STATE_ACTIVE, reason="accepted")
        return PersonReidSearchService(
            person_sbs_embedding_db=SimpleNamespace(
                asearch_similar_person_candidates_with_hit_info=milvus),
            sbs_buffer=SimpleNamespace(asearch_candidates_with_hit_info=pending),
            resolver=SimpleNamespace(resolve=resolve),
        )

    async def run_search(self, service):
        async with self.lock:
            return await service.search(np.asarray([1.0], dtype=np.float32), "full_body",
                                        None, face_id="face-1", excluded_reids={"occupied"},
                                        query_device_id="camera", query_timestamp=100)

    async def test_search_keeps_milvus_then_pending_then_resolution_snapshot_order(self):
        result = await self.run_search(self.search_service())
        self.assertEqual(self.events, ["milvus", "pending", "resolve"])
        self.assertEqual(result.reid, "pending-person")
        self.assertEqual(result.vector_hit_reid, "pending-person")
        self.assertEqual(result.search_mode, REID_SEARCH_MODE_SAME_FACE)
        values = self.metrics.snapshot()["operations"]
        for name, expected in (("person_reid_milvus_search", 11),
                               ("person_reid_pending_search", 4),
                               ("person_reid_combine_candidates", 0),
                               ("person_reid_resolve", 3)):
            self.assertEqual(values[name]["total_ms"], expected)
            self.assertEqual(values[name]["count"], 1)
            self.assertEqual(values[name]["errors"], 0)

    async def test_pending_failure_is_not_mislabeled_as_milvus_or_resolver_failure(self):
        failure = ValueError("pending failed")
        with self.assertRaises(ValueError) as raised:
            await self.run_search(self.search_service(pending_failure=failure))
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.events, ["milvus", "pending"])
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(values["person_reid_milvus_search"]["errors"], 0)
        self.assertEqual(values["person_reid_pending_search"]["errors"], 1)
        self.assertEqual(values["person_reid_combine_candidates"]["count"], 0)
        self.assertEqual(values["person_reid_resolve"]["count"], 0)

    async def test_resolver_failure_preserves_completed_io_stage_measurements(self):
        failure = ValueError("resolver failed")
        with self.assertRaises(ValueError) as raised:
            await self.run_search(self.search_service(resolver_failure=failure))
        self.assertIs(raised.exception, failure)
        values = self.metrics.snapshot()["operations"]
        self.assertEqual(values["person_reid_milvus_search"]["errors"], 0)
        self.assertEqual(values["person_reid_pending_search"]["errors"], 0)
        self.assertEqual(values["person_reid_resolve"]["errors"], 1)
        self.assertEqual(values["person_reid_resolve"]["total_ms"], 3)


if __name__ == "__main__":
    unittest.main()
