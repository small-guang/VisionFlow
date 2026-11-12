"""锁外候选预取必须保留身份登记与 pending 迁移的并发语义。"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from utils.buffers.sbs_feature_buffer import SbsFeatureBuffer
from utils.reid.person_identity import IdentityRequest, PersonIdentityService
from utils.reid.person_search import PersonReidSearchService
from utils.reid.reid_models import ReidCandidate
from utils.reid.reid_resolution import ReidClassResolver


def candidate(reid="known", *, face="face-1", source="milvus", prototype="p-1"):
    return ReidCandidate(
        reid=reid, score=.95, source=source, threshold=.56,
        body_completeness="full_body", face_id=face, device_id="camera",
        prototype_id=prototype,
    )


class ControlledGallery:
    """在查询入口固定远程结果，由事件控制返回时机。"""

    collection_name = "alogSBS"

    def __init__(self, identity_lock):
        self.identity_lock = identity_lock
        self.revision = 0
        self.active_writers = 0
        self.candidates = []
        self.calls = []
        self.search_hook = None

    def candidate_revision(self):
        return self.revision, self.active_writers

    async def asearch_similar_person_candidates_with_hit_info(self, embedding, **kwargs):
        self.calls.append((kwargs, self.identity_lock.locked()))
        index = len(self.calls)
        matches = [
            item for item in self.candidates
            if item.reid not in kwargs.get("excluded_reids", ())
            and (not kwargs.get("face_id") or item.face_id == kwargs["face_id"])
        ]
        if self.search_hook is not None:
            await self.search_hook(index)
        if not matches:
            return [], 0., "", ""
        highest = max(matches, key=lambda item: item.score)
        return matches, highest.score, highest.reid, highest.source

    async def aapply_prototype_updates(self, items):
        self.revision += 1
        self.active_writers += 1
        try:
            outcomes = []
            for item in items:
                self.candidates.append(candidate(
                    item["reId"], face=item["faceId"], prototype=item["prototypeId"],
                ))
                outcomes.append({
                    "item": item, "action": "insert", "persisted": True,
                    "retry": False,
                    "insert_result": {"primary_keys": [item["prototypeId"]]},
                })
            return outcomes
        finally:
            self.active_writers -= 1
            self.revision += 1


class LockedResolver(ReidClassResolver):
    def __init__(self, testcase):
        super().__init__()
        self.testcase = testcase
        self.calls = 0

    def resolve(self, *args, **kwargs):
        self.testcase.assertTrue(self.testcase.lock.locked())
        self.calls += 1
        return super().resolve(*args, **kwargs)


class PersonIdentityPrefetchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.lock = asyncio.Lock()
        self.gallery = ControlledGallery(self.lock)
        self.inserted = []

        async def inserted(items, *_args):
            self.inserted.extend(items)

        self.buffer = SbsFeatureBuffer(
            vector_db=self.gallery, max_batch_size=100, on_vectors_inserted=inserted,
        )
        self.resolver = LockedResolver(self)
        self.search = PersonReidSearchService(
            person_sbs_embedding_db=self.gallery, sbs_buffer=self.buffer,
            resolver=self.resolver,
        )
        self.created = []
        self.registered = []

        def new_id():
            reid = f"new-{len(self.created) + 1}"
            self.created.append(reid)
            return reid

        self.service = PersonIdentityService(
            identity_lock=self.lock, search_persons=self.search.search,
            search_service=self.search, sbs_buffer=self.buffer,
            logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
            id_factory=new_id,
        )

    def request(self, **changes):
        return replace(IdentityRequest(
            embedding=np.asarray([1., 0.], dtype=np.float32),
            observed_face_id="face-1", person_quality_level=1,
            person_quality_score=8., body_completeness="full_body",
            device_id="camera", timestamp=100, keyframe_id="frame",
            picture_vector_id=1, keyframe_url="frame.jpg",
            person_rect={"x": 0, "y": 0, "w": 20, "h": 40},
            ai_id="ai", segment_index=0,
        ), **changes)

    def register(self, reid):
        self.assertTrue(self.lock.locked())
        self.registered.append(reid)

    async def assign(self, request=None, **kwargs):
        return await self.service.assign(
            request or self.request(), register_assignment=self.register, **kwargs,
        )

    async def add_pending(self, reid="migrating"):
        return await self.buffer.offer_feature(
            observed_face_id="face-1", canonical_face_id="face-1", re_id=reid,
            identity_state="ACTIVE", body_completeness="full_body",
            embedding=np.asarray([1., 0.], dtype=np.float32), device_id="camera",
            keyframe_id="frame", picture_vector_id=1, keyframe_url="frame.jpg",
            person_rect={"x": 0, "y": 0, "w": 20, "h": 40},
            timestamp=100, ai_id="ai", quality_score=8.,
        )

    async def test_stable_gallery_queries_overlap_and_commits_stay_locked(self):
        self.gallery.candidates = [candidate()]
        both_started = asyncio.Event()
        release = asyncio.Event()

        async def hold(index):
            if index == 2:
                both_started.set()
            await release.wait()

        self.gallery.search_hook = hold
        tasks = [asyncio.create_task(self.assign()) for _ in range(2)]
        try:
            await asyncio.wait_for(both_started.wait(), timeout=2)
            self.assertEqual([locked for _, locked in self.gallery.calls], [False, False])
            self.assertEqual(self.registered, [])
            release.set()
            outcomes = await asyncio.gather(*tasks)
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual([outcome.reid for outcome in outcomes], ["known", "known"])
        self.assertEqual(self.resolver.calls, 2)
        self.assertFalse(self.lock.locked())

    async def test_concurrent_empty_queries_create_one_identity_using_latest_pending(self):
        both_started = asyncio.Event()

        async def hold(index):
            if index == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)

        self.gallery.search_hook = hold
        outcomes = await asyncio.gather(*(
            self.assign(self.request(person_quality_level=2, keyframe_id=f"frame-{index}"))
            for index in range(2)
        ))
        self.assertEqual({outcome.reid for outcome in outcomes}, {"new-1"})
        self.assertEqual(sum(outcome.is_new for outcome in outcomes), 1)
        self.assertEqual(self.created, ["new-1"])
        self.assertEqual(len(self.gallery.calls), 2)
        self.assertEqual(len(self.buffer.pending), 2)

    async def test_pending_added_during_prefetch_is_used_without_extra_remote_query(self):
        async def add(index):
            if index == 1:
                await self.add_pending()

        self.gallery.search_hook = add
        outcome = await self.assign()
        self.assertEqual(outcome.reid, "migrating")
        self.assertEqual(len(self.gallery.calls), 1)

    async def test_pending_added_then_flushed_during_prefetch_triggers_full_query(self):
        async def migrate(index):
            if index == 1:
                await self.add_pending()
                await self.buffer.flush()

        self.gallery.search_hook = migrate
        outcome = await self.assign()
        self.assertEqual(outcome.reid, "migrating")
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, False])
        self.assertEqual(len(self.buffer.pending), 0)
        self.assertEqual(self.resolver.calls, 1)

    async def test_replaced_prototype_during_prefetch_is_not_accepted(self):
        self.gallery.candidates = [candidate("stale")]

        async def replace_prototype(index):
            if index == 1:
                self.gallery.candidates = [candidate("current", prototype="p-2")]
                self.gallery.revision += 2

        self.gallery.search_hook = replace_prototype
        outcome = await self.assign()
        self.assertEqual(outcome.reid, "current")
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, False])
        self.assertEqual(self.resolver.calls, 1)

    async def test_active_writer_at_start_uses_single_locked_query(self):
        self.gallery.candidates = [candidate()]
        self.gallery.active_writers = 1
        outcome = await self.assign()
        self.assertEqual(outcome.reid, "known")
        self.assertEqual([locked for _, locked in self.gallery.calls], [True])

    async def test_writer_becomes_active_during_prefetch_forces_locked_query(self):
        self.gallery.candidates = [candidate()]

        async def start_writer(index):
            if index == 1:
                self.gallery.active_writers = 1
                self.gallery.revision += 1

        self.gallery.search_hook = start_writer
        outcome = await self.assign()
        self.assertEqual(outcome.reid, "known")
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, True])

    async def test_changed_same_frame_exclusions_are_sent_to_remote_fallback(self):
        self.gallery.candidates = [candidate("occupied"), candidate("other", prototype="p-2")]
        excluded = set()

        async def occupy(index):
            if index == 1:
                excluded.add("occupied")

        self.gallery.search_hook = occupy
        outcome = await self.assign(excluded_reids_factory=lambda: excluded)
        self.assertEqual(outcome.reid, "other")
        self.assertEqual([call[0]["excluded_reids"] for call in self.gallery.calls],
                         [frozenset(), frozenset({"occupied"})])
        self.assertEqual(outcome.diagnostics.same_frame_excluded_reid_count, 1)

    async def test_mutable_request_exclusions_are_refreshed_without_factory(self):
        self.gallery.candidates = [candidate("occupied"), candidate("other", prototype="p-2")]
        excluded = set()

        async def occupy(index):
            if index == 1:
                excluded.add("occupied")

        self.gallery.search_hook = occupy
        outcome = await self.assign(self.request(excluded_reids=excluded))
        self.assertEqual(outcome.reid, "other")
        self.assertEqual([call[0]["excluded_reids"] for call in self.gallery.calls],
                         [frozenset(), frozenset({"occupied"})])
        self.assertEqual(outcome.diagnostics.same_frame_excluded_reid_count, 1)

    async def test_pending_removal_after_pending_read_invalidates_prefetch(self):
        await self.add_pending("removed")
        original_search = self.buffer.asearch_candidates_with_hit_info
        calls = 0

        async def remove_after_read(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = await original_search(*args, **kwargs)
            if calls == 1:
                await self.buffer._remove_pending(list(self.buffer.pending.values()))
            return result

        with patch.object(self.buffer, "asearch_candidates_with_hit_info", remove_after_read):
            outcome = await self.assign()
        self.assertEqual(outcome.reid, "0")
        self.assertEqual(self.registered, [])
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, False])
        self.assertEqual(self.resolver.calls, 1)

    async def test_prefetch_failure_preserves_exception_without_registration(self):
        failure = RuntimeError("remote query failed")

        async def fail(_index):
            raise failure

        self.gallery.search_hook = fail
        with self.assertRaises(RuntimeError) as raised:
            await self.assign()
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.registered, [])
        self.assertFalse(self.lock.locked())

    async def test_pending_read_resolution_and_registration_do_not_yield(self):
        self.gallery.candidates = [candidate()]
        original_pending = self.buffer.asearch_candidates_with_hit_info
        original_resolve = self.resolver.resolve
        events = []

        async def pending(*args, **kwargs):
            result = await original_pending(*args, **kwargs)
            asyncio.get_running_loop().call_soon(events.append, "next-loop-turn")
            return result

        def resolve(*args, **kwargs):
            events.append("resolve")
            return original_resolve(*args, **kwargs)

        def register(reid):
            self.register(reid)
            events.append("register")

        with patch.object(self.buffer, "asearch_candidates_with_hit_info", pending), \
                patch.object(self.resolver, "resolve", resolve):
            await self.service.assign(self.request(), register_assignment=register)
        self.assertEqual(events, ["resolve", "register"])
        await asyncio.sleep(0)
        self.assertEqual(events, ["resolve", "register", "next-loop-turn"])

    async def test_pending_failure_after_prefetch_does_not_register(self):
        failure = RuntimeError("pending snapshot failed")

        async def fail(*_args, **_kwargs):
            self.assertTrue(self.lock.locked())
            raise failure

        with patch.object(self.buffer, "asearch_candidates_with_hit_info", fail):
            with self.assertRaises(RuntimeError) as raised:
                await self.assign()
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(self.gallery.calls), 1)
        self.assertEqual(self.resolver.calls, 0)
        self.assertEqual(self.registered, [])
        self.assertFalse(self.lock.locked())

    async def test_cancelled_prefetch_releases_lock_and_allows_next_request(self):
        entered = asyncio.Event()

        async def hold(_index):
            entered.set()
            await asyncio.Event().wait()

        self.gallery.search_hook = hold
        task = asyncio.create_task(self.assign())
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.lock.locked())
        self.assertEqual(self.registered, [])
        self.gallery.search_hook = None
        self.gallery.candidates = [candidate()]
        self.assertEqual((await asyncio.wait_for(self.assign(), timeout=2)).reid, "known")

    async def test_fallback_failure_preserves_exception_and_releases_lock(self):
        failure = RuntimeError("fallback query failed")

        async def conflict_then_fail(index):
            if index == 1:
                self.gallery.revision += 2
            else:
                raise failure

        self.gallery.search_hook = conflict_then_fail
        with self.assertRaises(RuntimeError) as raised:
            await self.assign()
        self.assertIs(raised.exception, failure)
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, False, True])
        self.assertEqual(self.registered, [])
        self.assertFalse(self.lock.locked())

    async def test_anonymous_prefetch_match_cannot_enroll_inherited_face(self):
        self.gallery.candidates = [candidate()]
        outcome = await self.assign(self.request(observed_face_id="0", person_quality_level=2))
        self.assertEqual(outcome.reid, "known")
        self.assertEqual(outcome.canonical_face_id, "face-1")
        self.assertEqual(self.buffer.pending, {})
        self.assertEqual(self.created, [])

    async def test_one_physical_prototype_cannot_become_two_anonymous_supports(self):
        await self.add_pending("duplicate")
        item = next(iter(self.buffer.pending.values()))
        item["embedding"] = np.asarray([.8, .6], dtype=np.float32)
        self.gallery.candidates = [
            replace(candidate("duplicate", prototype=item["prototypeId"]), score=.8),
            replace(candidate("raw-runner", face="face-2", prototype="p-2"), score=.7),
        ]
        # 一个 .8 原型的 raw margin 只有 .1，不能借两源重复计数绕过
        # 单锚点门槛并冒充普通双原型类。预取和旧锁内路径都必须去重。
        for active_writers in (0, 1):
            with self.subTest(active_writers=active_writers):
                self.gallery.active_writers = active_writers
                outcome = await self.assign(self.request(observed_face_id="0"))
                self.assertEqual(outcome.reid, "0")
        self.assertEqual(self.registered, [])

    async def test_duplicate_prototype_keeps_higher_score_and_milvus_on_ties(self):
        for pending_score, milvus_score, expected_source in (
            (1., .8, "buffer"), (.8, 1., "milvus"), (1., 1., "milvus"),
        ):
            with self.subTest(pending=pending_score, milvus=milvus_score):
                await self.buffer._remove_pending(list(self.buffer.pending.values()))
                await self.add_pending("same")
                item = next(iter(self.buffer.pending.values()))
                item["embedding"] = np.asarray(
                    [pending_score, np.sqrt(1. - pending_score ** 2)], dtype=np.float32,
                )
                self.gallery.candidates = [replace(
                    candidate("same", prototype=item["prototypeId"]), score=milvus_score,
                )]
                outcome = await self.assign()
                self.assertEqual(outcome.reid, "same")
                self.assertEqual(outcome.diagnostics.support_count, 1)
                self.assertEqual(outcome.source, expected_source)
                self.assertAlmostEqual(outcome.score, max(pending_score, milvus_score))

    async def test_new_identity_recognition_waits_for_first_persisted_prototype(self):
        outcome = await self.assign(self.request(person_quality_level=2))
        self.assertTrue(outcome.is_new)
        release = outcome.new_reid_recognition_release
        self.assertIsNotNone(release)
        published = []

        async def publish(record):
            published.append(record)

        record = {"reId": outcome.reid}
        await self.buffer.attach_new_reid_recognition(release, record, publish)
        self.assertEqual(published, [])
        await self.buffer.flush()
        self.assertEqual(published, [record])
        await self.buffer.attach_new_reid_recognition(release, record, publish)
        self.assertEqual(published, [record])


if __name__ == "__main__":
    unittest.main()
