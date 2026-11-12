"""Person 有界候选刷新保留原子登记、同帧排除与取消语义。"""

import asyncio
import unittest
from unittest.mock import patch

from core.performance import RuntimePerformance
from tests import test_person_identity_prefetch as fixtures


class PersonIdentityRefreshTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fixtures.PersonIdentityPrefetchTests.setUp(self)
        self.metrics = RuntimePerformance()
        metrics_patch = patch("utils.reid.person_identity.runtime_performance", self.metrics)
        metrics_patch.start()
        self.addCleanup(metrics_patch.stop)

    request = fixtures.PersonIdentityPrefetchTests.request
    register = fixtures.PersonIdentityPrefetchTests.register
    assign = fixtures.PersonIdentityPrefetchTests.assign
    add_pending = fixtures.PersonIdentityPrefetchTests.add_pending

    def counts(self):
        return {name: values["count"]
                for name, values in self.metrics.snapshot()["operations"].items()}

    async def stop_task(self, task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_stable_miss_does_not_refresh_and_retains_first_persistence_ticket(self):
        outcome = await self.assign(self.request(person_quality_level=2))
        self.assertTrue(outcome.is_new)
        self.assertIsNotNone(outcome.new_reid_recognition_release)
        self.assertEqual(len(self.gallery.calls), 1)
        counts = self.counts()
        self.assertEqual(counts["person_identity_requests"], 1)
        self.assertEqual(counts["person_identity_prefetch_used"], 1)
        self.assertEqual(counts["person_identity_refresh"], 0)
        self.assertEqual(counts["person_identity_locked_search"], 0)

    async def test_refresh_releases_lock_and_merges_pending_added_during_remote_read(self):
        refreshing, release = asyncio.Event(), asyncio.Event()

        async def remote(index):
            self.assertFalse(self.lock.locked())
            if index == 1:
                self.gallery.revision += 2
            else:
                refreshing.set()
                await release.wait()

        self.gallery.search_hook = remote
        task = asyncio.create_task(self.assign(self.request(person_quality_level=2)))
        try:
            await asyncio.wait_for(refreshing.wait(), 1)
            async with asyncio.timeout(1):
                async with self.lock:
                    await self.add_pending("added-during-refresh")
            self.assertEqual(self.created, [])
            self.assertEqual(self.registered, [])
            release.set()
            outcome = await asyncio.wait_for(task, 1)
            self.assertEqual(outcome.reid, "added-during-refresh")
            self.assertFalse(outcome.is_new)
            self.assertEqual(self.created, [])
            self.assertEqual(self.registered, ["added-during-refresh"])
            self.assertEqual(len(self.gallery.calls), 2)
            counts = self.counts()
            self.assertEqual(counts["person_identity_prefetch_fallback"], 1)
            self.assertEqual(counts["person_identity_refresh_used"], 1)
            self.assertEqual(counts["person_identity_locked_search"], 0)
        finally:
            release.set()
            await self.stop_task(task)

    async def test_refresh_invalidated_again_uses_one_locked_query(self):
        self.gallery.candidates = [fixtures.candidate("stale")]

        async def remote(index):
            if index <= 2:
                self.gallery.revision += 2
                self.gallery.candidates = [fixtures.candidate(f"revision-{index}")]

        self.gallery.search_hook = remote
        outcome = await self.assign()
        self.assertEqual(outcome.reid, "revision-2")
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, False, True])
        self.assertEqual(self.resolver.calls, 1)
        self.assertEqual(self.registered, ["revision-2"])
        counts = self.counts()
        self.assertEqual(counts["person_identity_refresh"], 1)
        self.assertEqual(counts["person_identity_refresh_invalid"], 1)
        self.assertEqual(counts["person_identity_refresh_exhausted"], 1)
        self.assertEqual(counts["person_identity_locked_search"], 1)
        self.assertEqual(counts["person_identity_prefetch_fallback"], 2)

    async def test_exclusions_are_recomputed_after_refresh_before_locked_fallback(self):
        self.gallery.candidates = [fixtures.candidate(name, prototype=name)
                                   for name in ("first", "second", "available")]
        excluded = set()

        async def remote(index):
            if index == 1:
                excluded.add("first")
            elif index == 2:
                excluded.add("second")

        self.gallery.search_hook = remote
        outcome = await self.assign(excluded_reids_factory=lambda: excluded)
        self.assertEqual(outcome.reid, "available")
        self.assertEqual([args["excluded_reids"] for args, _ in self.gallery.calls],
                         [frozenset(), frozenset({"first"}), frozenset({"first", "second"})])
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, False, True])
        self.assertEqual(outcome.diagnostics.same_frame_excluded_reid_count, 2)

    async def test_mutable_request_exclusions_remain_live_across_refresh(self):
        self.gallery.candidates = [fixtures.candidate(name, prototype=name)
                                   for name in ("first", "second", "available")]
        excluded = set()

        async def remote(index):
            if index < 3:
                excluded.add("first" if index == 1 else "second")

        self.gallery.search_hook = remote
        outcome = await self.assign(self.request(excluded_reids=excluded))
        self.assertEqual(outcome.reid, "available")
        self.assertEqual(self.gallery.calls[-1][0]["excluded_reids"], {"first", "second"})
        self.assertEqual(outcome.diagnostics.same_frame_excluded_reid_count, 2)

    async def test_refresh_failure_falls_back_without_repeating_business_commit(self):
        self.gallery.candidates = [fixtures.candidate()]

        async def remote(index):
            if index == 1:
                self.gallery.revision += 2
            elif index == 2:
                raise RuntimeError("refresh read failed")

        self.gallery.search_hook = remote
        outcome = await self.assign()
        self.assertEqual(outcome.reid, "known")
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, False, True])
        self.assertEqual(self.registered, ["known"])
        counts = self.counts()
        self.assertEqual(counts["person_identity_prefetch_failed"], 1)
        self.assertEqual(counts["person_identity_refresh_failed"], 1)
        self.assertEqual(counts["person_identity_locked_search"], 1)

    async def test_cancelled_refresh_leaves_lock_free_without_side_effects(self):
        refreshing = asyncio.Event()

        async def remote(index):
            if index == 1:
                self.gallery.revision += 2
            else:
                refreshing.set()
                await asyncio.Event().wait()

        self.gallery.search_hook = remote
        task = asyncio.create_task(self.assign(self.request(person_quality_level=2)))
        try:
            await asyncio.wait_for(refreshing.wait(), 1)
            self.assertFalse(self.lock.locked())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(self.lock.locked())
            self.assertEqual(self.registered, [])
            self.assertEqual(self.created, [])
            self.assertEqual(self.buffer.pending, {})
            self.assertEqual(len(self.gallery.calls), 2)
            self.gallery.search_hook = None
            self.gallery.candidates = [fixtures.candidate()]
            self.assertEqual((await asyncio.wait_for(self.assign(), 1)).reid, "known")
        finally:
            await self.stop_task(task)

    async def test_cancelled_reacquire_does_not_release_another_request_lock(self):
        held = asyncio.Event()

        async def remote(index):
            if index == 1:
                self.gallery.revision += 2
            else:
                await self.lock.acquire()
                held.set()

        self.gallery.search_hook = remote
        task = asyncio.create_task(self.assign(self.request(person_quality_level=2)))
        try:
            await asyncio.wait_for(held.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(self.lock.locked())
            self.assertEqual(self.created, [])
            self.assertEqual(self.registered, [])
            self.assertEqual(self.buffer.pending, {})
        finally:
            await self.stop_task(task)
            if self.lock.locked():
                self.lock.release()

    async def test_pending_migration_during_refresh_requires_latest_complete_search(self):
        async def remote(index):
            if index == 1:
                self.gallery.revision += 2
            elif index == 2:
                await self.add_pending("persisted-during-refresh")
                await self.buffer.flush()

        self.gallery.search_hook = remote
        outcome = await self.assign(self.request(person_quality_level=2))
        self.assertEqual(outcome.reid, "persisted-during-refresh")
        self.assertFalse(outcome.is_new)
        self.assertEqual(self.created, [])
        self.assertEqual([locked for _, locked in self.gallery.calls], [False, False, True])
        self.assertEqual(self.resolver.calls, 1)

    async def test_new_identity_after_refresh_still_waits_for_first_persistence(self):
        async def remote(index):
            if index == 1:
                self.gallery.revision += 2

        self.gallery.search_hook = remote
        outcome = await self.assign(self.request(person_quality_level=2))
        published = []

        async def publish(record):
            published.append(record)

        record = {"reid": outcome.reid}
        ticket = outcome.new_reid_recognition_release
        self.assertIsNotNone(ticket)
        await self.buffer.attach_new_reid_recognition(ticket, record, publish)
        self.assertEqual(published, [])
        self.assertEqual(self.created, ["new-1"])
        self.assertEqual(self.registered, ["new-1"])
        await self.buffer.flush()
        await self.buffer.attach_new_reid_recognition(ticket, record, publish)
        self.assertEqual(published, [record])

    async def test_business_exception_after_refresh_is_never_retried(self):
        self.gallery.candidates = [fixtures.candidate()]

        async def remote(index):
            if index == 1:
                self.gallery.revision += 2

        calls = []
        failure = RuntimeError("registration failed")

        def register(reid):
            calls.append(reid)
            raise failure

        self.gallery.search_hook = remote
        with self.assertRaises(RuntimeError) as raised:
            await self.service.assign(self.request(), register_assignment=register)
        self.assertIs(raised.exception, failure)
        self.assertEqual(calls, ["known"])
        self.assertEqual(len(self.gallery.calls), 2)
        self.assertFalse(self.lock.locked())

    async def test_active_writer_never_waits_for_refresh_or_busy_retries(self):
        self.gallery.active_writers = 1
        self.gallery.candidates = [fixtures.candidate()]
        outcome = await self.assign()
        self.assertEqual(outcome.reid, "known")
        self.assertEqual([locked for _, locked in self.gallery.calls], [True])
        counts = self.counts()
        self.assertEqual(counts["person_identity_prefetch_skipped"], 1)
        self.assertEqual(counts["person_identity_refresh_unavailable"], 1)
        self.assertEqual(counts["person_identity_refresh"], 0)
        self.assertEqual(counts["person_identity_locked_search"], 1)

    async def test_concurrent_stale_empty_queries_refresh_and_create_one_identity(self):
        both_reading, release = asyncio.Event(), asyncio.Event()

        async def remote(index):
            if index <= 2:
                self.gallery.revision += 2
                if index == 2:
                    both_reading.set()
                await release.wait()

        self.gallery.search_hook = remote
        tasks = [asyncio.create_task(self.assign(self.request(
            person_quality_level=2, keyframe_id=f"frame-{index}",
        ))) for index in range(2)]
        try:
            await asyncio.wait_for(both_reading.wait(), 1)
            self.assertEqual(self.created, [])
            release.set()
            outcomes = await asyncio.wait_for(asyncio.gather(*tasks), 1)
            self.assertEqual({outcome.reid for outcome in outcomes}, {"new-1"})
            self.assertEqual(sum(outcome.is_new for outcome in outcomes), 1)
            self.assertEqual(self.created, ["new-1"])
            self.assertEqual(self.registered, ["new-1", "new-1"])
            self.assertEqual(len(self.buffer.pending), 2)
            counts = self.counts()
            self.assertEqual(counts["person_identity_refresh"], 2)
            self.assertEqual(counts["person_identity_locked_search"], 0)
        finally:
            release.set()
            for task in tasks:
                await self.stop_task(task)

    async def test_refresh_resolution_to_registration_never_yields(self):
        self.gallery.candidates = [fixtures.candidate()]
        original_pending = self.buffer.asearch_candidates_with_hit_info
        original_resolve = self.resolver.resolve
        events = []

        async def remote(index):
            if index == 1:
                self.gallery.revision += 2

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

        self.gallery.search_hook = remote
        with patch.object(self.buffer, "asearch_candidates_with_hit_info", pending), \
                patch.object(self.resolver, "resolve", resolve):
            await self.service.assign(self.request(), register_assignment=register)
        self.assertEqual(events, ["resolve", "register"])
        self.assertEqual(len(self.gallery.calls), 2)
        await asyncio.sleep(0)
        self.assertEqual(events, ["resolve", "register", "next-loop-turn"])

    async def test_anonymous_refresh_does_not_enroll_inherited_face(self):
        self.gallery.candidates = [fixtures.candidate()]

        async def remote(index):
            if index == 1:
                self.gallery.revision += 2

        self.gallery.search_hook = remote
        outcome = await self.assign(self.request(observed_face_id="0", person_quality_level=2))
        self.assertEqual(outcome.reid, "known")
        self.assertEqual(outcome.canonical_face_id, "face-1")
        self.assertEqual(self.created, [])
        self.assertEqual(self.buffer.pending, {})
        self.assertEqual(len(self.gallery.calls), 2)

    async def test_pending_read_failure_after_refresh_propagates_without_retry(self):
        self.gallery.candidates = [fixtures.candidate()]
        failure = RuntimeError("pending snapshot failed")

        async def remote(index):
            if index == 1:
                self.gallery.revision += 2

        async def pending(*_args, **_kwargs):
            self.assertTrue(self.lock.locked())
            raise failure

        self.gallery.search_hook = remote
        with patch.object(self.buffer, "asearch_candidates_with_hit_info", pending):
            with self.assertRaises(RuntimeError) as raised:
                await self.assign()
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(self.gallery.calls), 2)
        self.assertEqual(self.registered, [])
        self.assertFalse(self.lock.locked())


if __name__ == "__main__":
    unittest.main()
