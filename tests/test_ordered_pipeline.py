import asyncio
import unittest

from core.ordered_pipeline import run_ordered_pipeline
from core.queue_metrics import PipelineMetrics


class OrderedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.metrics = PipelineMetrics()

    async def run_pipeline(self, items, prepare, commit, *, concurrency=2, lookahead=2):
        await run_ordered_pipeline(
            items, prepare, commit,
            concurrency=concurrency, lookahead=lookahead,
            stage_name="ordered_commit", metrics=self.metrics,
        )

    async def wait_until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), timeout=2)

    def waiting(self):
        return self.metrics.snapshot()["stages"].get("ordered_commit", {}).get("waiting", 0)

    async def test_empty_input_does_not_prepare_or_commit(self):
        async def unexpected(_):
            self.fail("empty pipeline called a callback")

        await self.run_pipeline([], unexpected, unexpected)
        self.assertEqual(self.metrics.snapshot()["stages"], {})

    async def test_limits_require_integers_and_allow_zero_lookahead(self):
        async def noop(_):
            pass

        for concurrency in (0, -1, 1.5, True, None):
            with self.subTest(concurrency=concurrency), self.assertRaises(ValueError):
                await self.run_pipeline([], noop, noop, concurrency=concurrency)
        for lookahead in (-1, 1.5, True, None):
            with self.subTest(lookahead=lookahead), self.assertRaises(ValueError):
                await self.run_pipeline([], noop, noop, lookahead=lookahead)
        await self.run_pipeline([], noop, noop, concurrency=1, lookahead=0)

    async def test_slow_head_releases_execution_slots_but_bounds_all_admitted_items(self):
        release_head = asyncio.Event()
        started, pulled, committed = [], [], []
        active = 0
        maximum_active = 0

        def items():
            for item in range(12):
                pulled.append(item)
                yield item

        async def prepare(item):
            nonlocal active, maximum_active
            started.append(item)
            active += 1
            maximum_active = max(active, maximum_active)
            try:
                if item == 0:
                    await release_head.wait()
                else:
                    await asyncio.sleep(0)
                return item
            finally:
                active -= 1

        async def commit(item):
            committed.append(item)

        task = asyncio.create_task(self.run_pipeline(items(), prepare, commit))
        try:
            await self.wait_until(lambda: started == [0, 1, 2, 3] and active == 1)
            self.assertEqual(pulled, [0, 1, 2, 3])
            self.assertEqual(committed, [])
            self.assertEqual(self.waiting(), 3)
            release_head.set()
            await asyncio.wait_for(task, timeout=2)
        finally:
            release_head.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(committed, list(range(12)))
        self.assertEqual(maximum_active, 2)
        self.assertEqual(active, 0)
        self.assertEqual(self.waiting(), 0)

    async def test_lookahead_runs_while_commit_is_blocked_and_includes_committing_item(self):
        release_commit = asyncio.Event()
        commit_started = asyncio.Event()
        prepared, committed = [], []

        async def prepare(item):
            prepared.append(item)
            return item

        async def commit(item):
            if item == 0:
                commit_started.set()
                await release_commit.wait()
            committed.append(item)

        task = asyncio.create_task(self.run_pipeline(
            range(20), prepare, commit, concurrency=2, lookahead=3,
        ))
        try:
            await asyncio.wait_for(commit_started.wait(), timeout=2)
            await self.wait_until(lambda: len(prepared) == 5)
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertEqual(prepared, [0, 1, 2, 3, 4])
            self.assertEqual(self.waiting(), 4)
            self.assertEqual(committed, [])
            release_commit.set()
            await asyncio.wait_for(task, timeout=2)
        finally:
            release_commit.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(committed, list(range(20)))
        self.assertEqual(self.waiting(), 0)

    async def test_zero_lookahead_preserves_window_until_head_commit_finishes(self):
        release_head = asyncio.Event()
        release_commit = asyncio.Event()
        commit_started = asyncio.Event()
        started = []

        async def prepare(item):
            started.append(item)
            if item == 0:
                await release_head.wait()
            return item

        async def commit(item):
            if item == 0:
                commit_started.set()
                await release_commit.wait()

        task = asyncio.create_task(self.run_pipeline(
            range(6), prepare, commit, lookahead=0,
        ))
        try:
            await self.wait_until(lambda: self.waiting() == 1)
            self.assertEqual(started, [0, 1])
            release_head.set()
            await asyncio.wait_for(commit_started.wait(), timeout=2)
            self.assertEqual(started, [0, 1])
            self.assertEqual(self.waiting(), 1)
            release_commit.set()
            await asyncio.wait_for(task, timeout=2)
        finally:
            release_head.set()
            release_commit.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(started, list(range(6)))

    async def test_first_item_commits_without_waiting_for_other_preparations(self):
        release_later = asyncio.Event()
        first_committed = asyncio.Event()
        committed = []

        async def prepare(item):
            if item:
                await release_later.wait()
            return item

        async def commit(item):
            committed.append(item)
            if item == 0:
                first_committed.set()

        task = asyncio.create_task(self.run_pipeline(range(4), prepare, commit))
        try:
            await asyncio.wait_for(first_committed.wait(), timeout=2)
            self.assertEqual(committed, [0])
            self.assertEqual(self.waiting(), 0)
            release_later.set()
            await asyncio.wait_for(task, timeout=2)
        finally:
            release_later.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(committed, [0, 1, 2, 3])

    async def test_ready_wait_excludes_preparation_and_running_commit(self):
        now = [1.0]
        self.metrics = PipelineMetrics(clock=lambda: now[0])
        release_first_prepare = asyncio.Event()
        release_first_commit = asyncio.Event()
        first_commit_started = asyncio.Event()

        async def prepare(item):
            if item == 0:
                await release_first_prepare.wait()
            return item

        async def commit(item):
            if item == 0:
                first_commit_started.set()
                await release_first_commit.wait()

        task = asyncio.create_task(self.run_pipeline([0, 1], prepare, commit))
        try:
            await self.wait_until(lambda: self.waiting() == 1)
            now[0] = 4.0
            before = self.metrics.snapshot()["stages"]["ordered_commit"]
            self.assertEqual((before["waiting"], before["oldest_wait_ms"]), (1, 3000))
            release_first_prepare.set()
            await asyncio.wait_for(first_commit_started.wait(), timeout=2)
            during = self.metrics.snapshot()["stages"]["ordered_commit"]
            self.assertEqual((during["waiting"], during["waits_finished"]), (1, 1))
            self.assertEqual(during["avg_wait_ms"], 0)
            now[0] = 6.0
            release_first_commit.set()
            await asyncio.wait_for(task, timeout=2)
        finally:
            release_first_prepare.set()
            release_first_commit.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        after = self.metrics.snapshot()["stages"]["ordered_commit"]
        self.assertEqual((after["waiting"], after["waits_finished"]), (0, 1))
        self.assertEqual(after["avg_wait_ms"], 5000)
        self.assertEqual(after["completed_total"], 0)

    async def test_later_preparation_failure_preserves_earlier_commits_and_cancels_tail(self):
        fail = asyncio.Event()
        release_head = asyncio.Event()
        started, committed, cancelled = [], [], []
        slots = asyncio.Semaphore(3)

        async def prepare(item):
            async with slots:
                started.append(item)
                try:
                    if item == 0:
                        await release_head.wait()
                    elif item == 2:
                        await fail.wait()
                        raise ValueError("prepare failed")
                    elif item == 3:
                        await asyncio.Event().wait()
                    return item
                except asyncio.CancelledError:
                    cancelled.append(item)
                    raise

        async def commit(item):
            committed.append(item)

        task = asyncio.create_task(self.run_pipeline(
            range(10), prepare, commit, concurrency=3, lookahead=2,
        ))
        try:
            await self.wait_until(lambda: started == [0, 1, 2, 3])
            fail.set()
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(committed, [])
            release_head.set()
            with self.assertRaisesRegex(ValueError, "prepare failed"):
                await asyncio.wait_for(task, timeout=2)
        finally:
            release_head.set()
            fail.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(committed, [0, 1])
        self.assertEqual(started, [0, 1, 2, 3])
        self.assertEqual(cancelled, [3])
        self.assertEqual(self.waiting(), 0)
        await self.assert_slots_recovered(slots, 3)

    async def test_later_failure_does_not_cancel_current_commit(self):
        release_commit = asyncio.Event()
        commit_started = asyncio.Event()
        failure_reported = asyncio.Event()
        committed = []

        async def prepare(item):
            if item == 1:
                await commit_started.wait()
                failure_reported.set()
                raise RuntimeError("later failure")
            return item

        async def commit(item):
            commit_started.set()
            await release_commit.wait()
            committed.append(item)

        task = asyncio.create_task(self.run_pipeline([0, 1], prepare, commit))
        try:
            await asyncio.wait_for(failure_reported.wait(), timeout=2)
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertFalse(task.done())
            release_commit.set()
            with self.assertRaisesRegex(RuntimeError, "later failure"):
                await asyncio.wait_for(task, timeout=2)
        finally:
            release_commit.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(committed, [0])
        self.assertEqual(self.waiting(), 0)

    async def test_first_preparation_failure_cancels_tail_without_committing(self):
        tail_started = asyncio.Event()
        tail_ended = asyncio.Event()

        async def prepare(item):
            if item == 0:
                await tail_started.wait()
                raise ValueError("first prepare failed")
            tail_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                tail_ended.set()

        async def commit(_):
            self.fail("failed preparation cannot commit")

        with self.assertRaisesRegex(ValueError, "first prepare failed"):
            await asyncio.wait_for(
                self.run_pipeline(range(10), prepare, commit), timeout=2,
            )
        self.assertTrue(tail_ended.is_set())
        self.assertEqual(self.waiting(), 0)

    async def test_commit_failure_drains_preparations_ready_tickets_and_shared_slots(self):
        release_failure = asyncio.Event()
        started, cancelled = [], []
        slots = asyncio.Semaphore(2)

        async def prepare(item):
            async with slots:
                started.append(item)
                try:
                    if item in (1, 3):
                        await asyncio.Event().wait()
                    return item
                except asyncio.CancelledError:
                    cancelled.append(item)
                    raise

        async def commit(_):
            await release_failure.wait()
            raise RuntimeError("commit failed")

        task = asyncio.create_task(self.run_pipeline(range(8), prepare, commit))
        try:
            await self.wait_until(lambda: started == [0, 1, 2, 3])
            self.assertEqual(self.waiting(), 1)
            release_failure.set()
            with self.assertRaisesRegex(RuntimeError, "commit failed"):
                await asyncio.wait_for(task, timeout=2)
        finally:
            release_failure.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertCountEqual(cancelled, [1, 3])
        self.assertEqual(self.waiting(), 0)
        await self.assert_slots_recovered(slots, 2)

    async def assert_slots_recovered(self, slots, count):
        async def recover():
            for _ in range(count):
                await slots.acquire()
            for _ in range(count):
                slots.release()

        await asyncio.wait_for(recover(), timeout=2)
        recovered = []

        async def prepare(item):
            async with slots:
                return item

        async def commit(item):
            recovered.append(item)

        await self.run_pipeline(range(5), prepare, commit)
        self.assertEqual(recovered, list(range(5)))
        self.assertEqual(self.waiting(), 0)

    async def test_cancellation_drains_running_commit_preparations_and_ready_results(self):
        started, ended = [], []
        commit_started = asyncio.Event()
        commit_ended = asyncio.Event()

        async def prepare(item):
            started.append(item)
            try:
                if item in (1, 3):
                    await asyncio.Event().wait()
                return item
            finally:
                ended.append(item)

        async def commit(_):
            commit_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                commit_ended.set()

        task = asyncio.create_task(self.run_pipeline(range(8), prepare, commit))
        await asyncio.wait_for(commit_started.wait(), timeout=2)
        await self.wait_until(lambda: started == [0, 1, 2, 3])
        self.assertEqual(self.waiting(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        self.assertTrue(commit_ended.is_set())
        self.assertCountEqual(ended, started)
        self.assertEqual(self.waiting(), 0)

    async def test_repeated_cancellation_waits_for_owned_task_cleanup(self):
        prepare_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()
        cleanup_done = asyncio.Event()

        async def prepare(_):
            prepare_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await cleanup_release.wait()
                cleanup_done.set()

        async def commit(_):
            self.fail("cancelled preparation cannot commit")

        task = asyncio.create_task(self.run_pipeline([0], prepare, commit))
        try:
            await asyncio.wait_for(prepare_started.wait(), timeout=2)
            task.cancel()
            await asyncio.wait_for(cleanup_started.wait(), timeout=2)
            task.cancel()
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertFalse(task.done())
            cleanup_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        finally:
            cleanup_release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(cleanup_done.is_set())
        self.assertEqual(self.waiting(), 0)

    async def test_each_frame_has_independent_commit_progress(self):
        release_slow = asyncio.Event()
        slow_started = asyncio.Event()
        committed_fast = []

        async def prepare_slow(item):
            slow_started.set()
            await release_slow.wait()
            return item

        async def prepare_fast(item):
            return item

        async def commit_fast(item):
            committed_fast.append(item)

        async def commit_slow(_):
            pass

        slow = asyncio.create_task(self.run_pipeline([0], prepare_slow, commit_slow))
        try:
            await asyncio.wait_for(slow_started.wait(), timeout=2)
            await asyncio.wait_for(
                self.run_pipeline(range(10), prepare_fast, commit_fast), timeout=2,
            )
            self.assertEqual(committed_fast, list(range(10)))
            self.assertFalse(slow.done())
            release_slow.set()
            await asyncio.wait_for(slow, timeout=2)
        finally:
            release_slow.set()
            slow.cancel()
            await asyncio.gather(slow, return_exceptions=True)
        self.assertEqual(self.waiting(), 0)


if __name__ == "__main__":
    unittest.main()
