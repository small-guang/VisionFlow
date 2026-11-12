"""Exercise bounded video work with real controlled asynchronous functions."""

import asyncio
import unittest

import numpy as np

from core.gpu_video_pipeline import process_video_frames
from core.queue_metrics import PipelineMetrics


def frames(indices):
    return ((index, np.full((2, 3, 3), index, dtype=np.uint8)) for index in indices)


class VideoPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_out_of_order_inference_commits_in_input_order(self):
        later_finished = asyncio.Event()
        inferred, committed = [], []

        async def infer(frame):
            index = int(frame[0, 0, 0])
            if index == 8:
                await later_finished.wait()
            inferred.append(index)
            if index == 2:
                later_finished.set()
            return {"index": index}

        async def commit(index, frame, result):
            self.assertEqual(result["index"], index)
            self.assertEqual(int(frame[0, 0, 0]), index)
            committed.append(index)

        summary = await process_video_frames(frames([8, 2, 5]), infer, commit)
        self.assertLess(inferred.index(2), inferred.index(8))
        self.assertEqual(committed, [8, 2, 5])
        self.assertEqual(summary["frames"], 3)
        self.assertGreater(summary["elapsed_seconds"], 0)
        self.assertGreaterEqual(summary["infer_latency_ms"]["p50"], 0)
        self.assertGreaterEqual(summary["infer_latency_ms"]["p95"], summary["infer_latency_ms"]["p50"])
        stages = summary["pipeline_metrics"]["stages"]
        self.assertEqual(stages["video_inference"]["completed_total"], 3)
        self.assertEqual(stages["video_commit"]["completed_total"], 3)

    async def test_slow_commit_bounds_frames_and_parallel_inferences(self):
        release_commit, four_prepared = asyncio.Event(), asyncio.Event()
        pulled = committed = active = peak_active = peak_backlog = prepared = 0

        def source():
            nonlocal pulled, peak_backlog
            for item in frames(range(12)):
                pulled += 1
                peak_backlog = max(peak_backlog, pulled - committed)
                yield item

        async def infer(frame):
            nonlocal active, peak_active, prepared
            active += 1
            peak_active = max(peak_active, active)
            try:
                await asyncio.sleep(0)
                prepared += 1
                if prepared == 4:
                    four_prepared.set()
                return {}
            finally:
                active -= 1

        async def commit(index, frame, result):
            nonlocal committed
            if index == 0:
                await release_commit.wait()
            committed += 1

        task = asyncio.create_task(process_video_frames(source(), infer, commit, concurrency=2, lookahead=2))
        try:
            await asyncio.wait_for(four_prepared.wait(), 1)
            self.assertEqual(pulled, 4)
            self.assertEqual(peak_active, 2)
            release_commit.set()
            summary = await asyncio.wait_for(task, 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(summary["frames"], 12)
        self.assertEqual(committed, 12)
        self.assertLessEqual(peak_backlog, 4)
        self.assertEqual(active, 0)

    async def test_inference_error_propagates_after_sibling_cleanup(self):
        sibling_started, sibling_cleaned = asyncio.Event(), asyncio.Event()
        failure = RuntimeError("inference failed")
        committed = []
        metrics = PipelineMetrics()

        async def infer(frame):
            if int(frame[0, 0, 0]) == 0:
                await sibling_started.wait()
                raise failure
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                sibling_cleaned.set()

        async def commit(index, frame, result):
            committed.append(index)

        with self.assertRaises(RuntimeError) as caught:
            await asyncio.wait_for(process_video_frames(frames(range(3)), infer, commit, metrics=metrics), 1)
        self.assertIs(caught.exception, failure)
        self.assertTrue(sibling_cleaned.is_set())
        self.assertEqual(committed, [])
        stages = metrics.snapshot()["stages"]
        self.assertEqual(stages["video_inference"]["failed"], 1)
        self.assertEqual(stages["video_inference"]["cancelled"], 1)
        self.assertTrue(all(stage["active"] == stage["waiting"] == 0 for stage in stages.values()))

    async def test_commit_error_cancels_inference(self):
        started, cleaned = asyncio.Event(), asyncio.Event()
        failure = OSError("video writer failed")

        async def infer(frame):
            if int(frame[0, 0, 0]) == 0:
                return {}
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        async def commit(index, frame, result):
            await started.wait()
            raise failure

        with self.assertRaises(OSError) as caught:
            await asyncio.wait_for(process_video_frames(frames(range(2)), infer, commit), 1)
        self.assertIs(caught.exception, failure)
        self.assertTrue(cleaned.is_set())

    async def test_empty_video_has_no_latencies_or_callback_work(self):
        async def unused(*args):
            raise AssertionError("empty input must not run callbacks")

        summary = await process_video_frames([], unused, unused)
        self.assertEqual(summary["frames"], 0)
        self.assertEqual(summary["infer_latency_ms"], {"p50": None, "p95": None})
        self.assertGreaterEqual(summary["elapsed_seconds"], 0)

    async def test_invalid_limits_fail_before_consuming_input(self):
        def source():
            raise AssertionError("invalid settings must not consume frames")
            yield

        async def unused(*args):
            raise AssertionError("invalid settings must not run callbacks")

        for limits in ({"concurrency": 0}, {"concurrency": -1}, {"concurrency": True},
                       {"concurrency": 1.5}, {"lookahead": -1}, {"lookahead": True}, {"lookahead": 1.5}):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    await process_video_frames(source(), unused, unused, **limits)


if __name__ == "__main__":
    unittest.main()
