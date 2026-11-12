import asyncio
import json
from types import SimpleNamespace
import unittest
import uuid

from core.performance import (
    RuntimePerformance, SUMMARY_PREFIX, REPLAY_MAX_RUNS, REPLAY_RUN_TTL_SECONDS,
    parse_replay_uid,
)


class CaptureLogger:
    def __init__(self):
        self.messages = []
        self.emitted = asyncio.Event()

    def info(self, message):
        self.messages.append(message)
        self.emitted.set()


class RuntimePerformanceTests(unittest.TestCase):
    def setUp(self):
        self.clock = SimpleNamespace(now=10.0, wall=1000.0)
        self.metrics = RuntimePerformance(
            clock=lambda: self.clock.now,
            wall_clock=lambda: self.clock.wall,
        )

    def test_sync_timings_include_failures_without_swallowing_them(self):
        with self.metrics.measure("image_decode"):
            self.clock.now += 0.010
        with self.assertRaisesRegex(ValueError, "decode failed"):
            with self.metrics.measure("image_decode"):
                self.clock.now += 0.030
                raise ValueError("decode failed")
        self.clock.now = 15.0
        self.clock.wall = 1005.0

        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["interval_seconds"], 5.0)
        self.assertEqual(snapshot["interval_start_unix_ms"], 1000000)
        self.assertEqual(snapshot["interval_end_unix_ms"], 1005000)
        self.assertEqual(snapshot["operations"]["image_decode"], {
            "count": 2, "errors": 1, "total_ms": 40.0,
            "avg_ms": 20.0, "max_ms": 30.0,
        })
        self.assertEqual(self.metrics.snapshot()["operations"]["image_decode"]["count"], 0)
        # 下一周期开始后，前一快照的计数不能被重置或修改。
        self.assertEqual(snapshot["operations"]["image_decode"]["count"], 2)

    def test_inference_distinguishes_batch_samples_and_missing_rpc_stage(self):
        self.metrics.inference_completed(
            "person_seg", batch_size=4, batch_wait_ms=12,
            executor_queue_wait_ms=3, rpc_ms=20, failed=False,
        )
        self.metrics.inference_completed(
            "person_seg", batch_size=1, batch_wait_ms=8,
            executor_queue_wait_ms=7, rpc_ms=None, failed=True,
        )
        self.metrics.message_completed("PtzPersonDetSync")
        self.metrics.image_completed("PtzPersonDetSync")
        self.metrics.image_completed("PtzPersonDetSync")
        result = self.metrics.snapshot()
        model = result["inference_by_model"]["person_seg"]
        self.assertEqual((model["batches"], model["samples"], model["errors"]), (2, 5, 1))
        self.assertEqual(model["batch_wait_ms"]["avg_ms"], 10.0)
        self.assertEqual(model["executor_queue_wait_ms"]["avg_ms"], 5.0)
        self.assertEqual(model["rpc_ms"], {
            "count": 1, "errors": 0, "total_ms": 20.0,
            "avg_ms": 20.0, "max_ms": 20.0,
        })
        self.assertEqual(result["messages_completed_by_topic"], {"PtzPersonDetSync": 1})
        self.assertEqual(result["images_completed_by_topic"], {"PtzPersonDetSync": 2})
        self.assertFalse(self.metrics.snapshot()["inference_by_model"])

    def test_invalid_timings_do_not_poison_json(self):
        for value in (float("nan"), float("inf"), -1.0):
            self.metrics.record("image_decode", value)
        result = self.metrics.snapshot()
        self.assertEqual(result["operations"]["image_decode"]["count"], 0)
        json.dumps(result, allow_nan=False)


class RuntimePerformanceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_periodic_report_and_final_flush_own_their_background_tasks(self):
        log = CaptureLogger()
        metrics = RuntimePerformance(log=log)
        metrics.start(interval_seconds=0.01, probe_seconds=0.001)
        tasks = list(metrics._tasks)
        metrics.record("person_seg_unpack", 3.25)
        metrics.image_completed("PtzPersonDetSync")
        await asyncio.wait_for(log.emitted.wait(), timeout=1.0)
        metrics.record("image_decode", 7.5)
        await metrics.stop()
        await metrics.stop()

        self.assertTrue(all(task.done() for task in tasks))
        self.assertEqual(len(log.messages), 2)
        reports = [json.loads(message.removeprefix(SUMMARY_PREFIX)) for message in log.messages]
        first, final = reports
        self.assertFalse(first["final"])
        self.assertEqual(first["operations"]["person_seg_unpack"]["avg_ms"], 3.25)
        self.assertGreater(first["operations"]["event_loop_lag"]["count"], 0)
        self.assertEqual(first["images_completed_by_topic"], {"PtzPersonDetSync": 1})
        self.assertTrue(final["final"])
        self.assertEqual(final["operations"]["image_decode"]["avg_ms"], 7.5)


class ReplayCompletionTests(unittest.TestCase):
    def setUp(self):
        self.run_id = uuid.uuid4().hex
        self.clock = SimpleNamespace(now=10.0, wall=1000.0)
        self.metrics = RuntimePerformance(
            clock=lambda: self.clock.now, wall_clock=lambda: self.clock.wall,
        )

    def uid(self, sequence=0):
        return f"alogtest-v1:{self.run_id}:{sequence}"

    def test_only_explicit_valid_test_ids_allocate_run_state(self):
        for uid in (str(uuid.uuid4()), "ordinary", self.uid(-1), self.uid(65536),
                    self.uid("00"), self.uid("1\n"), self.uid().upper(), None):
            self.assertIsNone(parse_replay_uid(uid))
            self.metrics.image_completed("topic", uid=uid)
        self.assertEqual(self.metrics.snapshot()["replay_runs"], {})
        self.assertEqual(parse_replay_uid(self.uid(65535)), (self.run_id, 65535))

    def test_unique_main_and_pano_counters_are_cumulative_across_snapshots(self):
        self.metrics.image_completed("topic", uid=self.uid(1))
        first = self.metrics.snapshot()
        self.clock.wall += 1.234567
        self.metrics.image_completed("topic", uid=self.uid(0))
        self.metrics.image_completed("topic", uid=self.uid(0))
        self.metrics.image_completed("topic", uid=self.uid(0), is_pano=True)
        second = self.metrics.snapshot()
        main = second["replay_runs"][self.run_id]["main"]
        self.assertEqual(main["completed_count"], 2)
        self.assertEqual(main["duplicate_completions"], 1)
        self.assertEqual((main["min_sequence"], main["max_sequence"]), (0, 1))
        self.assertEqual(main["first_completed_unix_ms"], 1000000.0)
        self.assertAlmostEqual(main["last_completed_unix_ms"], 1001234.567)
        self.assertEqual(second["replay_runs"][self.run_id]["pano"]["completed_count"], 1)
        self.assertEqual(first["replay_runs"][self.run_id]["main"]["completed_count"], 1)
        self.assertEqual(second["images_completed_by_topic"], {"topic": 3})

    def test_run_capacity_ttl_and_recreated_generation_are_bounded(self):
        self.metrics.image_completed("topic", uid=self.uid())
        generation = self.metrics.snapshot()["replay_runs"][self.run_id]["generation"]
        for _ in range(REPLAY_MAX_RUNS):
            self.metrics.image_completed("topic", uid=f"alogtest-v1:{uuid.uuid4().hex}:0")
        snapshot = self.metrics.snapshot()
        self.assertEqual(len(snapshot["replay_runs"]), REPLAY_MAX_RUNS)
        self.assertNotIn(self.run_id, snapshot["replay_runs"])
        self.assertEqual(snapshot["replay_run_evictions"], 1)
        self.metrics.image_completed("topic", uid=self.uid())
        self.assertGreater(self.metrics.snapshot()["replay_runs"][self.run_id]["generation"], generation)
        self.clock.now += REPLAY_RUN_TTL_SECONDS
        self.assertEqual(self.metrics.snapshot()["replay_runs"], {})

    def test_topic_conflict_and_wall_clock_jump_are_visible(self):
        self.metrics.image_completed("topic", uid=self.uid())
        self.clock.wall -= 1
        self.metrics.image_completed("topic", uid=self.uid(1))
        self.metrics.image_completed("other", uid=self.uid(2))
        run = self.metrics.snapshot()["replay_runs"][self.run_id]
        self.assertTrue(run["conflicting_topic"])
        self.assertTrue(run["main"]["clock_went_backwards"])
        self.assertEqual(run["main"]["completed_count"], 2)


if __name__ == "__main__":
    unittest.main()
