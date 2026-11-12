"""离线吞吐分析口径测试，不连接外部服务。"""

import json
from pathlib import Path
import tempfile
import unittest
import uuid
from types import SimpleNamespace

from performance_tools.analyze_run import aggregate_timings, analyze_run, load_summaries, summarize_business
from core.performance import RuntimePerformance


def period(start, end, images, *, seconds=None):
    return {"pid": 1, "interval_start_unix_ms": start * 1000,
            "interval_end_unix_ms": end * 1000,
            "interval_seconds": seconds if seconds is not None else end - start,
            "images_completed_by_topic": {"PtzPersonDetSync": images},
            "messages_completed_by_topic": {"PtzPersonDetSync": images + 2}}


class AnalyzeRunTests(unittest.TestCase):
    def test_weighted_timing_does_not_average_period_averages(self):
        result = aggregate_timings([
            {"count": 1, "total_ms": 100, "avg_ms": 100, "max_ms": 100},
            {"count": 9, "total_ms": 90, "avg_ms": 10, "max_ms": 10},
        ])
        self.assertEqual(result["avg_ms"], 19)
        self.assertEqual(result["max_ms"], 100)

    def test_partial_period_is_retained_without_prorating(self):
        result = summarize_business([period(0, 5, 20), period(5, 10, 10), period(10, 15, 0)],
                                    topic="PtzPersonDetSync", run_start=2, run_end=15, last_ack=4)
        self.assertEqual(result["images_in_captured_periods"], 30)
        self.assertEqual(result["peak_period"]["images_per_second"], 4)
        self.assertFalse(result["peak_period"]["fully_inside_run"])
        self.assertEqual(result["peak_fully_contained_period"]["images_per_second"], 2)
        self.assertEqual(result["active_period_span_images_per_second"], 3)
        self.assertEqual(result["last_observed_topic_image_period_after_last_ack_seconds"], [1, 6])
        self.assertIsNone(result["run_business_drain_seconds"])

    def test_middle_idle_period_remains_in_active_span(self):
        result = summarize_business([period(0, 5, 10), period(5, 10, 0), period(10, 15, 5)],
                                    topic="PtzPersonDetSync", run_start=0, run_end=15, last_ack=1)
        self.assertEqual(result["active_period_span_images_per_second"], 1)

    def test_log_and_summary_periods_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "ALog.log"
            rows = [period(0, 5, 3), period(5, 10, 4), period(10, 15, 7)]
            log.write_text("\n".join("ALog performance summary: " + json.dumps(row) for row in rows))
            result = load_summaries({"started_at": 1, "finished_at": 10,
                                     "performance_summaries": rows[:2]}, [log])
            self.assertEqual(len(result), 2)
            self.assertEqual(sum(row["images_completed_by_topic"]["PtzPersonDetSync"] for row in result), 7)

    def test_uid_picture_outputs_are_separate_from_business_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "summary.json").write_text(json.dumps({
                "run_id": "test", "topic": "PtzPersonDetSync", "started_at": 1,
                "finished_at": 16, "target_qps": 50, "keep_pano": False,
                "performance_summaries": [period(0, 5, 10), period(5, 10, 0)]}))
            sent = [{"uId": "a", "status": "acked", "sent_at": 1, "ack_at": 1.01},
                    {"uId": "b", "status": "acked", "sent_at": 1.1, "ack_at": 1.11}]
            (path / "sent.jsonl").write_text("\n".join(json.dumps(row) for row in sent))
            outputs = [{"uId": "a", "targetType": "picture", "received_at": 6},
                       {"uId": "a", "targetType": "picture", "received_at": 7},
                       {"uId": "b", "targetType": "person", "received_at": 8},
                       {"uId": "other", "targetType": "picture", "received_at": 9}]
            (path / "outputs.jsonl").write_text("\n".join(json.dumps(row) for row in outputs))
            result = analyze_run(path)
            self.assertEqual(result["input_broker_ack"]["acked_messages"], 2)
            self.assertEqual(result["picture_vector_persisted_output"]["observed_images"], 1)
            self.assertAlmostEqual(result["picture_vector_persisted_output"]["observed_images_per_second"], .2)
            self.assertFalse(result["picture_vector_persisted_output"]["all_main_images_observed"])
            self.assertEqual(result["business_images_by_topic"]["images_in_captured_periods"], 10)
            self.assertFalse(result["business_images_by_topic"]["full_pipeline_completion_verified_for_run"])

    def test_pano_does_not_invent_expected_image_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "summary.json").write_text(json.dumps({
                "started_at": 1, "finished_at": 10, "keep_pano": True}))
            result = analyze_run(path)
            self.assertIsNone(result["picture_vector_persisted_output"]["expected_images"])
            self.assertIsNone(result["input_broker_ack"]["messages_per_second"])


class TracedRunAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.run_id = uuid.uuid4().hex
        self.clock = SimpleNamespace(now=0.0, wall=1000.0)
        self.metrics = RuntimePerformance(
            clock=lambda: self.clock.now, wall_clock=lambda: self.clock.wall,
        )
        self.sent = [{"uId": self.uid(i), "status": "acked", "sent_at": 1000 + i * .02,
                      "ack_at": 1000.01 + i * .02} for i in range(2)]

    def uid(self, sequence):
        return f"alogtest-v1:{self.run_id}:{sequence}"

    def complete(self, sequence, at, *, pano=False):
        self.clock.wall = at
        self.clock.now = at - 1000
        self.metrics.image_completed("PtzPersonDetSync", uid=self.uid(sequence), is_pano=pano)

    def snapshot(self, at):
        self.clock.wall = at
        self.clock.now = at - 1000
        return self.metrics.snapshot()

    def analyze(self, rows, **overrides):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            summary = {"run_id": self.run_id, "topic": "PtzPersonDetSync", "trace_completion": True,
                       "started_at": 1000, "finished_at": 1020, "keep_pano": False,
                       "failed_messages": 0, "performance_summaries": rows, **overrides}
            (path / "summary.json").write_text(json.dumps(summary))
            (path / "sent.jsonl").write_text("\n".join(json.dumps(v) for v in self.sent))
            (path / "outputs.jsonl").write_text("\n".join(json.dumps({
                "uId": v["uId"], "targetType": "picture", "received_at": 1001,
            }) for v in self.sent))
            return analyze_run(path)

    def test_cumulative_snapshots_measure_exact_run_finish_after_picture_output(self):
        self.complete(0, 1002.123)
        rows = [self.snapshot(1005)]
        self.complete(1, 1007.456)
        rows.extend([self.snapshot(1010), self.snapshot(1015)])
        result = self.analyze(rows)
        run = result["business_images_by_run"]
        self.assertEqual(run["completed_count"], 2)  # 不累加 1+2+2。
        self.assertTrue(run["count_is_exact"])
        self.assertTrue(run["full_pipeline_completion_verified_for_run"])
        self.assertAlmostEqual(run["first_send_to_last_completion_seconds"], 7.456)
        self.assertAlmostEqual(run["run_business_drain_seconds"], 1007.456 - 1000.03)
        self.assertEqual(result["picture_vector_persisted_output"]["first_send_to_last_output_seconds"], 1)

    def test_duplicate_completions_do_not_hide_a_missing_frame(self):
        self.complete(0, 1002)
        self.complete(0, 1003)
        run = self.analyze([self.snapshot(1005)])["business_images_by_run"]
        self.assertEqual(run["completed_count"], 1)
        self.assertEqual(run["duplicate_main_completions"], 1)
        self.assertFalse(run["all_main_images_completed"])
        self.assertIsNone(run["run_business_drain_seconds"])

    def test_process_restart_or_run_eviction_does_not_invent_total_unique_count(self):
        self.complete(0, 1002)
        first = self.snapshot(1005)
        self.complete(1, 1007)
        second = self.snapshot(1010)
        second["replay_runs"][self.run_id]["generation"] += 1
        run = self.analyze([first, second])["business_images_by_run"]
        self.assertIsNone(run["completed_count"])
        self.assertFalse(run["full_pipeline_completion_verified_for_run"])

    def test_pano_and_noncontiguous_manifests_cannot_claim_full_run_completion(self):
        self.complete(0, 1002)
        self.complete(1, 1003)
        self.complete(0, 1004, pano=True)
        rows = [self.snapshot(1005)]
        run = self.analyze(rows, keep_pano=True)["business_images_by_run"]
        self.assertTrue(run["all_main_images_completed"])
        self.assertFalse(run["full_pipeline_completion_verified_for_run"])
        self.assertEqual(run["pano_completed_count"], 1)
        self.sent[1]["uId"] = self.uid(2)
        run = self.analyze(rows)["business_images_by_run"]
        self.assertFalse(run["all_main_images_completed"])

    def test_legacy_mode_has_no_exact_count_even_when_picture_outputs_are_complete(self):
        result = self.analyze([], trace_completion=False)
        self.assertTrue(result["picture_vector_persisted_output"]["all_main_images_observed"])
        self.assertIsNone(result["business_images_by_run"]["completed_count"])
        self.assertFalse(result["business_images_by_run"]["full_pipeline_completion_verified_for_run"])

    def test_counter_or_clock_regression_does_not_verify_full_completion(self):
        self.complete(0, 1002)
        self.complete(1, 1003)
        first = self.snapshot(1005)
        second = self.snapshot(1010)
        second["replay_runs"][self.run_id]["main"]["completed_count"] = 1
        run = self.analyze([first, second])["business_images_by_run"]
        self.assertFalse(run["count_is_exact"])
        self.assertFalse(run["full_pipeline_completion_verified_for_run"])
        self.assertIsNone(run["first_send_to_last_completion_seconds"])

        self.complete(2, 1001)
        self.sent.append({"uId": self.uid(2), "status": "acked", "sent_at": 1000.04, "ack_at": 1000.05})
        run = self.analyze([self.snapshot(1015)])["business_images_by_run"]
        self.assertFalse(run["full_pipeline_completion_verified_for_run"])
        self.assertTrue(any("wall clock" in reason for reason in run["reasons"]))


if __name__ == "__main__":
    unittest.main()
