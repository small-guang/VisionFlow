"""验证压测直方图聚合与窗口边界，均不连接业务服务。"""
import tempfile
import unittest
from pathlib import Path

from performance_tools.observe_benchmark import aggregate_operations, collection_metrics, histogram_delta, parse_prometheus, pipeline_queues, read_app_log


class BenchmarkSummaryTests(unittest.TestCase):
    def test_histogram_label_filter_weighted_percentiles_and_units(self):
        before = parse_prometheus('''
milvus_proxy_collection_sq_latency_bucket{collection_name="alogFace",query_type="search",le="1"} 1
milvus_proxy_collection_sq_latency_bucket{collection_name="alogFace",query_type="search",le="8"} 1
milvus_proxy_collection_sq_latency_bucket{collection_name="alogFace",query_type="search",le="+Inf"} 1
milvus_proxy_collection_sq_latency_count{collection_name="alogFace",query_type="search"} 1
milvus_proxy_collection_sq_latency_sum{collection_name="alogFace",query_type="search"} 1
milvus_proxy_collection_sq_latency_count{collection_name="alogFace",query_type="query"} 100
''')
        after = parse_prometheus('''
milvus_proxy_collection_sq_latency_bucket{collection_name="alogFace",query_type="search",le="1"} 2
milvus_proxy_collection_sq_latency_bucket{collection_name="alogFace",query_type="search",le="8"} 5
milvus_proxy_collection_sq_latency_bucket{collection_name="alogFace",query_type="search",le="+Inf"} 5
milvus_proxy_collection_sq_latency_count{collection_name="alogFace",query_type="search"} 5
milvus_proxy_collection_sq_latency_sum{collection_name="alogFace",query_type="search"} 15
milvus_proxy_collection_sq_latency_count{collection_name="alogFace",query_type="query"} 500
''')
        result = histogram_delta(before, after, 'milvus_proxy_collection_sq_latency', {'collection_name': 'alogFace', 'query_type': 'search'}, 'ms')
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['mean'], 3.5)
        self.assertEqual(result['percentile_upper_bound']['95'], 8)
        self.assertEqual(result['unit'], 'ms')
        self.assertEqual(histogram_delta(after, before, 'milvus_proxy_collection_sq_latency', {'query_type': 'search'}, 'ms')['reason'], 'counter_reset')
        self.assertEqual(histogram_delta(before, {}, 'milvus_proxy_collection_sq_latency', {}, 'ms')['reason'], 'metric_missing')
        self.assertIsNone(histogram_delta(before, before, 'milvus_proxy_collection_sq_latency', {'query_type': 'search'}, 'ms')['percentile_upper_bound']['95'])

    def test_app_complete_intervals_only_and_weighted_mean(self):
        import json
        def row(start, end, count, total):
            return {'interval_start_unix_ms': start, 'interval_end_unix_ms': end, 'operations': {'event_loop_lag': {'count': count, 'total_ms': total, 'max_ms': 7, 'errors': 0}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log'
            rows = [row(0, 1000, 1, 2), row(1000, 2000, 2, 8), row(2000, 3000, 3, 9)]
            path.write_text(''.join('ALog performance summary: ' + json.dumps(x) + '\n' for x in rows))
            selected, error = read_app_log(path, 1, 3)
        self.assertIsNone(error)
        result = aggregate_operations(selected)['event_loop_lag']
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['mean_ms'], 3.4)

    def test_queue_snapshots_separate_sessions_and_reject_missing_identity(self):
        import json
        def record(session, pid, queued):
            return {"source_path": "/logs/a", "snapshot": {"session_id": session, "pid": pid,
                    "queues": {"input": {"queued_messages": queued}}}}
        records = [record("first", 10, 2), record("first", 10, 3), record("second", 11, 20), record(None, 11, 99)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in records))
            result = pipeline_queues(path)
        self.assertEqual(result["invalid_identity_samples"], 1)
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual(result["groups"][0]["values"]["input_queued_messages"]["max"], 3)
        self.assertEqual(result["groups"][1]["values"]["input_queued_messages"]["max"], 20)

    def test_querynode_gauges_keep_labels_in_raw_samples(self):
        metrics = parse_prometheus('''
milvus_querynode_read_task_ready_len{node_id="1"} 4
milvus_querynode_read_task_unsolved_len{node_id="1"} 2
milvus_querynode_read_task_concurrency{node_id="1"} 8
milvus_querynode_segment_num{collection="alogFace",state="growing"} 3
other_private_metric{key="x"} 11
''')
        self.assertEqual(metrics[("milvus_querynode_read_task_ready_len", (("node_id", "1"),))], 4)
        self.assertEqual(metrics[("milvus_querynode_read_task_unsolved_len", (("node_id", "1"),))], 2)
        self.assertEqual(metrics[("milvus_querynode_read_task_concurrency", (("node_id", "1"),))], 8)
        self.assertEqual(metrics[("milvus_querynode_segment_num", (("collection", "alogFace"), ("state", "growing")))], 3)
        self.assertEqual(len(metrics), 4)

    def test_custom_collection_filters_without_mutating_defaults(self):
        custom = collection_metrics("isolatedFace", "isolatedSBS")
        defaults = collection_metrics("alogFace", "alogSBS")
        self.assertEqual(custom["face_search"][1]["collection_name"], "isolatedFace")
        self.assertEqual(custom["sbs_search"][1]["collection_name"], "isolatedSBS")
        self.assertEqual(defaults["face_search"][1]["collection_name"], "alogFace")
        before = parse_prometheus('''
milvus_proxy_collection_sq_latency_count{collection_name="alogFace",query_type="search"} 5
milvus_proxy_collection_sq_latency_sum{collection_name="alogFace",query_type="search"} 5
milvus_proxy_collection_sq_latency_count{collection_name="isolatedFace",query_type="search"} 3
milvus_proxy_collection_sq_latency_sum{collection_name="isolatedFace",query_type="search"} 6
''')
        after = parse_prometheus('''
milvus_proxy_collection_sq_latency_count{collection_name="alogFace",query_type="search"} 8
milvus_proxy_collection_sq_latency_sum{collection_name="alogFace",query_type="search"} 11
milvus_proxy_collection_sq_latency_count{collection_name="isolatedFace",query_type="search"} 4
milvus_proxy_collection_sq_latency_sum{collection_name="isolatedFace",query_type="search"} 10
''')
        prefix, labels, unit = custom["face_search"]
        result = histogram_delta(before, after, prefix, labels, unit)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["mean"], 4)


if __name__ == '__main__':
    unittest.main()
