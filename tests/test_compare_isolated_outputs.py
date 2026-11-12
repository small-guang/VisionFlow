"""隔离A/B比对使用fake Kafka与临时文件；不连接任何业务服务。"""
import asyncio
from collections import namedtuple
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from performance_tools.compare_isolated_outputs import (
    ReadOnlyMetadataConsumer, RunInput, capture_outputs, compare_outputs, identity_summary, load_capture,
    load_run, read_kafka, save_capture, validate_pair,
)
from performance_tools.isolated_replay import isolation_names
from performance_tools.kafka_replay import make_replay_uid

TP = namedtuple("TP", "topic partition")
RUN_IDS = {"left": "11111111111141118111111111111111", "right": "22222222222242228222222222222222"}


def run_input(side):
    names = isolation_names(side)
    uid = make_replay_uid(RUN_IDS[side], 0, trace_completion=True)
    manifest = {"schema": 1, "namespace": side, "names": names, "input": {"sha256": "input-same"}}
    summary = {"run_id": RUN_IDS[side], "trace_completion": True, "topic": names["input_topic"],
               "acked_messages": 1, "failed_messages": 0, "event_time": "preserve", "keep_pano": True}
    proof = {"warm_rows": {"face": 20, "sbs": 1}, "warm_snapshot_sha256": "same-warm"}
    sent = {0: {"sequence": 0, "uId": uid, "topic": names["input_topic"], "status": "acked",
                "source": {"topic": "history", "partition": 0, "offset": 7}}}
    return RunInput(manifest, summary, proof, sent, {uid: 0}, side + "-manifest")


def envelope(run, kind, payload):
    return {"kind": kind, "topic": run.manifest["names"]["output_topics"][kind],
            "partition": 0, "offset": 0, "payload": payload}


def outputs(run, *, pano=False, person=False):
    kind = "person" if person else "face"
    rect = {"x": 10, "y": 20, "w": 30, "h": 40}
    common = {"keyframeId": next(iter(run.uid_to_sequence)), "isPanoKeyframe": pano,
              "aiId": run.summary["run_id"] + "-random-ai", "pictureVectorId": id(run),
              "createdAt": id(run), "timestamp": id(run), "devSn": "private-device"}
    detection = {**common, "faceOrientation": "front", "faceQualityScore": 5.0,
                 "faceQualityLevel": 2, "confidence": 0.9, "status": 1,
                 "faceKeypoint": {"points": [[1.5, 2.5]]}, "crop_url": run.manifest["names"]["storage_prefix"] + "/random.jpg",
                 "peopleRect" if person else "faceRect": rect, "info": "private-human-diagnostic"}
    recognition = {**common, "aiFaceId": "private-face", "vectorHitFaceId": "private-face",
                   "faceOrientation": "front", "faceQualityScore": 5.0, "faceQualityLevel": 2,
                   "faceStatus": 2, "score": 0.88, "faceUrl": "private-random-crop"}
    if person:
        recognition.update(peopleRect=rect, reid="private-reid", vectorHitReid="private-reid",
                           canonicalFaceId="private-face", reidStatus=2)
    vector = {**common, "targetType": kind, "vectorDim": 512, "modelName": "same-model",
              "vectorDbName": run.manifest["names"]["collections"]["tbps" if person else "face"],
              "vectorId": id(run), "keyframeUrl": "private-random-vector-image"}
    return [envelope(run, kind + "_detection", detection), envelope(run, kind + "_recognition", recognition),
            envelope(run, "vector_record", vector)]


class OutputComparisonTests(unittest.TestCase):
    def setUp(self):
        self.left, self.right = run_input("left"), run_input("right")

    def test_random_transport_fields_and_namespace_do_not_change_business_result_or_leak(self):
        result = compare_outputs(self.left, self.right,
                                 {"left": outputs(self.left), "right": outputs(self.right)})
        face = result["targets"]["face"]
        self.assertEqual(face["exact_rect_matches"], 1)
        self.assertEqual(face["detection_changed"], 0)
        self.assertEqual(face["recognition_changed"], 0)
        self.assertEqual(face["recognition_compared"], 1)
        self.assertEqual(result["vectors"]["changed_metadata_or_count"], 0)
        rendered = json.dumps(result)
        for private in ("private-face", "private-reid", "private-device", "private-human-diagnostic",
                        "private-random", "random-ai", RUN_IDS["left"], RUN_IDS["right"]):
            self.assertNotIn(private, rendered)

    def test_face_recognition_without_rect_joins_ai_id_and_detects_quality_and_identity_change(self):
        a, b = outputs(self.left), outputs(self.right)
        a[1]["payload"]["aiFaceId"] = "0"
        b[0]["payload"]["faceQualityScore"] = 5.5
        b[1]["payload"]["faceStatus"] = 1
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})["targets"]["face"]
        self.assertEqual(result["detection_field_mismatches"], {"faceQualityScore": 1})
        self.assertEqual(result["recognition_field_mismatches"], {"faceStatus": 1})
        self.assertEqual(result["identities"]["aiFaceId"]["unassigned_to_assigned"], 1)

    def test_pano_and_box_changes_are_not_guessed_as_matches(self):
        a, b = outputs(self.left), outputs(self.right, pano=True)
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})["targets"]["face"]
        self.assertEqual(result["exact_rect_matches"], 0)
        self.assertEqual(result["left_unmatched_rects"], 1)
        self.assertEqual(result["right_unmatched_rects"], 1)
        b = outputs(self.right)
        b[0]["payload"]["faceRect"]["x"] += 1
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})["targets"]["face"]
        self.assertEqual(result["exact_rect_matches"], 0)

    def test_orphan_or_duplicate_ai_ids_are_reported_without_silent_overwrite(self):
        a, b = outputs(self.left), outputs(self.right)
        b[1]["payload"]["aiId"] = "unrelated-ai"
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})
        self.assertEqual(result["runs"]["right"]["diagnostics"]["face_unlinked_recognition"], 1)
        self.assertEqual(result["targets"]["face"]["recognition_presence"], {"left_only": 1})
        b = outputs(self.right)
        b.append(deepcopy(b[0]))
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})
        self.assertEqual(result["targets"]["face"]["ambiguous_matched_rects"], 1)
        self.assertEqual(result["targets"]["face"]["right_detection_records"], 2)

    def test_person_recognition_rect_must_agree_with_linked_detection(self):
        a, b = outputs(self.left, person=True), outputs(self.right, person=True)
        b[1]["payload"]["peopleRect"] = {"x": 0, "y": 0, "w": 10, "h": 10}
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})
        self.assertEqual(result["runs"]["right"]["diagnostics"]["person_recognition_rect_disagrees"], 1)
        self.assertEqual(result["targets"]["person"]["recognition_presence"], {"left_only": 1})

    def test_record_from_other_run_and_invalid_pano_are_excluded(self):
        a, b = outputs(self.left), outputs(self.right)
        extra = deepcopy(b[0]); extra["payload"]["keyframeId"] = "another-run"
        b.append(extra)
        extra = deepcopy(b[0]); extra["payload"]["isPanoKeyframe"] = "false"
        b.append(extra)
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})
        self.assertEqual(result["runs"]["right"]["diagnostics"], {"other_run_records": 1, "invalid_pano_flag": 1})

    def test_malformed_correlation_fields_are_diagnosed_without_crashing(self):
        a, b = outputs(self.left), outputs(self.right)
        b[1]["payload"]["aiId"] = ["invalid"]
        b[2]["payload"]["targetType"] = {"invalid": True}
        extra = deepcopy(b[0]); extra["payload"]["keyframeId"] = ["invalid"]
        b.append(extra)
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})
        self.assertEqual(result["runs"]["right"]["diagnostics"], {
            "invalid_keyframe_id": 1, "face_invalid_recognition_ai_id": 1, "unknown_vector_target": 1})

    def test_missing_identity_field_is_not_silently_treated_as_zero(self):
        a, b = outputs(self.left), outputs(self.right)
        a[1]["payload"]["aiFaceId"] = "0"
        del b[1]["payload"]["aiFaceId"]
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})
        identity = result["targets"]["face"]["identities"]["aiFaceId"]
        self.assertEqual(identity["right_missing"], 1)
        self.assertEqual(identity["not_comparable_missing"], 1)
        self.assertNotIn("both_unassigned", identity)

    def test_numeric_tolerance_does_not_hide_integer_quality_level_change(self):
        a, b = outputs(self.left), outputs(self.right)
        b[0]["payload"]["confidence"] += 1e-7
        b[0]["payload"]["faceQualityLevel"] = 1
        result = compare_outputs(self.left, self.right, {"left": a, "right": b})["targets"]["face"]
        self.assertEqual(result["detection_field_mismatches"], {"faceQualityLevel": 1})

    def test_identity_partition_summary_preserves_splits_merges_without_values(self):
        result = identity_summary([("a", "x"), ("a", "y"), ("b", "y"), ("0", "z"), ("q", "0")])
        self.assertEqual(result["left_identities_split"], 1)
        self.assertEqual(result["right_identities_merged"], 1)
        self.assertEqual(result["unassigned_to_assigned"], 1)
        self.assertEqual(result["assigned_to_unassigned"], 1)
        self.assertEqual(result["different_nonzero_value"], 3)

    def test_different_sources_or_warm_start_are_rejected(self):
        self.right.proof["warm_snapshot_sha256"] = "changed"
        with self.assertRaises(ValueError): validate_pair(self.left, self.right)
        self.right = run_input("right")
        self.right.sent[0]["source"]["offset"] = 8
        with self.assertRaises(ValueError): validate_pair(self.left, self.right)


class FakeConsumer:
    def __init__(self, records, ends=None):
        self.records = records
        self.ends = ends or {tp: max((row.offset for row in rows), default=-1) + 1 for tp, rows in records.items()}
        self.positions = {}
        self.calls = []
    async def beginning_offsets(self, partitions):
        self.calls.append("begin")
        return {tp: 0 for tp in partitions}
    async def end_offsets(self, partitions):
        self.calls.append("end")
        return {tp: self.ends[tp] for tp in partitions}
    def assign(self, partitions): self.calls.append("assign")
    def seek(self, tp, offset): self.positions[tp] = offset
    def pause(self, tp): self.calls.append("pause")
    async def position(self, tp): return self.positions[tp]
    async def getmany(self, *partitions, **kwargs):
        self.calls.append("getmany")
        result = {}
        for tp in partitions:
            batch = [row for row in self.records.get(tp, []) if row.offset >= self.positions[tp]][:kwargs["max_records"]]
            if batch:
                result[tp] = batch
                self.positions[tp] = batch[-1].offset + 1
        return result


class BoundedOutputReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_freezes_end_filters_foreign_run_and_retains_complete_matching_payload(self):
        run = run_input("left")
        topic = run.manifest["names"]["output_topics"]["face_detection"]
        tp = TP(topic, 0)
        good = outputs(run)[0]["payload"]
        foreign = {**good, "keyframeId": "foreign-run"}
        rows = [SimpleNamespace(offset=i, value=json.dumps(payload).encode())
                for i, payload in enumerate((foreign, good, good))]
        consumer = FakeConsumer({tp: rows}, {tp: 2})
        records, capture = await capture_outputs(consumer, [tp], {topic: ("left", "face_detection")}, {"left": run})
        self.assertEqual(len(records["left"]), 1)
        self.assertEqual(records["left"][0]["payload"], good)
        self.assertEqual(records["left"][0]["offset"], 1)
        self.assertTrue(capture["read_complete"])
        self.assertEqual(capture["counts"], {"scanned": 2, "other_run_records": 1})
        self.assertEqual(consumer.calls[:4], ["begin", "end", "assign", "getmany"])
        self.assertFalse(capture["committed_offsets"])

    async def test_payload_limit_marks_read_incomplete_instead_of_silent_success(self):
        run = run_input("left")
        topic = run.manifest["names"]["output_topics"]["face_detection"]
        tp = TP(topic, 0)
        payload = json.dumps(outputs(run)[0]["payload"]).encode()
        consumer = FakeConsumer({tp: [SimpleNamespace(offset=0, value=payload)]})
        records, capture = await capture_outputs(consumer, [tp], {topic: ("left", "face_detection")},
                                                  {"left": run}, max_payload_bytes=1)
        self.assertEqual(records["left"], [])
        self.assertFalse(capture["read_complete"])
        self.assertEqual(capture["stop_reason"], "max_payload_bytes")

    async def test_invalid_json_and_keyframe_id_count_without_claiming_they_are_other_runs(self):
        run = run_input("left")
        topic = run.manifest["names"]["output_topics"]["face_detection"]
        tp = TP(topic, 0)
        values = (b"not json", b'{"keyframeId": []}')
        consumer = FakeConsumer({tp: [SimpleNamespace(offset=index, value=value) for index, value in enumerate(values)]})
        records, capture = await capture_outputs(consumer, [tp], {topic: ("left", "face_detection")}, {"left": run})
        self.assertEqual(records["left"], [])
        self.assertEqual(capture["counts"], {"scanned": 2, "invalid_json": 1, "invalid_keyframe_id": 1})

    async def test_live_entry_disables_group_commits_and_topic_creation_and_stops_consumer(self):
        from aiokafka import AIOKafkaConsumer

        run = run_input("left")
        topics = list(run.manifest["names"]["output_topics"].values())
        consumer = FakeConsumer({TP(topic, 0): [] for topic in topics})
        consumer.start = unittest.mock.AsyncMock()
        consumer.stop = unittest.mock.AsyncMock()
        consumer.topics = unittest.mock.AsyncMock(return_value=set(topics))
        consumer.partitions_for_topic = lambda topic: {0}
        consumer._client = SimpleNamespace(set_topics=unittest.mock.Mock(),
                                           _metadata_update=unittest.mock.AsyncMock())
        # 使用安装版本的真实构造签名，不能让宽松Mock掩盖不支持的SDK参数。
        constructor = unittest.mock.create_autospec(AIOKafkaConsumer, return_value=consumer)
        with patch.dict("sys.modules", {
            "aiokafka": SimpleNamespace(AIOKafkaConsumer=constructor),
            "aiokafka.structs": SimpleNamespace(TopicPartition=TP),
            "core.config": SimpleNamespace(settings=SimpleNamespace(kafka_client_config={"bootstrap_servers": "fake"})),
        }):
            _records, capture = await read_kafka({"left": run}, timeout=1, max_records=100, max_payload_bytes=1024)
        self.assertTrue(capture["read_complete"])
        self.assertIsNone(constructor.call_args.kwargs["group_id"])
        self.assertFalse(constructor.call_args.kwargs["enable_auto_commit"])
        self.assertNotIn("allow_auto_create_topics", constructor.call_args.kwargs)
        self.assertEqual(constructor.call_args.kwargs["auto_offset_reset"], "none")
        consumer._client.set_topics.assert_called_once_with([])
        consumer.stop.assert_awaited_once()

    async def test_real_consumer_assignment_preserves_partitions_and_requests_only_all_metadata(self):
        from aiokafka import AIOKafkaConsumer
        from aiokafka.structs import TopicPartition

        # 仅构造，不start；网络连接由fake替代，但使用真实SDK的assign和协议编码。
        native = AIOKafkaConsumer(bootstrap_servers="127.0.0.1:1", group_id=None,
                                  enable_auto_commit=False, auto_offset_reset="none")
        adapter = ReadOnlyMetadataConsumer(native)
        partition = TopicPartition("must-not-create-a-topic", 0)
        connection = SimpleNamespace(send=unittest.mock.AsyncMock(return_value=SimpleNamespace(brokers=[])))
        native._client._api_version = (2, 0, 0)
        try:
            with patch.object(native._client.cluster, "brokers", return_value=[SimpleNamespace(nodeId=1)]), \
                    patch.object(native._client, "_get_conn", unittest.mock.AsyncMock(return_value=connection)):
                adapter.assign([partition])
                self.assertEqual(native.assignment(), {partition})
                self.assertEqual(native._client._topics, set())
                await native._client._metadata_update(native._client.cluster, native._client._topics)
                # Fetcher 在查offset遇到未知leader时会再次add_topic，保护必须覆盖此分支。
                native._client.add_topic(partition.topic)
                self.assertEqual(native._client._topics, {partition.topic})
                await native._client._metadata_update(native._client.cluster, native._client._topics)
            self.assertEqual(connection.send.call_count, 2)
            for call in connection.send.call_args_list:
                request = call.args[0]
                self.assertEqual(request.API_KEY, 3)
                self.assertEqual(request.API_VERSION, 1)
                self.assertIsNone(request.topics)
        finally:
            await native.stop()


class ComparisonFileTests(unittest.TestCase):
    def test_run_loader_rejects_foreign_uid_and_ack_mismatch(self):
        run = run_input("left")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(run.manifest))
            proof = {**run.proof, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            (root / "prepared.json").write_text(json.dumps(proof))
            (root / "summary.json").write_text(json.dumps(run.summary))
            sent = root / "sent.jsonl"
            sent.write_text(json.dumps(run.sent[0]) + "\n")
            self.assertEqual(load_run(manifest, root).uid_to_sequence, run.uid_to_sequence)
            broken = {**run.sent[0], "uId": "foreign-run"}
            sent.write_text(json.dumps(broken) + "\n")
            with self.assertRaises(ValueError): load_run(manifest, root)
            sent.write_text(json.dumps({**run.sent[0], "status": "failed"}) + "\n")
            with self.assertRaises(ValueError): load_run(manifest, root)

    def test_full_capture_is_private_confined_no_overwrite_and_bound_to_run(self):
        runs = {side: run_input(side) for side in ("left", "right")}
        selected = {side: outputs(run) for side, run in runs.items()}
        with tempfile.TemporaryDirectory() as directory, patch("performance_tools.compare_isolated_outputs.PROJECT_ROOT", Path(directory)):
            root = Path(directory)
            with self.assertRaises(ValueError): save_capture(root / "tracked.json", runs, selected, {})
            path = root / "exports" / "private.json"
            save_capture(path, runs, selected, {"read_complete": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(load_capture(path, runs)[0], selected)
            with self.assertRaises(FileExistsError): save_capture(path, runs, selected, {})
            result_path = root / "performance_tools" / "results" / "private.json"
            save_capture(result_path, runs, selected, {"read_complete": True})
            self.assertEqual(result_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(load_capture(result_path, runs)[0], selected)
            with self.assertRaises(FileExistsError): save_capture(result_path, runs, selected, {})
            runs["left"].manifest_sha256 = "changed"
            with self.assertRaises(ValueError): load_capture(path, runs)


if __name__ == "__main__":
    unittest.main()
