from __future__ import annotations

import argparse
import asyncio
from collections import namedtuple
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import AsyncMock, Mock, patch

from performance_tools.kafka_replay import (
    OutputObserver, build_parser, capture_bounded, positive_float,
    read_performance_summaries, replay_payload, replay, snapshot, make_replay_uid,
)


class ReplayLogicTests(unittest.TestCase):
    def test_replay_keeps_original_message_and_image_identity(self):
        original = {"uId": "old", "sn": "camera", "timestamp": 123,
                    "keyFrameUrl": "camera/picture/main.jpg",
                    "panoKeyframeUrl": "camera/picture/pano.jpg",
                    "extra": {"items": [1]}}
        result = replay_payload(original, uid="new", event_time="preserve",
                                keep_pano=False, now_ms=456)
        self.assertEqual(result["uId"], "new")
        self.assertEqual(result["timestamp"], 123)
        self.assertEqual(result["sn"], original["sn"])
        self.assertEqual(result["keyFrameUrl"], original["keyFrameUrl"])
        self.assertNotIn("panoKeyframeUrl", result)
        result["extra"]["items"].append(2)
        self.assertEqual(original["extra"]["items"], [1])
        self.assertIn("panoKeyframeUrl", original)

    def test_now_event_time_has_consistent_aliases_and_optional_pano(self):
        result = replay_payload({"time": 123, "ts": 123, "panoKeyframeUrl": "p.jpg"},
                                uid="new", event_time="now", keep_pano=True, now_ms=456)
        self.assertEqual([result[k] for k in ("timestamp", "time", "ts")], [456] * 3)
        self.assertEqual(result["panoKeyframeUrl"], "p.jpg")

    def test_observer_ignores_other_runs_and_deduplicates_picture_outputs(self):
        observer = OutputObserver(None, "vectors", StringIO())
        observer.sent_at["ours"] = 100
        self.assertFalse(observer.record("vectors", {"keyframeId": "other", "targetType": "picture"}, 101))
        for pano in (False, False, True):
            self.assertTrue(observer.record("vectors", {"keyframeId": "ours", "targetType": "picture",
                                                         "isPanoKeyframe": pano}, 101))
        observer.record("vectors", {"keyframeId": "ours", "targetType": "person"}, 102)
        summary = observer.summary()
        self.assertEqual(summary["picture_vectors_persisted_and_emitted"], 2)
        self.assertEqual(summary["vector_records_by_target"], {"picture": 3, "person": 1})
        self.assertEqual(summary["matched_uids_with_output"], 1)
        self.assertEqual(summary["first_output_latency_ms"]["avg"], 1000)
        self.assertFalse(summary["full_pipeline_completion_verified"])

    def test_summary_reader_filters_other_logs_and_prior_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ALog.log"
            path.write_text('ALog performance summary: {"prior":true}\n', encoding="utf-8")
            offset = path.stat().st_size
            with path.open("a", encoding="utf-8") as stream:
                stream.write('ordinary log\nINFO | ALog performance summary: {"schema_version":1}\n')
                stream.write('INFO | ALog performance summary: partial\n')
            self.assertEqual(read_performance_summaries(path, offset), [{"schema_version": 1}])

    def test_default_subcommand_is_read_only_and_replay_rate_is_conservative(self):
        sample = build_parser().parse_args(["snapshot"])
        self.assertEqual(sample.command, "snapshot")
        self.assertEqual(sample.topic, "PtzPersonDetSync")
        replay = build_parser().parse_args(["replay", "--input", "sample.jsonl"])
        self.assertEqual(replay.qps, 1)
        self.assertIsNone(replay.count)  # 无duration时执行层使用10条。
        self.assertFalse(replay.keep_pano)
        self.assertEqual(replay.event_time, "preserve")

    def test_nonfinite_or_nonpositive_rates_are_rejected(self):
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_float(value)


class SnapshotBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_freezes_upper_bound_and_skips_invalid_messages(self):
        class Partition(tuple):
            @property
            def partition(self):
                return self[1]

        tp = Partition(("PtzPersonDetSync", 0))

        class Consumer:
            def __init__(self):
                self.assigned = []
                self.seeks = []
                self.paused = []

            async def beginning_offsets(self, partitions):
                return {tp: 2}

            async def end_offsets(self, partitions):
                return {tp: 5}

            def assign(self, partitions):
                self.assigned = partitions

            def seek(self, partition, offset):
                self.seeks.append((partition, offset))

            def pause(self, partition):
                self.paused.append(partition)

            async def position(self, partition):
                return 7

            async def getmany(self, *partitions, **kwargs):
                def record(offset, payload):
                    return SimpleNamespace(topic="PtzPersonDetSync", partition=0, offset=offset,
                                           timestamp=100, value=json.dumps(payload).encode())
                return {tp: [record(2, {"uId": "a"}), record(3, []),
                             record(4, {"uId": "b"}), record(5, {"uId": "new"})]}

        def validator(payload):
            if not isinstance(payload, dict) or not payload.get("uId"):
                raise ValueError("invalid")

        consumer = Consumer()
        selected, metadata = await capture_bounded(
            consumer, [tp], count=10, lookback=100, max_scan=10, timeout=5, validator=validator,
        )
        self.assertEqual([item["payload"]["uId"] for item in selected], ["a", "b"])
        self.assertEqual(metadata["scanned"], 3)
        self.assertEqual(metadata["invalid"], 1)
        self.assertEqual(metadata["partition_bounds"], {"0": {"start": 2, "end_exclusive": 5}})
        self.assertEqual(consumer.seeks, [(tp, 2)])
        self.assertEqual(consumer.paused, [tp])


class KafkaClientApiTests(unittest.IsolatedAsyncioTestCase):
    def fake_clients(self):
        topics = {"PtzPersonDetSync", "faces", "face_rec", "persons", "person_rec", "vectors"}
        consumer = SimpleNamespace(
            start=AsyncMock(), stop=AsyncMock(), topics=AsyncMock(return_value=topics),
            # aiokafka 0.10 的这个 API 同步返回 set，不能 await。
            partitions_for_topic=Mock(return_value={0}), assign=Mock(), seek=Mock(),
            end_offsets=AsyncMock(side_effect=lambda partitions: {tp: 10 for tp in partitions}),
            getmany=AsyncMock(),
        )

        async def poll(**kwargs):
            await __import__("asyncio").sleep(.01)
            return {}

        consumer.getmany.side_effect = poll
        producer = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(),
                                   send_and_wait=AsyncMock(return_value=SimpleNamespace(partition=0, offset=10)))
        kafka_module = SimpleNamespace(AIOKafkaConsumer=Mock(return_value=consumer),
                                       AIOKafkaProducer=Mock(return_value=producer))
        settings = SimpleNamespace(
            kafka_client_config={}, KAFKA_FACE_DETECTION_RECORD_TOPIC="faces",
            KAFKA_FACE_RECOGNITION_RECORD_TOPIC="face_rec",
            KAFKA_PERSON_DETECTION_RECORD_TOPIC="persons",
            KAFKA_PERSON_RECOGNITION_RECORD_TOPIC="person_rec", KAFKA_VECTOR_RECORD_TOPIC="vectors",
        )
        modules = {"aiokafka": kafka_module,
                   "aiokafka.structs": SimpleNamespace(TopicPartition=namedtuple("TopicPartition", "topic partition")),
                   "core.config": SimpleNamespace(settings=settings)}
        return consumer, producer, kafka_module, modules

    async def test_snapshot_uses_synchronous_partition_metadata_api(self):
        consumer, _, kafka_module, modules = self.fake_clients()
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["snapshot", "--count", "1", "--output", str(Path(directory) / "input.jsonl")])
            captured = ([{"source": {"offset": 2}, "payload": {"uId": "old"}}], {"selected": 1})
            with patch.dict(sys.modules, modules), \
                 patch("performance_tools.kafka_replay.capture_bounded", AsyncMock(return_value=captured)), \
                 redirect_stdout(StringIO()):
                await snapshot(args)
            consumer.partitions_for_topic.assert_called_once_with("PtzPersonDetSync")
            consumer.stop.assert_awaited_once()
            self.assertIsNone(kafka_module.AIOKafkaConsumer.call_args.kwargs["group_id"])
            self.assertFalse(kafka_module.AIOKafkaConsumer.call_args.kwargs["enable_auto_commit"])

    async def test_replay_observer_initialization_uses_same_synchronous_api(self):
        consumer, producer, _, modules = self.fake_clients()
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "replay", "--input", "unused.jsonl", "--count", "1", "--qps", "1",
                "--wait-seconds", ".001", "--output-dir", directory,
            ])
            with patch.dict(sys.modules, modules), \
                 patch("performance_tools.kafka_replay.load_snapshot", return_value=[{"payload": {"uId": "old"}}]), \
                 redirect_stdout(StringIO()):
                await replay(args)
            self.assertEqual(consumer.partitions_for_topic.call_count, 5)
            producer.send_and_wait.assert_awaited_once()
            producer.stop.assert_awaited_once()
            consumer.stop.assert_awaited_once()
            summaries = list(Path(directory).glob("*/summary.json"))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(json.loads(summaries[0].read_text())["acked_messages"], 1)

    async def test_replay_absolute_schedule_does_not_accumulate_wakeup_and_ack_delay(self):
        _, producer, _, modules = self.fake_clients()
        clock = SimpleNamespace(now=0.0)
        sent_at = []
        real_sleep = asyncio.sleep

        async def delayed_ack(*args, **kwargs):
            sent_at.append(clock.now)
            clock.now += .002
            return SimpleNamespace(partition=0, offset=len(sent_at))

        async def late_sleep(delay):
            clock.now += delay + .003
            await real_sleep(0)

        producer.send_and_wait.side_effect = delayed_ack
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "replay", "--input", "unused.jsonl", "--count", "5", "--qps", "50",
                "--max-inflight", "1", "--no-observe", "--output-dir", directory,
            ])
            with patch.dict(sys.modules, modules), \
                 patch("performance_tools.kafka_replay.load_snapshot", return_value=[{"payload": {"uId": "old"}}]), \
                 patch("performance_tools.kafka_replay.time", SimpleNamespace(
                     monotonic=lambda: clock.now, time=lambda: 1000 + clock.now)), \
                 patch("performance_tools.kafka_replay.asyncio.sleep", late_sleep), \
                 redirect_stdout(StringIO()):
                await replay(args)
        self.assertEqual(len(sent_at), 5)
        for actual, expected in zip(sent_at, (0, .023, .043, .063, .083)):
            self.assertAlmostEqual(actual, expected)
        producer.stop.assert_awaited_once()


class ReplayTraceIdTests(unittest.TestCase):
    def test_uuid5_remains_the_default_and_trace_mode_is_explicit(self):
        run_id = uuid.uuid4().hex
        expected = str(uuid.uuid5(uuid.UUID(hex=run_id), "12"))
        self.assertEqual(make_replay_uid(run_id, 12), expected)
        traced = make_replay_uid(run_id, 12, trace_completion=True)
        self.assertEqual(traced, f"alogtest-v1:{run_id}:12")
        self.assertLessEqual(len(make_replay_uid(run_id, 65535, trace_completion=True)), 50)
        args = build_parser().parse_args(["replay", "--input", "unused.jsonl"])
        self.assertFalse(args.trace_completion)
        for sequence in (-1, 65536):
            with self.assertRaises(ValueError):
                make_replay_uid(run_id, sequence, trace_completion=True)


if __name__ == "__main__":
    unittest.main()
