"""从现有 Kafka 取有限快照，再限速回放到 ALog；不加入业务 consumer group。"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
from urllib.request import urlopen
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_RESULTS = PROJECT_ROOT / "performance_tools" / "results"
PERFORMANCE_MARKER = "ALog performance summary: "


def make_replay_uid(run_id: str, sequence: int, *, trace_completion: bool = False) -> str:
    if not trace_completion:
        return str(uuid.uuid5(uuid.UUID(hex=run_id), str(sequence)))
    from core.performance import parse_replay_uid

    uid = f"alogtest-v1:{run_id}:{sequence}"
    if parse_replay_uid(uid) is None:
        raise ValueError("完成追踪需要 UUID4 run_id，sequence 范围为 0..65535")
    return uid


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("必须为有限正数")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def encode_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def validate_payload(payload: Any) -> Any:
    # 延迟导入：--help 与纯逻辑单测无需加载连接配置，不导入 main / db。
    from utils.keyframe.detection_helpers import parse_keyframe_context

    if not isinstance(payload, dict):
        raise ValueError("消息必须是 JSON object")
    return parse_keyframe_context(payload)


def load_snapshot(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("payload"), dict):
                raise ValueError(f"快照第 {line_number} 行缺少 payload object")
            validate_payload(item["payload"])
            result.append(item)
    if not result:
        raise ValueError("快照没有有效消息")
    return result


def replay_payload(
    payload: dict[str, Any], *, uid: str, event_time: str,
    keep_pano: bool, now_ms: int,
) -> dict[str, Any]:
    result = deepcopy(payload)
    result["uId"] = uid
    if not keep_pano:
        result.pop("panoKeyframeUrl", None)
    if event_time == "now":
        # timestamp 是生产解析器的首选字段；覆盖兼容字段以免后续读取产生歧义。
        result["timestamp"] = now_ms
        for field in ("time", "ts"):
            if field in result:
                result[field] = now_ms
    return result


async def capture_bounded(
    consumer: Any, partitions: list[Any], *, count: int,
    lookback: int, max_scan: int, timeout: float,
    validator: Any = validate_payload,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """冻结 end offsets 后只读其前数据，绝不跟随本轮新写入消息。"""
    beginnings = await consumer.beginning_offsets(partitions)
    ends = await consumer.end_offsets(partitions)
    starts = {tp: max(beginnings[tp], ends[tp] - lookback) for tp in partitions}
    consumer.assign(partitions)
    for tp in partitions:
        consumer.seek(tp, starts[tp])
    active = {tp for tp in partitions if starts[tp] < ends[tp]}
    selected: list[dict[str, Any]] = []
    scanned = invalid = 0
    deadline = time.monotonic() + timeout
    while active and len(selected) < count and scanned < max_scan:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        records = await consumer.getmany(
            *active, timeout_ms=min(1000, max(1, int(remaining * 1000))),
            max_records=min(500, max_scan - scanned),
        )
        for tp, batch in records.items():
            if tp not in active:
                continue
            for record in batch:
                if record.offset >= ends[tp]:
                    break
                scanned += 1
                try:
                    payload = json.loads(record.value)
                    validator(payload)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, AttributeError):
                    invalid += 1
                else:
                    selected.append({
                        "source": {"topic": record.topic, "partition": record.partition,
                                   "offset": record.offset, "kafka_timestamp_ms": record.timestamp},
                        "payload": payload,
                    })
                if len(selected) >= count or scanned >= max_scan:
                    break
            if len(selected) >= count or scanned >= max_scan:
                break
        for tp in tuple(active):
            if await consumer.position(tp) >= ends[tp]:
                active.remove(tp)
                consumer.pause(tp)
    bounds = {str(tp.partition): {"start": starts[tp], "end_exclusive": ends[tp]}
              for tp in partitions}
    return selected, {"partition_bounds": bounds, "scanned": scanned, "invalid": invalid,
                      "selected": len(selected), "timed_out": time.monotonic() >= deadline}


async def snapshot(args: argparse.Namespace) -> None:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.structs import TopicPartition
    from core.config import settings

    consumer = AIOKafkaConsumer(
        group_id=None, enable_auto_commit=False, auto_offset_reset="none",
        **settings.kafka_client_config,
    )
    try:
        await consumer.start()
        if args.topic not in await consumer.topics():
            raise ValueError(f"Topic 不存在: {args.topic}")
        partition_ids = consumer.partitions_for_topic(args.topic)
        if not partition_ids:
            raise ValueError(f"Topic 不存在或没有分区: {args.topic}")
        partitions = [TopicPartition(args.topic, i) for i in sorted(partition_ids)]
        records, metadata = await capture_bounded(
            consumer, partitions, count=args.count, lookback=args.lookback,
            max_scan=args.max_scan, timeout=args.timeout,
        )
    finally:
        await consumer.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 不覆盖已有采样，避免误把一次新采样当成旧回放的输入。
    with args.output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(encode_json(record).decode("utf-8") + "\n")
    metadata.update({"topic": args.topic, "created_at": datetime.now(timezone.utc).isoformat(),
                     "snapshot": str(args.output), "group_id": None, "committed_offsets": False,
                     "image_access_verified": False})
    write_json(args.output.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not records:
        raise ValueError("没有采集到有效关键帧；可增大 --lookback / --max-scan")


class OutputObserver:
    """只关联本轮 ID；picture 向量输出不能代表该图片所有阶段已完成。"""

    def __init__(self, consumer: Any, vector_topic: str, stream: Any):
        self.consumer = consumer
        self.vector_topic = vector_topic
        self.stream = stream
        self.sent_at: dict[str, float] = {}
        self.records_by_topic: Counter[str] = Counter()
        self.vector_records: Counter[str] = Counter()
        self.picture_images: set[tuple[str, bool]] = set()
        self.picture_latency_ms: list[float] = []
        self.first_output: set[str] = set()
        self.first_output_latency_ms: list[float] = []

    def record(self, topic: str, payload: Any, received_at: float) -> bool:
        if not isinstance(payload, dict):
            return False
        uid = str(payload.get("keyframeId", ""))
        if uid not in self.sent_at:
            return False
        latency_ms = max(0.0, (received_at - self.sent_at[uid]) * 1000)
        self.records_by_topic[topic] += 1
        if uid not in self.first_output:
            self.first_output.add(uid)
            self.first_output_latency_ms.append(latency_ms)
        if topic == self.vector_topic:
            target = str(payload.get("targetType", "unknown"))
            self.vector_records[target] += 1
            image_id = (uid, bool(payload.get("isPanoKeyframe", False)))
            if target == "picture" and image_id not in self.picture_images:
                self.picture_images.add(image_id)
                self.picture_latency_ms.append(latency_ms)
        # 向量内容体积较大，仅保存定位和延迟，结果中不复制 embedding。
        record = {"topic": topic, "uId": uid, "received_at": received_at,
                  "latency_ms": latency_ms, "isPanoKeyframe": bool(payload.get("isPanoKeyframe", False)),
                  "targetType": payload.get("targetType"), "pictureVectorId": payload.get("pictureVectorId"),
                  "vectorDbName": payload.get("vectorDbName")}
        self.stream.write(encode_json(record).decode("utf-8") + "\n")
        self.stream.flush()
        return True

    async def run(self) -> None:
        while True:
            records = await self.consumer.getmany(timeout_ms=500, max_records=1000)
            for tp, batch in records.items():
                for record in batch:
                    try:
                        payload = json.loads(record.value)
                    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                        continue
                    self.record(tp.topic, payload, time.time())

    def summary(self) -> dict[str, Any]:
        return {"records_by_topic": dict(self.records_by_topic),
                "vector_records_by_target": dict(self.vector_records),
                "matched_uids_with_output": len(self.first_output),
                "picture_vectors_persisted_and_emitted": len(self.picture_images),
                "first_output_latency_ms": describe(self.first_output_latency_ms),
                "picture_persisted_output_latency_ms": describe(self.picture_latency_ms),
                "full_pipeline_completion_verified": False}


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {"count": len(values), "avg": sum(values) / len(values),
            "p50": ordered[(len(ordered) - 1) // 2],
            "p95": ordered[max(0, math.ceil(len(ordered) * .95) - 1)], "max": ordered[-1]}


def read_performance_summaries(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(offset if path.stat().st_size >= offset else 0)
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    result = []
    for line in lines:
        if PERFORMANCE_MARKER not in line:
            continue
        try:
            result.append(json.loads(line.split(PERFORMANCE_MARKER, 1)[1]))
        except json.JSONDecodeError:
            continue
    return result


def fetch_metrics(url: str) -> dict[str, float]:
    with urlopen(url, timeout=3) as response:
        text = response.read(8 * 1024 * 1024).decode("utf-8")
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            values[name] = value
    return values


async def metric_snapshot(entries: list[str]) -> dict[str, Any]:
    result = {}
    for entry in entries:
        name, separator, url = entry.partition("=")
        if not separator or not name or not url.startswith(("http://", "https://")):
            raise ValueError("--metrics 格式为 name=http://host:port/metrics")
        try:
            result[name] = await asyncio.to_thread(fetch_metrics, url)
        except Exception as error:
            # 只记录类型，不把可能含凭据的 URL / 服务错误复制到结果。
            result[name] = {"error": type(error).__name__}
    return result


async def replay(args: argparse.Namespace) -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.structs import TopicPartition
    from core.config import settings

    if args.trace_completion and args.count is not None and args.count > 65536:
        raise ValueError("--trace-completion 每轮最多 65536 条消息")
    records = load_snapshot(args.input)
    run_id = uuid.uuid4().hex
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    log_offset = 0
    metrics_before = await metric_snapshot(args.metrics)
    consumer = None
    observer = None
    observer_task = None
    pending: set[asyncio.Task] = set()
    producer = AIOKafkaProducer(acks="all", **settings.kafka_client_config)
    sent_records: list[dict[str, Any]] = []
    started_at = time.time()
    started_clock = time.monotonic()
    interrupted = False
    count_limit = args.count if args.count is not None else (None if args.duration else 10)
    with (output_dir / "sent.jsonl").open("w", encoding="utf-8") as sent_stream, \
         (output_dir / "outputs.jsonl").open("w", encoding="utf-8") as output_stream:
        try:
            if not args.no_observe:
                topics = [settings.KAFKA_FACE_DETECTION_RECORD_TOPIC,
                          settings.KAFKA_FACE_RECOGNITION_RECORD_TOPIC,
                          settings.KAFKA_PERSON_DETECTION_RECORD_TOPIC,
                          settings.KAFKA_PERSON_RECOGNITION_RECORD_TOPIC,
                          settings.KAFKA_VECTOR_RECORD_TOPIC]
                consumer = AIOKafkaConsumer(group_id=None, enable_auto_commit=False,
                                           auto_offset_reset="none", **settings.kafka_client_config)
                await consumer.start()
                available_topics = await consumer.topics()
                partitions = []
                for topic in dict.fromkeys(topics):
                    if topic not in available_topics:
                        raise ValueError(f"输出 Topic 不存在: {topic}；可 --no-observe 只发送")
                    ids = consumer.partitions_for_topic(topic)
                    if not ids:
                        raise ValueError(f"输出 Topic 不存在: {topic}；可 --no-observe 只发送")
                    partitions.extend(TopicPartition(topic, i) for i in sorted(ids))
                consumer.assign(partitions)
                ends = await consumer.end_offsets(partitions)
                for tp in partitions:
                    consumer.seek(tp, ends[tp])
                observer = OutputObserver(consumer, settings.KAFKA_VECTOR_RECORD_TOPIC, output_stream)
                observer_task = asyncio.create_task(observer.run())
            await producer.start()
            # 客户端就绪后再定位日志，排除连接初始化期间的历史汇总。
            log_offset = args.log_file.stat().st_size if args.log_file and args.log_file.exists() else 0
            started_at = time.time()
            started_clock = time.monotonic()

            async def publish_one(sequence: int) -> None:
                source = records[sequence % len(records)]
                uid = make_replay_uid(run_id, sequence, trace_completion=args.trace_completion)
                sent_at = time.time()
                payload = replay_payload(source["payload"], uid=uid, event_time=args.event_time,
                                         keep_pano=args.keep_pano, now_ms=int(sent_at * 1000))
                if observer:
                    observer.sent_at[uid] = sent_at
                entry = {"sequence": sequence, "uId": uid, "sent_at": sent_at,
                         "source": source.get("source", {}), "topic": args.topic}
                try:
                    metadata = await producer.send_and_wait(args.topic, value=encode_json(payload))
                    entry.update({"ack_at": time.time(), "partition": metadata.partition,
                                  "offset": metadata.offset, "status": "acked"})
                except Exception as error:
                    entry.update({"status": "failed", "error": type(error).__name__})
                sent_records.append(entry)
                sent_stream.write(encode_json(entry).decode("utf-8") + "\n")
                sent_stream.flush()

            sequence = 0
            next_due = started_clock
            while count_limit is None or sequence < count_limit:
                if args.duration and time.monotonic() - started_clock >= args.duration:
                    break
                if observer_task and observer_task.done():
                    observer_task.result()
                while len(pending) >= args.max_inflight:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                delay = next_due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                if args.duration and time.monotonic() - started_clock >= args.duration:
                    break
                task = asyncio.create_task(publish_one(sequence))
                pending.add(task)
                sequence += 1
                # 绝对计划避免把每次唤醒迟到和 ACK/调度开销逐条叠加；
                # 落后时仍受 max_inflight 限制，不能无限创建发送任务。
                next_due = started_clock + sequence / args.qps
            if pending:
                await asyncio.gather(*pending)
                pending.clear()
            send_finished_at = time.time()
            print(f"发送结束；结果目录: {output_dir}", flush=True)
            if observer_task:
                print(f"等待 {args.wait_seconds:g}s 观察输出。", flush=True)
                # 固定观察窗口允许异步向量缓冲刷新，不能见到整图向量就提前结束。
                done, _ = await asyncio.wait([observer_task], timeout=args.wait_seconds)
                if done:
                    observer_task.result()
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            # 中断也保留已得到 ACK 的发送清单，不遗留 Kafka 后台任务。
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if observer_task:
                observer_task.cancel()
                await asyncio.gather(observer_task, return_exceptions=True)
            try:
                await producer.stop()
            finally:
                if consumer:
                    await consumer.stop()
            finished_at = time.time()
            acked = sum(item["status"] == "acked" for item in sent_records)
            send_elapsed = max(.001, locals().get("send_finished_at", finished_at) - started_at)
            summary = {"run_id": run_id, "topic": args.topic, "snapshot_messages": len(records),
                       "target_qps": args.qps, "requested_count": count_limit,
                       "requested_duration_seconds": args.duration,
                       "event_time": args.event_time, "keep_pano": args.keep_pano,
                       "trace_completion": args.trace_completion,
                       "acked_messages": acked, "failed_messages": len(sent_records) - acked,
                       "send_seconds": send_elapsed, "acked_messages_per_second": acked / send_elapsed,
                       "started_at": started_at, "finished_at": finished_at, "interrupted": interrupted,
                       "observation": observer.summary() if observer else None,
                       "metrics_before": metrics_before, "metrics_after": await metric_snapshot(args.metrics),
                       "performance_summaries": read_performance_summaries(args.log_file, log_offset)
                       if args.log_file else [],
                       "notes": ["显式 --trace-completion 的完成数可由 performance replay_runs 按 run 聚合；旧模式仅有 Topic 聚合。",
                                 "picture 向量输出仅证明其写库后已输出，不能证明该帧所有阶段完成。",
                                 "各类人脸/人物向量一图可多行；启用 pano 后一消息可处理两张图。",
                                 "metrics_before/after 的 gauge 差值不是吞吐量；counter 重置也不能直接相减。"]}
            write_json(output_dir / "summary.json", summary)
            display_summary = {key: value for key, value in summary.items()
                               if key not in ("metrics_before", "metrics_after")}
            display_summary["result_directory"] = str(output_dir)
            display_summary["metric_snapshots"] = {
                stage: {name: values if "error" in values else {"series": len(values)}
                        for name, values in summary[stage].items()}
                for stage in ("metrics_before", "metrics_after")
            }
            print(json.dumps(display_summary, ensure_ascii=False, indent=2), flush=True)
    if any(item["status"] == "failed" for item in sent_records):
        raise RuntimeError("存在发送失败，详见本轮 sent.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("snapshot", help="只读采样有限历史消息，不生产消息、不提交 offset")
    sample.add_argument("--topic", default="PtzPersonDetSync")
    sample.add_argument("--count", type=positive_int, default=100)
    sample.add_argument("--lookback", type=positive_int, default=1000, help="每分区最多回看 offset 数")
    sample.add_argument("--max-scan", type=positive_int, default=10000)
    sample.add_argument("--timeout", type=positive_float, default=30)
    sample.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "person_snapshot.jsonl")
    sample.set_defaults(handler=snapshot)
    send = commands.add_parser("replay", help="显式写入 Kafka，运行真实下载/推理/写库/输出链路")
    send.add_argument("--input", type=Path, required=True)
    send.add_argument("--topic", default="PtzPersonDetSync")
    send.add_argument("--qps", type=positive_float, default=1)
    send.add_argument("--count", type=positive_int, help="默认10条；超出快照条数时循环使用")
    send.add_argument("--duration", type=positive_float, help="与 count 同时指定时先达到者停止；单独指定则按时长")
    send.add_argument("--max-inflight", type=positive_int, default=20)
    send.add_argument("--event-time", choices=("preserve", "now"), default="preserve")
    send.add_argument("--keep-pano", action="store_true", help="保留原 panoKeyframeUrl，消息数可能少于图片数")
    send.add_argument("--trace-completion", action="store_true",
                      help="测试专用非UUID uId，启用有界run完成追踪；每轮最多65536条，默认保持UUID5")
    send.add_argument("--wait-seconds", type=positive_float, default=60)
    send.add_argument("--no-observe", action="store_true", help="跳过输出 Topic 观察")
    send.add_argument("--log-file", type=Path, help="采集同窗口 INFO performance summary，不启动 ALog")
    send.add_argument("--metrics", action="append", default=[], metavar="NAME=URL", help="可重复，读取指标前后快照")
    send.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    send.set_defaults(handler=replay)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        asyncio.run(args.handler(args))
    except KeyboardInterrupt:
        print("已中断，已完成的发送记录保留在结果目录。", file=sys.stderr)
        return 130
    except Exception as error:
        # 不打印连接配置或可能含敏感 URL 的服务异常正文。
        print(f"运行失败: {type(error).__name__}；检查参数、服务与本地日志。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
