"""只读比对两次隔离回放：冻结 Kafka end offsets，stdout 仅输出聚合，不输出身份值。"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from performance_tools.isolated_replay import load_manifest
from performance_tools.kafka_replay import make_replay_uid, positive_float, positive_int

KINDS = ("face_detection", "face_recognition", "person_detection", "person_recognition", "vector_record")
FACE_FIELDS = ("faceOrientation", "faceQualityScore", "faceQualityLevel")
PERSON_FIELDS = (
    "faceOrientation", "personOrientation", "personQualityScore", "personQualityLevel",
    "personQualityAction", "maskScore", "cropScore", "poseScore", "visibleKeypointCount",
    "torsoKeypointCount", "shoulderHipChainCount", "halfBodyPoseEligible", "bodyCompleteness",
    "isHalfBody", "isTruncated", "hasSecondPerson", "foreignPersonRatio", "foreignPersonCoreRatio",
    "foreignPersonPixelCount", "foreignPersonCorePixelCount", "foreignPersonBBoxFallbackRatio",
    "foreignPersonBBoxFallbackPixelCount", "foreignPersonEdgeDominant", "foreignPersonEvidenceSource",
    "paddingRatio", "letterboxPaddingRatio", "sourceOverflowRatio", "bboxOverflowRatio",
    "nearImageEdgeCount", "truncatedEdgeCount", "qualityReasons",
)
DETECTION_FIELDS = {
    "face": (*FACE_FIELDS, "confidence", "status", "faceKeypoint"),
    "person": (*PERSON_FIELDS, "confidence", "status", "maskPixelCount", "personKeypoint"),
}
RECOGNITION_FIELDS = {
    "face": (*FACE_FIELDS, "score", "faceStatus"),
    "person": (*PERSON_FIELDS, "score", "reidStatus", "reidState", "reidEnrollmentState",
               "reidEnrollmentReason", "reidDecision", "reidDecisionReason", "reidSearchMode",
               "reidSource", "isNewReid", "reidTop1Score", "reidMeanScore", "reidClassScore",
               "reidSupportCount", "reidRunnerUpScore", "reidMargin", "reidRequiredMargin",
               "sameFrameExcludedReidCount"),
}
IDENTITY_FIELDS = {"face": ("aiFaceId", "vectorHitFaceId"),
                   "person": ("aiFaceId", "reid", "vectorHitReid", "canonicalFaceId")}
MISSING = object()


@dataclass
class RunInput:
    manifest: dict[str, Any]
    summary: dict[str, Any]
    proof: dict[str, Any]
    sent: dict[int, dict[str, Any]]
    uid_to_sequence: dict[str, int]
    manifest_sha256: str


def load_run(manifest_path: Path, run_directory: Path) -> RunInput:
    manifest = load_manifest(manifest_path, prepared=True)
    summary = json.loads((run_directory / "summary.json").read_text())
    proof = json.loads(manifest_path.with_name("prepared.json").read_text())
    if summary.get("topic") != manifest["names"]["input_topic"]:
        raise ValueError("run summary 的输入 Topic 与 manifest 不符")
    sent, uid_to_sequence = {}, {}
    rows = [json.loads(line) for line in (run_directory / "sent.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        sequence = row.get("sequence")
        if type(sequence) is not int or sequence < 0 or sequence in sent:
            raise ValueError("sent sequence 缺失、非法或重复")
        expected = make_replay_uid(summary["run_id"], sequence,
                                   trace_completion=summary.get("trace_completion", False))
        if row.get("uId") != expected or row.get("topic") != summary["topic"]:
            raise ValueError("sent UID/Topic 不属于 summary 指定的 run")
        if row.get("status") not in {"acked", "failed"}:
            raise ValueError("sent status 非法")
        sent[sequence] = row
        # 发送失败也可能已到 broker；保留本轮全部合法 UID，单独报告失败数。
        uid_to_sequence[expected] = sequence
    if not rows or sum(row["status"] == "acked" for row in rows) != summary.get("acked_messages"):
        raise ValueError("summary ack 数与 sent 不符，或 sent 为空")
    if sum(row["status"] == "failed" for row in rows) != summary.get("failed_messages"):
        raise ValueError("summary failed 数与 sent 不符")
    return RunInput(manifest, summary, proof, sent, uid_to_sequence,
                    hashlib.sha256(manifest_path.read_bytes()).hexdigest())


def validate_pair(left: RunInput, right: RunInput) -> dict[str, Any]:
    if left.manifest["names"]["output_topics"] == right.manifest["names"]["output_topics"]:
        raise ValueError("A/B 必须使用不同隔离命名空间")
    if left.manifest["input"]["sha256"] != right.manifest["input"]["sha256"]:
        raise ValueError("A/B 输入快照哈希不同")
    if left.proof.get("warm_snapshot_sha256") != right.proof.get("warm_snapshot_sha256"):
        raise ValueError("A/B 初始暖图库快照不同")
    if left.proof.get("warm_rows") != right.proof.get("warm_rows"):
        raise ValueError("A/B 初始图库行数不同")
    for field in ("event_time", "keep_pano"):
        if left.summary.get(field) != right.summary.get(field):
            raise ValueError("A/B 回放图片/事件时间策略不同")
    common = set(left.sent) & set(right.sent)
    if not common or any(left.sent[key].get("source") != right.sent[key].get("source") for key in common):
        raise ValueError("A/B sequence 对应的源消息不同或没有公共 sequence")
    return {"shared_sequences": len(common), "left_only_sequences": len(set(left.sent) - common),
            "right_only_sequences": len(set(right.sent) - common),
            "same_input_snapshot": True, "same_prepared_warm_snapshot": True,
            "prepared_warm_rows": left.proof.get("warm_rows", {})}


async def capture_outputs(consumer, partitions, topic_routes, runs, *, timeout=60.0,
                          max_records=1_000_000, max_payload_bytes=128 * 1024 * 1024):
    """只读取开始时冻结的 [begin, end)；无 subscribe、group 或 commit 操作。"""
    beginnings = await consumer.beginning_offsets(partitions)
    ends = await consumer.end_offsets(partitions)
    consumer.assign(partitions)
    for tp in partitions:
        consumer.seek(tp, beginnings[tp])
    active = {tp for tp in partitions if beginnings[tp] < ends[tp]}
    selected = {side: [] for side in runs}
    counts = Counter()
    retained_bytes = 0
    deadline = time.monotonic() + timeout
    stop_reason = None
    while active and stop_reason is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop_reason = "timeout"
            break
        if counts["scanned"] >= max_records:
            stop_reason = "max_records"
            break
        batches = await consumer.getmany(*active, timeout_ms=min(1000, max(1, int(remaining * 1000))),
                                         max_records=min(1000, max_records - counts["scanned"]))
        for tp, batch in batches.items():
            if tp not in active:
                continue
            side, kind = topic_routes[tp.topic]
            for record in batch:
                if record.offset < beginnings[tp] or record.offset >= ends[tp]:
                    continue
                counts["scanned"] += 1
                try:
                    payload = json.loads(record.value)
                    if not isinstance(payload, dict):
                        raise ValueError
                except (ValueError, TypeError, UnicodeDecodeError):
                    counts["invalid_json"] += 1
                    continue
                uid = payload.get("keyframeId")
                if not isinstance(uid, str):
                    counts["invalid_keyframe_id"] += 1
                    continue
                if uid not in runs[side].uid_to_sequence:
                    counts["other_run_records"] += 1
                    continue
                retained_bytes += len(record.value)
                if retained_bytes > max_payload_bytes:
                    stop_reason = "max_payload_bytes"
                    break
                selected[side].append({"kind": kind, "topic": tp.topic, "partition": tp.partition,
                                       "offset": record.offset, "payload": payload})
            if stop_reason:
                break
        for tp in tuple(active):
            if await consumer.position(tp) >= ends[tp]:
                active.remove(tp)
                consumer.pause(tp)
    bounds = [{"topic": tp.topic, "partition": tp.partition, "begin": beginnings[tp],
               "end_exclusive": ends[tp]} for tp in partitions]
    return selected, {"frozen_end_offsets": True, "group_id": None, "committed_offsets": False,
                      "read_complete": not active and stop_reason is None,
                      "stop_reason": stop_reason, "retention_may_have_removed_records": any(beginnings.values()),
                      "partition_bounds": bounds, "counts": dict(counts),
                      "retained_payload_bytes": retained_bytes}


class ReadOnlyMetadataConsumer:
    """aiokafka 0.10 兼容层：只做全量 metadata 请求，不请求创建指定 Topic。

    0.10 没有 allow_auto_create_topics 参数，MetadataRequest v1 也没有该开关。
    assign 会同步设置显式 Topic 并唤醒后台元数据任务；在让出事件循环之前
    恢复空跟踪集合。Fetcher 遇到缺失 leader 还会 add_topic，因此同时在
    metadata 更新入口固定 topics=[]，确保 v1 请求的 topics 为 None（查询全部）。
    分区 assignment 保留，ListOffsets/Fetch 仍只读取调用方指定的已存在分区。
    此内部接口由真实 aiokafka 构造/协议请求测试约束，升级依赖后需复核。
    """

    def __init__(self, consumer):
        self._consumer = consumer
        self._original_metadata_update = consumer._client._metadata_update
        consumer._client._metadata_update = self._update_all_metadata

    async def _update_all_metadata(self, cluster_metadata, topics):
        return await self._original_metadata_update(cluster_metadata, [])

    def assign(self, partitions):
        try:
            self._consumer.assign(partitions)
        finally:
            # 必须同步执行；两步之间不能 await。
            self._consumer._client.set_topics([])

    def __getattr__(self, name):
        return getattr(self._consumer, name)


async def read_kafka(runs, *, timeout, max_records, max_payload_bytes):
    from aiokafka import AIOKafkaConsumer
    from aiokafka.structs import TopicPartition
    from core.config import settings

    topic_routes = {topic: (side, kind) for side, run in runs.items()
                    for kind, topic in run.manifest["names"]["output_topics"].items()}
    consumer = ReadOnlyMetadataConsumer(AIOKafkaConsumer(
        group_id=None, enable_auto_commit=False, auto_offset_reset="none", **settings.kafka_client_config))
    try:
        await consumer.start()
        if not set(topic_routes).issubset(await consumer.topics()):
            raise ValueError("隔离输出 Topic 缺失")
        partitions = []
        for topic in sorted(topic_routes):
            ids = consumer.partitions_for_topic(topic)
            if not ids:
                raise ValueError("隔离输出 Topic 没有可读分区")
            partitions.extend(TopicPartition(topic, index) for index in sorted(ids))
        return await capture_outputs(consumer, partitions, topic_routes, runs, timeout=timeout,
                                     max_records=max_records, max_payload_bytes=max_payload_bytes)
    finally:
        await consumer.stop()


def rectangle(payload, kind):
    rect = payload.get("faceRect" if kind == "face" else "peopleRect")
    if not isinstance(rect, dict) or any(type(rect.get(key)) is not int for key in ("x", "y", "w", "h")):
        raise ValueError("检测框缺失或非法")
    if rect["w"] <= 0 or rect["h"] <= 0:
        raise ValueError("检测框宽高必须为正数")
    return tuple(rect[key] for key in ("x", "y", "w", "h"))


def index_outputs(run, records):
    grouped = {kind: [] for kind in KINDS}
    counts, diagnostics = Counter(), Counter()
    for row in records:
        if not isinstance(row, dict):
            diagnostics["invalid_envelopes"] += 1
            continue
        kind, payload = row.get("kind"), row.get("payload")
        if not isinstance(kind, str) or kind not in grouped or not isinstance(payload, dict):
            diagnostics["invalid_envelopes"] += 1
            continue
        if row.get("topic") != run.manifest["names"]["output_topics"][kind]:
            raise ValueError("payload 快照包含非本manifest的Topic")
        uid = payload.get("keyframeId")
        if not isinstance(uid, str):
            diagnostics["invalid_keyframe_id"] += 1
            continue
        sequence = run.uid_to_sequence.get(uid)
        if sequence is None:
            diagnostics["other_run_records"] += 1
            continue
        if type(payload.get("isPanoKeyframe")) is not bool:
            diagnostics["invalid_pano_flag"] += 1
            continue
        grouped[kind].append((sequence, payload["isPanoKeyframe"], payload))
        counts[kind] += 1
    detections = {kind: defaultdict(list) for kind in ("face", "person")}
    associations = {kind: defaultdict(list) for kind in ("face", "person")}
    recognitions = {kind: defaultdict(list) for kind in ("face", "person")}
    for kind in detections:
        for sequence, pano, payload in grouped[kind + "_detection"]:
            try:
                key = sequence, pano, rectangle(payload, kind)
            except ValueError:
                diagnostics[kind + "_invalid_rect"] += 1
                continue
            detections[kind][key].append(payload)
            if isinstance(payload.get("aiId"), str) and payload["aiId"]:
                associations[kind][sequence, pano, payload["aiId"]].append(key)
            else:
                diagnostics[kind + "_missing_ai_id"] += 1
        for sequence, pano, payload in grouped[kind + "_recognition"]:
            ai_id = payload.get("aiId")
            if not isinstance(ai_id, str) or not ai_id:
                diagnostics[kind + "_invalid_recognition_ai_id"] += 1
                continue
            matches = associations[kind].get((sequence, pano, ai_id), [])
            if len(matches) != 1:
                diagnostics[kind + "_unlinked_recognition"] += 1
                continue
            key = matches[0]
            if kind == "person":
                try:
                    rect = rectangle(payload, kind)
                except ValueError:
                    rect = None
                if rect != key[2]:
                    diagnostics["person_recognition_rect_disagrees"] += 1
                    continue
            recognitions[kind][key].append(payload)
    vectors = defaultdict(Counter)
    vector_targets = Counter()
    collection_aliases = {value: key for key, value in run.manifest["names"]["collections"].items()}
    for sequence, pano, payload in grouped["vector_record"]:
        target = payload.get("targetType")
        if not isinstance(target, str) or target not in {"face", "person", "person_sbs", "picture"}:
            diagnostics["unknown_vector_target"] += 1
            continue
        vector_targets[target] += 1
        if target == "picture":
            key = sequence, pano, ()
        else:
            kind = "face" if target == "face" else "person"
            ai_id = payload.get("aiId")
            if not isinstance(ai_id, str) or not ai_id:
                diagnostics["invalid_vector_ai_id"] += 1
                continue
            matches = associations[kind].get((sequence, pano, ai_id), [])
            if len(matches) != 1:
                diagnostics["unlinked_vector_record"] += 1
                continue
            key = matches[0]
        database = payload.get("vectorDbName")
        collection = collection_aliases.get(database) if isinstance(database, str) else None
        if collection is None:
            diagnostics["unexpected_vector_collection"] += 1
            continue
        if type(payload.get("vectorDim")) is not int or not isinstance(payload.get("modelName"), str):
            diagnostics["invalid_vector_metadata"] += 1
            continue
        signature = target, payload["vectorDim"], payload["modelName"], collection
        vectors[target, key][signature] += 1
    return {"detections": detections, "recognitions": recognitions, "vectors": vectors,
            "records_by_kind": dict(counts), "vector_records_by_target": dict(vector_targets),
            "diagnostics": dict(diagnostics)}


def equivalent(left, right, tolerance):
    if left is MISSING or right is MISSING:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            return left == right
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b, tolerance) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def compare_fields(left, right, fields, mismatches, deltas, tolerance):
    different = False
    for field in fields:
        a, b = left.get(field, MISSING), right.get(field, MISSING)
        if not equivalent(a, b, tolerance):
            mismatches[field] += 1
            different = True
        if type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b):
            deltas[field] = max(deltas.get(field, 0), abs(a - b))
    return different


def identity_summary(pairs):
    counts, forward, backward = Counter(), defaultdict(set), defaultdict(set)
    for left, right in pairs:
        if left is MISSING or right is MISSING:
            counts["left_missing"] += int(left is MISSING)
            counts["right_missing"] += int(right is MISSING)
            counts["not_comparable_missing"] += 1
            continue
        if not isinstance(left, str) or not isinstance(right, str):
            counts["left_invalid_type"] += int(not isinstance(left, str))
            counts["right_invalid_type"] += int(not isinstance(right, str))
            counts["not_comparable_invalid_type"] += 1
            continue
        a = None if left in (None, "", "0", 0) else str(left)
        b = None if right in (None, "", "0", 0) else str(right)
        if a is None or b is None:
            counts["both_unassigned" if a is b else "unassigned_to_assigned" if a is None else "assigned_to_unassigned"] += 1
        else:
            counts["same_nonzero_value" if a == b else "different_nonzero_value"] += 1
            forward[a].add(b)
            backward[b].add(a)
    return {**dict(counts), "left_identities_split": sum(len(values) > 1 for values in forward.values()),
            "right_identities_merged": sum(len(values) > 1 for values in backward.values())}


def compare_outputs(left, right, selected, *, tolerance=1e-5):
    alignment = validate_pair(left, right)
    indexed = {"left": index_outputs(left, selected["left"]), "right": index_outputs(right, selected["right"])}
    result = {"schema": 1, "alignment": alignment, "numeric_tolerance": tolerance,
              "runs": {side: {"records_by_kind": values["records_by_kind"],
                               "vector_records_by_target": values["vector_records_by_target"],
                               "diagnostics": values["diagnostics"],
                               "sent_messages": len(run.sent), "failed_sends": run.summary["failed_messages"],
                               "interrupted": bool(run.summary.get("interrupted"))}
                       for side, run, values in (("left", left, indexed["left"]), ("right", right, indexed["right"]))},
              "targets": {}}
    for kind in ("face", "person"):
        a, b = indexed["left"]["detections"][kind], indexed["right"]["detections"][kind]
        common = set(a) & set(b)
        detection_diff, recognition_diff, recognition_presence = Counter(), Counter(), Counter()
        detection_deltas, recognition_deltas = {}, {}
        identities = {field: [] for field in IDENTITY_FIELDS[kind]}
        compared_detection = changed_detection = compared_recognition = changed_recognition = ambiguous = 0
        for key in common:
            if len(a[key]) != 1 or len(b[key]) != 1:
                ambiguous += 1
                continue
            compared_detection += 1
            changed_detection += compare_fields(a[key][0], b[key][0], DETECTION_FIELDS[kind],
                                                 detection_diff, detection_deltas, tolerance)
            ar = indexed["left"]["recognitions"][kind].get(key, [])
            br = indexed["right"]["recognitions"][kind].get(key, [])
            if len(ar) > 1 or len(br) > 1:
                recognition_presence["ambiguous_multiple_records"] += 1
            elif not ar or not br:
                recognition_presence["neither" if not ar and not br else "left_only" if ar else "right_only"] += 1
            else:
                recognition_presence["both"] += 1
                compared_recognition += 1
                changed_recognition += compare_fields(ar[0], br[0], RECOGNITION_FIELDS[kind],
                                                       recognition_diff, recognition_deltas, tolerance)
                for field in identities:
                    identities[field].append((ar[0].get(field, MISSING), br[0].get(field, MISSING)))
        result["targets"][kind] = {
            "left_detection_records": sum(map(len, a.values())), "right_detection_records": sum(map(len, b.values())),
            "exact_rect_matches": len(common), "left_unmatched_rects": len(set(a) - set(b)),
            "right_unmatched_rects": len(set(b) - set(a)), "ambiguous_matched_rects": ambiguous,
            "left_duplicate_rects": sum(len(values) > 1 for values in a.values()),
            "right_duplicate_rects": sum(len(values) > 1 for values in b.values()),
            "detection_compared": compared_detection, "detection_changed": changed_detection,
            "detection_field_mismatches": dict(detection_diff), "detection_max_abs_delta": detection_deltas,
            "recognition_presence": dict(recognition_presence), "recognition_compared": compared_recognition,
            "recognition_changed": changed_recognition, "recognition_field_mismatches": dict(recognition_diff),
            "recognition_max_abs_delta": recognition_deltas,
            "identities": {field: identity_summary(pairs) for field, pairs in identities.items()},
        }
    av, bv = indexed["left"]["vectors"], indexed["right"]["vectors"]
    result["vectors"] = {"shared_target_keys": len(set(av) & set(bv)),
                         "left_only_target_keys": len(set(av) - set(bv)),
                         "right_only_target_keys": len(set(bv) - set(av)),
                         "changed_metadata_or_count": sum(av[key] != bv[key] for key in set(av) & set(bv))}
    result["notes"] = [
        "目标按 sequence/isPanoKeyframe/精确rect匹配；框变化显示为两侧未匹配，不猜测跨框身份。",
        "face识别经同run aiId关联检测框；person识别同时校验peopleRect。重复目标保留并标记歧义。",
        "不同非零身份值可能是新生随机ID；split/merge只在双方都有身份的已匹配目标上统计，不自动判定回归。",
        "prepared哈希仅证明prepare时使用相同暖图库快照；同namespace先前回放和warmup可能已改变正式run开始时的图库。",
        "0与非零变化及身份状态/分数变化如实统计；相同prepare快照不保证并发学习过程中每次分配相同。",
        "忽略随机aiId/vectorId/pictureVectorId、图片object key、时间、设备SN和包含动态身份的info；aiId仅用于run内关联。",
        "向量输出只有metadata；比较targetType/vectorDim/modelName/去前缀后的集合角色和记录数，不比较embedding。",
        "冻结末端读取完成不代表Worker已完成所有图片；请结合run完成统计及输出排空状态判断。",
    ]
    return result


def save_capture(path, runs, selected, capture):
    # 完整payload仅允许落在已经被仓库忽略的产物目录，权限0600，不覆盖旧采集。
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "exports", PROJECT_ROOT / "performance_tools/results")
    if not any(resolved.is_relative_to(directory.resolve()) for directory in allowed):
        raise ValueError("完整payload文件必须位于exports/或performance_tools/results/下")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema": 1, "runs": {side: {"manifest_sha256": run.manifest_sha256,
                                            "run_id": run.summary["run_id"]} for side, run in runs.items()},
            "capture": capture, "records": selected}
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)


def load_capture(path, runs):
    data = json.loads(path.read_text())
    expected = {side: {"manifest_sha256": run.manifest_sha256, "run_id": run.summary["run_id"]}
                for side, run in runs.items()}
    if data.get("schema") != 1 or data.get("runs") != expected:
        raise ValueError("payload快照与当前manifest/run不匹配")
    return data["records"], data["capture"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-manifest", type=Path, required=True)
        parser.add_argument(f"--{side}-run", type=Path, required=True, help="含summary.json/sent.jsonl的目录")
    parser.add_argument("--timeout", type=positive_float, default=60.0)
    parser.add_argument("--max-records", type=positive_int, default=1_000_000)
    parser.add_argument("--max-payload-mib", type=positive_int, default=128)
    parser.add_argument("--tolerance", type=positive_float, default=1e-5)
    parser.add_argument("--from-capture", type=Path, help="离线重读本工具保存的完整payload快照")
    parser.add_argument("--save-capture", type=Path, help="可选：0600写入ignored产物目录，默认仅内存保留")
    return parser


def main():
    args = build_parser().parse_args()
    runs = {side: load_run(getattr(args, side + "_manifest"), getattr(args, side + "_run"))
            for side in ("left", "right")}
    validate_pair(runs["left"], runs["right"])
    if args.from_capture:
        selected, capture = load_capture(args.from_capture, runs)
    else:
        selected, capture = asyncio.run(read_kafka(runs, timeout=args.timeout, max_records=args.max_records,
                                                   max_payload_bytes=args.max_payload_mib * 1024 * 1024))
    if args.save_capture:
        save_capture(args.save_capture, runs, selected, capture)
    result = compare_outputs(runs["left"], runs["right"], selected, tolerance=args.tolerance)
    result["capture"] = capture
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if capture["read_complete"] and not capture.get("retention_may_have_removed_records") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # 不将payload/身份值或底层连接异常消息写到公开输出。
        print(f"比对未完成：{type(error).__name__}，请核对manifest、run文件和读取条件。", file=sys.stderr)
        raise SystemExit(2)
