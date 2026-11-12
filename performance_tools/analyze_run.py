"""离线比较 Kafka 回放结果；严格区分输入 ACK、图片业务完成和向量持久化输出。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.performance import parse_replay_uid


PERFORMANCE_MARKER = "ALog performance summary: "


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number} 必须是 JSON object")
        rows.append(row)
    return rows


def number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {"count": len(values), "avg": sum(values) / len(values),
            "p50": values[(len(values) - 1) // 2],
            "p95": values[math.ceil(len(values) * .95) - 1], "max": values[-1]}


def aggregate_timings(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = sum(max(0, int(number(row.get("count")))) for row in rows)
    total = sum(max(0.0, number(row.get("total_ms"))) for row in rows)
    return {"count": count,
            "errors": sum(max(0, int(number(row.get("errors")))) for row in rows),
            "total_ms": total, "avg_ms": total / count if count else 0.0,
            "max_ms": max((number(row.get("max_ms")) for row in rows), default=0.0)}


def load_summaries(
    summary: dict[str, Any], log_files: list[Path],
) -> list[dict[str, Any]]:
    candidates = list(summary.get("performance_summaries", []))
    for path in log_files:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if PERFORMANCE_MARKER not in line:
                    continue
                try:
                    candidates.append(json.loads(line.split(PERFORMANCE_MARKER, 1)[1]))
                except json.JSONDecodeError:
                    continue
    start_ms = number(summary.get("started_at")) * 1000
    end_ms = number(summary.get("finished_at")) * 1000
    unique = {}
    for row in candidates:
        if not isinstance(row, dict):
            continue
        start = number(row.get("interval_start_unix_ms"))
        end = number(row.get("interval_end_unix_ms"))
        if end <= start or end <= start_ms or start >= end_ms:
            continue
        key = (row.get("pid"), start, end)
        unique[key] = row
    return sorted(unique.values(), key=lambda row: row["interval_start_unix_ms"])


def summarize_business(
    rows: list[dict[str, Any]], *, topic: str, run_start: float, run_end: float,
    last_ack: float | None,
) -> dict[str, Any]:
    periods = []
    for row in rows:
        seconds = number(row.get("interval_seconds"))
        if seconds <= 0:
            continue
        start = number(row.get("interval_start_unix_ms")) / 1000
        end = number(row.get("interval_end_unix_ms")) / 1000
        images = max(0, int(number(row.get("images_completed_by_topic", {}).get(topic))))
        messages = max(0, int(number(row.get("messages_completed_by_topic", {}).get(topic))))
        periods.append({"start_unix": start, "end_unix": end,
                        "seconds": seconds, "images": images, "messages": messages,
                        "images_per_second": images / seconds,
                        "fully_inside_run": start >= run_start and end <= run_end,
                        "final": bool(row.get("final"))})
    images = sum(row["images"] for row in periods)
    seconds = sum(row["seconds"] for row in periods)
    active = [row for row in periods if row["images"]]
    contained = [row for row in periods if row["fully_inside_run"]]
    active_span = active[-1]["end_unix"] - active[0]["start_unix"] if active else 0
    result = {
        "scope": "ALog process + topic; may include other producers and pano images",
        "full_pipeline_completion_verified_for_run": False,
        "images_in_captured_periods": images,
        "messages_in_captured_periods": sum(row["messages"] for row in periods),
        "captured_period_seconds": seconds,
        "captured_period_images_per_second": images / seconds if seconds else None,
        "active_period_span_seconds": active_span,
        "active_period_span_images_per_second": images / active_span if active_span else None,
        "peak_period": max(periods, key=lambda row: row["images_per_second"], default=None),
        "peak_fully_contained_period": max(
            contained, key=lambda row: row["images_per_second"], default=None),
        "periods": periods,
        "last_observed_topic_image_period": active[-1] if active else None,
        "run_business_drain_seconds": None,
    }
    if active and last_ack is not None:
        result["last_observed_topic_image_period_after_last_ack_seconds"] = [
            max(0.0, active[-1]["start_unix"] - last_ack),
            max(0.0, active[-1]["end_unix"] - last_ack),
        ]
    return result


def summarize_run_completions(
    rows: list[dict[str, Any]], *, summary: dict[str, Any],
    acked: dict[str, dict[str, Any]], first_sent: float | None, last_ack: float | None,
) -> dict[str, Any]:
    """读取有界 run 的累计快照，不累加重复周期，不用 picture 输出补齐完成数。"""
    run_id = summary.get("run_id")
    samples = [
        (row.get("pid"), row.get("interval_end_unix_ms"), values)
        for row in rows
        if isinstance((values := row.get("replay_runs", {}).get(run_id)), dict)
    ]
    result = {
        "scope": "explicit test run main frames; pano counted separately; asynchronous persistence may remain",
        "trace_enabled": bool(summary.get("trace_completion")),
        "observed": bool(samples),
        "completed_count": None,
        "pano_completed_count": None,
        "count_is_exact": False,
        "expected_main_images": len(acked),
        "first_send_unix_ms": first_sent * 1000 if first_sent is not None else None,
        "all_main_images_completed": False,
        "full_pipeline_completion_verified_for_run": False,
        "first_completed_unix_ms": None,
        "last_completed_unix_ms": None,
        "first_send_to_last_completion_seconds": None,
        "observed_images_per_second": None,
        "run_business_drain_seconds": None,
        "reasons": [],
    }
    if not samples:
        result["reasons"].append("no run completion snapshots; legacy UUID runs cannot be recovered")
        return result
    reasons = result["reasons"]
    if not result["trace_enabled"]:
        reasons.append("trace mode was not explicitly enabled in replay summary")
    segments = {(pid, values.get("generation")) for pid, _, values in samples}
    if any(not isinstance(pid, int) or pid <= 0
           or not isinstance(values.get("generation"), int) or values["generation"] <= 0
           for pid, _, values in samples):
        reasons.append("missing valid process/generation identity")
    if len(segments) != 1:
        reasons.append("multiple process/generation segments; bounded tracking may have restarted or evicted this run")
    if any(values.get("topic") != summary.get("topic") or values.get("conflicting_topic")
           for _, _, values in samples):
        reasons.append("run topic does not match or was reused across topics")
    main_values = [values.get("main", {}) for _, _, values in samples]
    main = max(main_values, key=lambda value: number(value.get("completed_count")))
    count = int(number(main.get("completed_count")))
    result.update({
        "completed_count": count,
        "pano_completed_count": max(
            (int(number(values.get("pano", {}).get("completed_count"))) for _, _, values in samples), default=0),
        "first_completed_unix_ms": main.get("first_completed_unix_ms"),
        "last_completed_unix_ms": main.get("last_completed_unix_ms"),
        "observed_through_unix_ms": max(number(end) for _, end, _ in samples),
        "duplicate_main_completions": max(
            (int(number(value.get("duplicate_completions"))) for value in main_values), default=0),
    })
    counts = [number(value.get("completed_count")) for value in main_values]
    if any(later < earlier for earlier, later in zip(counts, counts[1:])):
        reasons.append("cumulative completion counter decreased")
    if any(value.get("clock_went_backwards") for value in main_values):
        reasons.append("worker wall clock went backwards")
    result["count_is_exact"] = not reasons
    if len(segments) != 1:
        # 不同进程或重新建立的位图可能重复计同一 UID，max 也不能代表整个 run。
        result["completed_count"] = None
        result["pano_completed_count"] = None
        return result
    sequences = [parse_replay_uid(uid) for uid in acked]
    manifest_matches = (
        bool(acked) and all(item is not None and item[0] == run_id for item in sequences)
        and {item[1] for item in sequences if item is not None} == set(range(len(acked)))
        and not number(summary.get("failed_messages"))
    )
    covered = (count == len(acked) and main.get("min_sequence") == 0
               and main.get("max_sequence") == len(acked) - 1)
    all_main = result["count_is_exact"] and manifest_matches and covered
    result["all_main_images_completed"] = all_main
    result["full_pipeline_completion_verified_for_run"] = all_main and not summary.get("keep_pano")
    if not manifest_matches:
        reasons.append("ACK manifest is not a contiguous, fully ACKed set of traced sequences")
    if summary.get("keep_pano"):
        reasons.append("pano expectations are unknown; only main-frame coverage can be verified")
    last_completion = number(main.get("last_completed_unix_ms")) / 1000
    if first_sent is not None and last_completion >= first_sent and count > 0 and result["count_is_exact"]:
        span = last_completion - first_sent
        result["first_send_to_last_completion_seconds"] = span
        result["observed_images_per_second"] = count / span if span > 0 else None
        if result["full_pipeline_completion_verified_for_run"] and last_ack is not None:
            result["run_business_drain_seconds"] = max(0.0, last_completion - last_ack)
    elif count > 0:
        reasons.append("completion time precedes first send or run measurement is unresolved")
        result["full_pipeline_completion_verified_for_run"] = False
    return result


def analyze_run(run_dir: Path, *, log_files: list[Path] | None = None) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    sent = read_jsonl(run_dir / "sent.jsonl")
    outputs = read_jsonl(run_dir / "outputs.jsonl")
    # 重复导出的 ACK/输出不重复计图。
    acked = {str(row["uId"]): row for row in sent
             if row.get("status") == "acked" and row.get("uId")}
    first_sent = min((number(row.get("sent_at")) for row in acked.values()), default=None)
    last_ack = max((number(row.get("ack_at")) for row in acked.values()), default=None)
    ack_span = last_ack - first_sent if first_sent is not None and last_ack is not None else 0
    pictures = {}
    for row in outputs:
        uid = str(row.get("uId", ""))
        if uid not in acked or row.get("targetType") != "picture":
            continue
        key = (uid, bool(row.get("isPanoKeyframe")))
        received = number(row.get("received_at"))
        if received > 0 and (key not in pictures or received < pictures[key]):
            pictures[key] = received
    latest_picture = max(pictures.values(), default=None)
    output_span = latest_picture - first_sent if latest_picture and first_sent else 0
    # pano 是否独立存在不能由 keep_pano 推断；仅默认单主图时有确切预期数。
    expected_pictures = len(acked) if not summary.get("keep_pano") else None
    all_main_pictures_seen = bool(acked) and all((uid, False) in pictures for uid in acked)
    rows = load_summaries(summary, log_files or [])
    operations = {name: aggregate_timings([row.get("operations", {}).get(name, {}) for row in rows])
                  for name in sorted({name for row in rows for name in row.get("operations", {})})}
    models = {}
    for model in sorted({model for row in rows for model in row.get("inference_by_model", {})}):
        values = [row.get("inference_by_model", {}).get(model, {}) for row in rows]
        batches = sum(int(number(value.get("batches"))) for value in values)
        samples = sum(int(number(value.get("samples"))) for value in values)
        models[model] = {
            "batches": batches, "samples": samples,
            "samples_per_batch": samples / batches if batches else None,
            "errors": sum(int(number(value.get("errors"))) for value in values),
            **{name: aggregate_timings([value.get(name, {}) for value in values])
               for name in ("batch_wait_ms", "executor_queue_wait_ms", "rpc_ms")},
        }
    return {
        "run_id": summary.get("run_id"), "result_directory": str(run_dir.resolve()),
        "target_input_qps": summary.get("target_qps"), "topic": summary.get("topic"),
        "input_broker_ack": {
            "acked_messages": len(acked), "failed_messages": summary.get("failed_messages"),
            "first_send_to_last_ack_seconds": ack_span,
            "messages_per_second": len(acked) / ack_span if ack_span > 0 else None,
            "latency_ms": distribution([
                max(0.0, (number(row.get("ack_at")) - number(row.get("sent_at"))) * 1000)
                for row in acked.values()]),
        },
        "picture_vector_persisted_output": {
            "scope": "run UID; output observed after picture-vector persistence; not full frame",
            "observed_images": len(pictures), "expected_images": expected_pictures,
            "all_main_images_observed": all_main_pictures_seen,
            "first_send_to_last_output_seconds": output_span if pictures else None,
            "observed_images_per_second": len(pictures) / output_span if output_span > 0 else None,
            "after_last_ack_seconds": max(0.0, latest_picture - last_ack)
            if latest_picture and last_ack else None,
            "latency_ms": distribution([
                max(0.0, (received - number(acked[uid].get("sent_at"))) * 1000)
                for (uid, _), received in pictures.items()]),
        },
        "business_images_by_topic": summarize_business(
            rows, topic=summary.get("topic", ""), run_start=number(summary.get("started_at")),
            run_end=number(summary.get("finished_at")), last_ack=last_ack),
        "business_images_by_run": summarize_run_completions(
            rows, summary=summary, acked=acked, first_sent=first_sent, last_ack=last_ack),
        "operations": operations, "inference_by_model": models,
        "notes": [
            "输入 broker ACK 的速度不是图片业务完成 QPS。短批次 N/(N-1) 的采样边界会使其略高于目标。",
            "整图向量先于人脸、人型完成时即可落库和输出，不能用于证明完整业务排空。",
            "Topic 周期计数包含其他输入；显式 trace 模式由有界 run 累计值去重验证主图完成，旧 UUID 模式不可恢复。",
            "periods 保留实际周期长度；不把跨边界周期的计数按时间比例摊进回放窗口。",
            "active_period_span 排除首尾空闲周期，包含中间空闲；短样本结果不是持续容量。",
            "这些结果不能单独证明 Kafka lag、内部队列或异步持久化队列已经排空。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="含 summary/sent/outputs 的回放结果目录")
    parser.add_argument("--log-file", action="append", type=Path, default=[],
                        help="可重复；补充当前或轮转 ALog 日志，按回放窗口过滤并去重")
    parser.add_argument("--output", type=Path, help="保存完整 JSON 比较报告；不修改原始结果")
    args = parser.parse_args()
    result = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "runs": [analyze_run(path, log_files=args.log_file) for path in args.run_dirs]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 默认显示紧凑口径，完整逐周期和模型耗时留在 JSON 文件。
    for run in result["runs"]:
        business = run["business_images_by_topic"]
        peak = business["peak_period"]
        print(json.dumps({
            "run_id": run["run_id"], "target_input_qps": run["target_input_qps"],
            "broker_acked": run["input_broker_ack"]["acked_messages"],
            "picture_persisted_outputs": run["picture_vector_persisted_output"]["observed_images"],
            "topic_business_images": business["images_in_captured_periods"],
            "topic_business_active_span_qps": business["active_period_span_images_per_second"],
            "topic_business_peak_period_qps": peak["images_per_second"] if peak else None,
            "business_images_by_run": run["business_images_by_run"],
            "full_pipeline_completion_verified_for_run": run["business_images_by_run"][
                "full_pipeline_completion_verified_for_run"],
            "notes": run["notes"],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
