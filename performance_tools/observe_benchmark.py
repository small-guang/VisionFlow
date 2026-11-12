"""只读采集压测窗口：复用 observe_io，并汇总 ALog 和 Prometheus 直方图。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

if __package__:
    from .observe_io import disk_delta
else:
    from observe_io import disk_delta

HERE = Path(__file__).resolve().parent
METRICS = {
    "face_search": ("milvus_proxy_collection_sq_latency", {"collection_name": "alogFace", "query_type": "search"}, "ms"),
    "sbs_search": ("milvus_proxy_collection_sq_latency", {"collection_name": "alogSBS", "query_type": "search"}, "ms"),
    "proxy_search_queue_all": ("milvus_proxy_req_in_queue_latency", {"function_name": "Search"}, "ms"),
    "querynode_search_queue_all": ("milvus_querynode_sq_queue_latency", {"query_type": "search"}, "ms"),
    "querynode_search_core_all": ("milvus_querynode_sq_core_latency", {"query_type": "search"}, "ms"),
    "querynode_search_tsafe_all": ("milvus_querynode_sq_wait_tsafe_latency", {"query_type": "search"}, "ms"),
    "etcd_wal_fsync": ("etcd_disk_wal_fsync_duration_seconds", {}, "seconds"),
    "etcd_backend_commit": ("etcd_disk_backend_commit_duration_seconds", {}, "seconds"),
}
OPERATIONS = ("face_identity_locked_search", "face_identity_locked_full_milvus", "face_identity_locked_query_admission",
              "face_identity_locked_query_batch", "face_identity_locked_query_executor", "face_identity_locked_query_sdk",
              "face_identity_locked_query_resume", "face_identity_lock_wait", "face_identity_lock_hold",
              "face_identity_snapshot_lock_wait", "sbs_search_thread_queue", "sbs_search_thread_run",
              "vector_search_queue_wait", "vector_search_run", "vector_query_rpc_run", "event_loop_lag")
GAUGES = {"milvus_datacoord_segment_num", "milvus_datacoord_task_count", "milvus_proxy_queue_task_num",
          "milvus_querycoord_task_num", "milvus_querynode_read_task_ready_len",
          "milvus_querynode_read_task_unsolved_len", "milvus_querynode_read_task_concurrency",
          "milvus_querynode_segment_num", "etcd_server_has_leader", "etcd_server_is_leader", "etcd_server_leader_changes_seen_total"}
LINE = re.compile(r'^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\S+)?$')
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"(?:,|$)')


def collection_metrics(face: str, sbs: str) -> dict:
    """为本次运行单独构造过滤器，避免污染全局默认指标。"""
    specs = {name: (prefix, dict(labels), unit) for name, (prefix, labels, unit) in METRICS.items()}
    specs["face_search"][1]["collection_name"] = face
    specs["sbs_search"][1]["collection_name"] = sbs
    return specs


def parse_prometheus(text: str) -> dict:
    result = {}
    prefixes = tuple(spec[0] + "_" for spec in METRICS.values())
    for line in text.splitlines():
        match = LINE.match(line)
        if not match or not (match[1].startswith(prefixes) or match[1] in GAUGES):
            continue
        try:
            value = float(match[3])
            if not math.isfinite(value):
                continue
        except ValueError:
            continue
        labels = dict(LABEL.findall(match[2] or ""))
        result[(match[1], tuple(sorted(labels.items())))] = value
    return result


def histogram_delta(before: dict, after: dict, prefix: str, required: dict, unit: str) -> dict:
    def select(data):
        return {key: value for key, value in data.items() if key[0] in {prefix + "_count", prefix + "_sum", prefix + "_bucket"}
                and all(dict(key[1]).get(k) == v for k, v in required.items())}
    a, b = select(before), select(after)
    if not a or not b:
        return {"available": False, "reason": "metric_missing"}
    if a.keys() != b.keys():
        return {"available": False, "reason": "series_changed"}
    delta = {key: b[key] - a[key] for key in a}
    if any(v < 0 for v in delta.values()):
        return {"available": False, "reason": "counter_reset"}
    if not any(name.endswith("_count") for name, _ in delta) or not any(name.endswith("_sum") for name, _ in delta):
        return {"available": False, "reason": "incomplete_histogram"}
    count = sum(v for (name, _), v in delta.items() if name.endswith("_count"))
    total = sum(v for (name, _), v in delta.items() if name.endswith("_sum"))
    buckets = {}
    for (name, labels), value in delta.items():
        if name.endswith("_bucket"):
            try:
                bound = float(dict(labels)["le"])
                buckets[bound] = buckets.get(bound, 0) + value
            except (KeyError, ValueError):
                pass
    bounds = {}
    for percentile in (50, 95, 99):
        val = next((le for le, n in sorted(buckets.items()) if n >= count * percentile / 100), None) if count else None
        bounds[str(percentile)] = val if val is not None and math.isfinite(val) else None
    return {"available": True, "count": count, "mean": total / count if count else None,
            "percentile_upper_bound": bounds, "unit": unit}


def aggregate_operations(rows: list[dict]) -> dict:
    result = {}
    for name in OPERATIONS:
        values = [row.get("operations", {}).get(name) for row in rows]
        values = [v for v in values if isinstance(v, dict)]
        count = sum(v.get("count", 0) for v in values)
        result[name] = {"count": count, "mean_ms": sum(v.get("total_ms", 0) for v in values) / count if count else None,
                        "max_ms": max((v.get("max_ms", 0) for v in values), default=None),
                        "errors": sum(v.get("errors", 0) for v in values)}
    return result


def read_app_log(path: Path, start: float, end: float) -> tuple[list[dict], str | None]:
    rows = []
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "ALog performance summary: " not in line:
                    continue
                try:
                    row = json.loads(line.split("ALog performance summary: ", 1)[1])
                    if row["interval_start_unix_ms"] >= start * 1000 and row["interval_end_unix_ms"] <= end * 1000:
                        rows.append(row)
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError as exc:
        return rows, type(exc).__name__
    return rows, None


def resource_summary(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    memory = [r["meminfo_kib"]["MemAvailable"] for r in rows if "MemAvailable" in r.get("meminfo_kib", {})]
    gpu = {}
    for row in rows:
        for key, value in row.get("gpu_sysfs_raw", {}).items():
            if key.endswith("/load"):
                gpu.setdefault(key, []).append(value)
    missing_disks = sorted({disk for row in rows for disk in row.get("missing_disk_devices", [])})
    result = {"samples": len(rows), "disk_devices_missing_in_at_least_one_sample": missing_disks, "memory_available_kib_mean": sum(memory) / len(memory) if memory else None,
              "gpu_load_raw_by_path": {key: {"mean": sum(values) / len(values), "samples": len(values)} for key, values in gpu.items()},
              "note": "GPU load 是 sysfs 原始点采样，Jetson 常为千分比；磁盘和进程数据详见 resources.jsonl。"}
    process_points = {}
    for row in rows:
        for proc in row.get("processes", []):
            key = (proc.get("pid"), proc.get("start_ticks"))
            process_points.setdefault(key, []).append(proc)
    result["processes"] = []
    for (pid, started), points in process_points.items():
        first_process, last_process = points[0], points[-1]
        seconds = last_process["stat_monotonic"] - first_process["stat_monotonic"]
        ticks = last_process["cpu_ticks"] - first_process["cpu_ticks"]
        result["processes"].append({"pid": pid, "start_ticks": started, "samples": len(points),
                                    "window_seconds": seconds,
                                    "cpu_cores_mean": ticks / os.sysconf("SC_CLK_TCK") / seconds if seconds > 0 and ticks >= 0 else None,
                                    "rss_last_bytes": last_process["rss_pages"] * os.sysconf("SC_PAGE_SIZE")})
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        a, b = first.get("cpu", {}).get("cpu", []), last.get("cpu", {}).get("cpu", [])
        if len(a) >= 5 and len(b) >= 5:
            # guest 时间已包含在 user/nice 中，避免重复计数。
            diff = [y-x for x, y in zip(a[:8], b[:8])]
            if all(n >= 0 for n in diff) and sum(diff):
                result["cpu_busy_percent"] = 100 * (sum(diff) - diff[3] - diff[4]) / sum(diff)
                result["cpu_iowait_percent"] = 100 * diff[4] / sum(diff)
        seconds_disk = last.get("disk_monotonic", 0) - first.get("disk_monotonic", 0)
        result["disks"] = {key: disk_delta(value, last.get("disks", {})[key], seconds_disk)
                           for key, value in first.get("disks", {}).items() if key in last.get("disks", {})}
        vm1, vm2 = first.get("vmstat", {}), last.get("vmstat", {})
        seconds = last.get("memory_monotonic", 0) - first.get("memory_monotonic", 0)
        for field in ("pswpin", "pswpout"):
            if seconds > 0 and field in vm1 and field in vm2 and vm2[field] >= vm1[field]:
                result[field + "_pages_s"] = (vm2[field] - vm1[field]) / seconds
    return result


def pipeline_queues(path: Path) -> dict:
    groups = {}
    invalid = 0
    for line in path.read_text().splitlines():
        record = json.loads(line)
        row = record["snapshot"]
        source, session, pid = record.get("source_path"), row.get("session_id"), row.get("pid")
        if not source or not session or type(pid) is not int or pid <= 0:
            invalid += 1
            continue
        groups.setdefault((source, session, pid), []).append(row)
    names = {"input_queued_messages": ("input", "queued_messages"),
             "messages_inflight": ("messages", "inflight_messages"),
             "search_pool_queued_jobs": ("resources", "vector_search", "queued_jobs"),
             "search_pool_inflight_jobs": ("resources", "vector_search", "inflight_jobs"),
             "search_pool_capacity_jobs": ("resources", "vector_search", "capacity_jobs")}
    result = {"invalid_identity_samples": invalid, "groups": []}
    for (source, session, pid), rows in groups.items():
        group = {"source": source, "session_id": session, "pid": pid, "samples": len(rows),
                 "first_timestamp": rows[0].get("timestamp"), "last_timestamp": rows[-1].get("timestamp"), "values": {}}
        for label, keys in names.items():
            values = []
            for row in rows:
                item = row.get("queues", {})
                for key in keys:
                    item = item.get(key, {}) if isinstance(item, dict) else None
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    values.append(item)
            group["values"][label] = {"first": values[0], "last": values[-1], "max": max(values), "samples": len(values)} if values else None
        result["groups"].append(group)
    return result


def positive(value: str) -> float:
    try:
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("必须为有限正数")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新建的结果目录")
    parser.add_argument("--seconds", type=positive, default=300)
    parser.add_argument("--interval", type=positive, default=5, help="Prometheus 采样间隔（秒）")
    parser.add_argument("--resource-interval", type=positive, default=1)
    parser.add_argument("--milvus-url", default="http://127.0.0.1:9091/metrics")
    parser.add_argument("--etcd-url", default=None, help="etcd /metrics 地址；不提供则跳过")
    parser.add_argument("--face-collection", default="alogFace", help="人脸集合名")
    parser.add_argument("--sbs-collection", default="alogSBS", help="SBS 集合名")
    parser.add_argument("--app-log", type=Path, default=Path("logs/ALog.log"))
    parser.add_argument("--pipeline-path", type=Path, action="append")
    parser.add_argument("--root", action="append", default=[], help="传给 observe_io 的 name=PID")
    parser.add_argument("--disk", action="append", default=[], help="额外块设备名，例如 nvme0n1；可重复")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.!-]*", disk) for disk in args.disk):
        parser.error("--disk 需要 /proc/diskstats 设备名，例如 nvme0n1")
    for url in (args.milvus_url, args.etcd_url):
        if url and (urlsplit(url).scheme not in ("http", "https") or urlsplit(url).username or urlsplit(url).password):
            parser.error("指标 URL 必须是无内嵌凭据的 HTTP(S) 地址")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("输出目录已存在，请使用新目录")
    cmd = [sys.executable, str(HERE / "observe_io.py"), "--output", str(output / "resources"),
           "--seconds", str(args.seconds), "--interval", str(args.resource_interval)]
    for path in args.pipeline_path or []:
        cmd.extend(("--pipeline-path", str(path)))
    for root in args.root:
        cmd.extend(("--root", root))
    for disk in args.disk:
        cmd.extend(("--disk", disk))
    start = time.time()
    monotonic_start = time.monotonic()
    print(f"采集开始：{datetime.fromtimestamp(start).astimezone().isoformat()}；输出 {output}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    samples = {name: [] for name in ("milvus", "etcd")}
    failures = {name: [] for name in samples}
    endpoints = {"milvus": args.milvus_url, "etcd": args.etcd_url}
    try:
        while True:
            for name, url in endpoints.items():
                if not url:
                    continue
                started = time.time()
                try:
                    with urlopen(url, timeout=3) as response:
                        body = response.read(20_000_000).decode("utf-8")
                    samples[name].append((started, parse_prometheus(body)))
                except (OSError, URLError, UnicodeError, ValueError) as exc:
                    failures[name].append({"timestamp": started, "error": type(exc).__name__})
            elapsed = time.monotonic() - monotonic_start
            if elapsed >= args.seconds:
                break
            time.sleep(min(args.interval, args.seconds - elapsed))
    except KeyboardInterrupt:
        proc.terminate()
    _, stderr = proc.communicate()
    end = time.time()
    if proc.returncode != 0:
        print("资源采集器未正常完成：" + stderr.strip(), file=sys.stderr)
    with (output / "prometheus.jsonl").open("x") as stream:
        for source, points in samples.items():
            for timestamp, values in points:
                record = {"source": source, "timestamp": timestamp, "metrics": [
                    {"name": name, "labels": dict(labels), "value": value} for (name, labels), value in values.items()]}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    prom = {}
    for label, (prefix, required, unit) in collection_metrics(args.face_collection, args.sbs_collection).items():
        kind = "etcd" if label.startswith("etcd_") else "milvus"
        points = samples[kind]
        if len(points) < 2:
            prom[label] = {"available": False, "reason": "insufficient_samples"}
        else:
            intervals = [histogram_delta(a[1], b[1], prefix, required, unit) for a, b in zip(points, points[1:])]
            invalid = next((v for v in intervals if not v["available"]), None)
            prom[label] = invalid or histogram_delta(points[0][1], points[-1][1], prefix, required, unit)
    rows, error = read_app_log(args.app_log, start, end)
    result = {"started_at": datetime.fromtimestamp(start, timezone.utc).isoformat(), "finished_at": datetime.fromtimestamp(end, timezone.utc).isoformat(),
              "app": {"log": str(args.app_log), "read_error": error, "complete_intervals": len(rows),
                      "intervals": [{"pid": r.get("pid"), "start_ms": r.get("interval_start_unix_ms"), "end_ms": r.get("interval_end_unix_ms")} for r in rows],
                      "operations_by_pid": {str(pid): aggregate_operations([r for r in rows if r.get("pid") == pid]) for pid in {r.get("pid") for r in rows}}},
              "prometheus": {"collections": {"face": args.face_collection, "sbs": args.sbs_collection}, "metrics": prom, "samples": {k: len(v) for k, v in samples.items()}, "failures": failures,
                             "windows": {k: [v[0][0], v[-1][0]] if len(v) >= 2 else None for k, v in samples.items()}},
              "resource_exit_code": proc.returncode,
              "notes": ["只汇总完整落在采集窗口中的 ALog 区间；日志轮转或采集刚好跨进程重启可能导致缺口，检查 intervals。",
                        "Prometheus 是区间累计差，P95/P99 是直方图桶上界；客户端和服务端指标未逐请求配对。",
                        "查询池、队列及吞吐见 resources/pipeline.jsonl 与 resources/summary.json；image 完成不代表异步输出持久化。",
                        "各次 Prometheus 白名单原始样本见 prometheus.jsonl；_all 为跨集合。资源详情与进程见 resources/resources.jsonl。",
                        "资源采样和指标采样窗口各有边界，请按各自时间对齐；采样会带来少量 I/O。"]}
    try:
        result["resources"] = resource_summary(output / "resources" / "resources.jsonl")
        result["pipeline"] = {"windows": json.loads((output / "resources" / "summary.json").read_text())["pipeline_windows"],
                              "queue_snapshots": pipeline_queues(output / "resources" / "pipeline.jsonl")}
    except (OSError, ValueError, KeyError) as exc:
        result["resources"] = {"error": type(exc).__name__}
    (output / "benchmark_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"采集结束；汇总：{output / 'benchmark_summary.json'}", flush=True)
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
