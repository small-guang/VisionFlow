"""只读 Linux I/O 观测；不导入业务配置、不调用 Docker、不向被观察进程发信号。"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from typing import Any


DISK_FIELDS = (
    "reads", "reads_merged", "read_sectors", "read_ms", "writes", "writes_merged",
    "write_sectors", "write_ms", "inflight", "busy_ms", "weighted_ms",
    "discards", "discards_merged", "discard_sectors", "discard_ms", "flushes", "flush_ms",
)
VM_FIELDS = {"pswpin", "pswpout", "pgpgin", "pgpgout", "pgmajfault", "nr_dirty", "nr_writeback", "nr_writeback_temp"}


def parse_diskstats(text: str, devices: set[str] | None = None) -> dict:
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 14 or (devices is not None and parts[2] not in devices):
            continue
        try:
            values = [int(value) for value in parts[3:]]
            result[parts[2]] = {"major": int(parts[0]), "minor": int(parts[1]), **dict(zip(DISK_FIELDS, values))}
        except ValueError:
            continue
    return result


def disk_delta(before: dict, after: dict, seconds: float) -> dict:
    """由同一设备的累计计数求速率；不把 in-flight 下降当作计数回退。"""
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        return {"valid": False, "reason": "nonpositive_or_invalid_window"}
    for row in (before, after):
        if any(type(row.get(key)) is not int or row[key] < 0 for key in ("major", "minor")):
            return {"valid": False, "reason": "missing_or_invalid_device_identity"}
    if (before.get("major"), before.get("minor")) != (after.get("major"), after.get("minor")):
        return {"valid": False, "reason": "device_identity_changed"}
    required = set(DISK_FIELDS[:11]) - {"inflight"}
    if not required <= before.keys() or not required <= after.keys():
        return {"valid": False, "reason": "missing_counters"}
    present_before = set(DISK_FIELDS) & before.keys()
    present_after = set(DISK_FIELDS) & after.keys()
    if present_before != present_after:
        return {"valid": False, "reason": "counter_schema_changed"}
    if ("flushes" in present_before) != ("flush_ms" in present_before):
        return {"valid": False, "reason": "missing_counters"}
    if any(type(row[key]) is not int or row[key] < 0 for row in (before, after) for key in present_before):
        return {"valid": False, "reason": "invalid_counters"}
    changes = {key: after[key] - before[key] for key in DISK_FIELDS if key != "inflight" and key in before and key in after}
    reset = [key for key, value in changes.items() if value < 0]
    if reset:
        return {"valid": False, "reason": "counter_reset", "fields": reset}
    mean = lambda total, count: changes[total] / changes[count] if changes.get(count) else None
    return {
        "valid": True, "seconds": seconds, "delta": changes,
        "read_iops": changes["reads"] / seconds, "write_iops": changes["writes"] / seconds,
        "read_mib_s": changes["read_sectors"] / 2048 / seconds,
        "write_mib_s": changes["write_sectors"] / 2048 / seconds,
        "read_await_ms": mean("read_ms", "reads"), "write_await_ms": mean("write_ms", "writes"),
        "flush_per_s": changes["flushes"] / seconds if "flushes" in changes else None,
        "flush_await_ms": mean("flush_ms", "flushes") if "flush_ms" in changes else None,
        "average_queue": changes["weighted_ms"] / 1000 / seconds,
        "util_percent": changes["busy_ms"] / 10 / seconds,
    }


def pipeline_window(before: dict, after: dict) -> dict:
    """仅同一 session 的两次快照求完成速度；图像计数可能包含 pano。"""
    if not before.get("session_id") or before.get("session_id") != after.get("session_id") or before.get("pid") != after.get("pid"):
        return {"valid": False, "reason": "session_changed"}
    if any(type(row.get("pid")) is not int or row["pid"] <= 0 for row in (before, after)):
        return {"valid": False, "reason": "missing_or_invalid_pid"}
    try:
        images = [row["counters"]["image_pipeline_completed"]["total"] for row in (before, after)]
        messages = [row["stages"]["message_processing"]["completed_total"] for row in (before, after)]
    except (KeyError, TypeError):
        return {"valid": False, "reason": "missing_counters"}
    try:
        start, end = (datetime.fromisoformat(row["timestamp"]) for row in (before, after))
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("timezone required")
        seconds = (end - start).total_seconds()
        if seconds <= 0 or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in images + messages):
            raise ValueError("invalid counter or window")
    except (KeyError, TypeError, ValueError):
        return {"valid": False, "reason": "invalid_window_or_counter"}
    if images[1] < images[0] or messages[1] < messages[0]:
        return {"valid": False, "reason": "counter_reset"}
    return {"valid": True, "start_timestamp": before["timestamp"], "end_timestamp": after["timestamp"],
            "seconds": seconds, "image_completed_delta": images[1] - images[0],
            "image_completed_per_second": (images[1] - images[0]) / seconds,
            "message_completed_delta": messages[1] - messages[0],
            "message_completed_per_second": (messages[1] - messages[0]) / seconds,
            "image_counter_may_include_pano": True, "not_output_persistence_completion": True}


class JsonlTail:
    """有界增量尾读；保留半行，识别 inode 轮转和原地重写，首读只返回最新快照。"""

    def __init__(self, path: Path, max_bytes: int = 2 * 1024 * 1024):
        self.path = path
        self.max_bytes = max_bytes
        self.identity = None
        self.offset = 0
        self.pending = b""
        self.anchor = b""

    def poll(self) -> tuple[list[dict], dict]:
        state: dict[str, Any] = {"path": str(self.path)}
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                reset = self.identity != identity or stat.st_size < self.offset
                if not reset and self.anchor:
                    stream.seek(self.offset - len(self.anchor))
                    reset = stream.read(len(self.anchor)) != self.anchor
                initial = self.identity is None
                if reset:
                    state["reset"] = "initial" if initial else "rotation_or_rewrite"
                    self.offset = max(0, stat.st_size - self.max_bytes)
                    self.pending = b""
                elif stat.st_size - self.offset > self.max_bytes:
                    self.offset = stat.st_size - self.max_bytes
                    self.pending = b""
                    state["gap"] = "tail_limit_exceeded"
                discard_prefix = self.offset > 0 and (reset or "gap" in state)
                stream.seek(self.offset)
                data = stream.read(self.max_bytes)
                self.offset = stream.tell()
                self.identity = identity
                stream.seek(max(0, self.offset - 64))
                self.anchor = stream.read(min(64, self.offset))
            if discard_prefix:
                data = data.split(b"\n", 1)[1] if b"\n" in data else b""
            complete = self.pending + data
            lines = complete.split(b"\n")
            self.pending = lines.pop()
            if len(self.pending) > self.max_bytes:
                self.pending = b""
                state["gap"] = "oversized_unterminated_line"
            records = []
            bad = 0
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                    else:
                        bad += 1
                except (ValueError, UnicodeError):
                    bad += 1
            state.update(offset=self.offset, partial_bytes=len(self.pending), invalid_lines=bad)
            return (records[-1:] if initial else records), state
        except OSError as error:
            state["error"] = {"type": type(error).__name__, "errno": error.errno}
            return [], state


def parse_key_values(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        fields = line.replace(":", " ", 1).split()
        if len(fields) >= 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return result


def parse_pressure(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        values = {}
        for item in fields[1:]:
            try:
                key, raw = item.split("=", 1)
                values[key] = int(raw) if key == "total" else float(raw)
            except ValueError:
                continue
        result[fields[0]] = values
    return result


def parse_io_stat(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"\d+:\d+", fields[0]):
            continue
        values = {}
        for item in fields[1:]:
            try:
                key, raw = item.split("=", 1)
                values[key] = int(raw)
            except ValueError:
                continue
        result[fields[0]] = values
    return result


class IOObserver:
    def __init__(self, roots: dict[str, int], proc: Path = Path("/proc"), cgroup: Path = Path("/sys/fs/cgroup"), sysfs: Path = Path("/sys"), disk_devices: set[str] | None = None):
        self.proc = proc
        self.disk_devices = {"sda", "mmcblk0"} if disk_devices is None else set(disk_devices)
        self.cgroup = cgroup
        self.roots = roots
        self.root_identities: dict[str, dict | None] = {}
        self.tracked: dict[int, int] = {}
        self.cgroups: set[Path] = set()
        self.next_tree_refresh = 0.0
        self.errors: list[dict] = []
        for name, pid in roots.items():
            info = self.process_stat(pid)
            self.root_identities[name] = info
            if info:
                self.tracked[pid] = info["start_ticks"]
        gpu_patterns = [str(sysfs / pattern) for pattern in ("devices/platform/*gpu*/load", "devices/*gpu*/load", "devices/platform/*ga10b*/load")]
        import glob
        self.gpu_paths = sorted({Path(p) for pattern in gpu_patterns for p in glob.glob(pattern)})
        for path in (sysfs / "class/devfreq").glob("*"):
            if any(part in path.name.lower() for part in ("gpu", "ga10b", "gv11b")):
                self.gpu_paths.extend([path / "cur_freq", path / "device/load"])

    def read(self, path: Path) -> str | None:
        try:
            return path.read_text()
        except OSError as error:
            self.errors.append({"path": str(path), "type": type(error).__name__, "errno": error.errno})
            return None

    def process_stat(self, pid: int) -> dict | None:
        text = self.read(self.proc / str(pid) / "stat")
        if text is None:
            return None
        try:
            fields = text.rsplit(")", 1)[1].split()
            return {"pid": pid, "ppid": int(fields[1]), "state": fields[0], "start_ticks": int(fields[19]),
                    "cpu_ticks": int(fields[11]) + int(fields[12]), "rss_pages": int(fields[21]),
                    "stat_monotonic": time.monotonic()}
        except (IndexError, ValueError):
            self.errors.append({"path": str(self.proc / str(pid) / "stat"), "type": "MalformedStat"})
            return None

    def refresh_tree(self, now: float) -> None:
        if now < self.next_tree_refresh:
            return
        self.next_tree_refresh = now + 10
        processes = {}
        for path in self.proc.iterdir():
            if path.name.isdigit():
                info = self.process_stat(int(path.name))
                if info:
                    processes[info["pid"]] = info
        selected = {pid for pid, start in self.tracked.items() if pid in processes and processes[pid]["start_ticks"] == start}
        while True:
            children = {pid for pid, info in processes.items() if info["ppid"] in selected}
            new = children - selected
            if not new:
                break
            selected.update(new)
        self.tracked.update({pid: processes[pid]["start_ticks"] for pid in selected})
        self.cgroups = set((self.cgroup / "system.slice").glob("docker-*.scope"))

    def collect_cgroup(self, path: Path) -> dict:
        result = {"path": str(path), "timestamp": time.time(), "monotonic": time.monotonic(), "availability": {}}
        for name in ["io.stat", "memory.current", "memory.stat", "memory.events", "memory.pressure", "cpu.stat", "cpu.pressure", "io.pressure"]:
            raw = self.read(path / name)
            result["availability"][name] = raw is not None
            if raw is None:
                continue
            if name == "io.stat":
                result["io_stat_monotonic"] = time.monotonic()
                result[name] = parse_io_stat(raw)
            elif name.endswith("pressure"):
                result[name] = parse_pressure(raw)
            elif name == "memory.current":
                try:
                    result[name] = int(raw)
                except ValueError:
                    result["availability"][name] = False
                    self.errors.append({"path": str(path / name), "type": "MalformedCounter"})
            else:
                result[name] = parse_key_values(raw)
        return result

    def sample(self, monotonic: float) -> dict:
        self.errors = []
        timestamp = time.time()
        # 块设备时间点单独保存，避免把10秒一次的树扫描耗时算入错位的差值窗口。
        disk_raw = self.read(self.proc / "diskstats")
        disk_monotonic = time.monotonic()
        disk_timestamp = time.time()
        self.refresh_tree(monotonic)
        processes = []
        groups = set(self.cgroups)
        for pid, expected in self.tracked.items():
            info = self.process_stat(pid)
            if not info:
                continue
            if info["start_ticks"] != expected:
                self.errors.append({"pid": pid, "type": "PidReused", "expected_start_ticks": expected, "observed_start_ticks": info["start_ticks"]})
                continue
            raw = self.read(self.proc / str(pid) / "io")
            info["io_available"] = raw is not None
            if raw is not None:
                info["io_monotonic"] = time.monotonic()
                info["io"] = parse_key_values(raw)
            raw = self.read(self.proc / str(pid) / "cgroup")
            if raw:
                for line in raw.splitlines():
                    if line.startswith("0::"):
                        relative = line[3:].lstrip("/")
                        if ".." not in Path(relative).parts:
                            group = self.cgroup / relative
                            groups.add(group)
                            info["cgroup_v2"] = str(group)
            processes.append(info)
        stat = self.read(self.proc / "stat")
        cpu_monotonic = time.monotonic()
        cpu = {}
        if stat:
            for line in stat.splitlines():
                fields = line.split()
                if fields and re.fullmatch(r"cpu\d*", fields[0]):
                    cpu[fields[0]] = [int(value) for value in fields[1:]]
        vm = parse_key_values(self.read(self.proc / "vmstat") or "")
        mem = parse_key_values(self.read(self.proc / "meminfo") or "")
        memory_monotonic = time.monotonic()
        gpu = {}
        for path in self.gpu_paths:
            raw = self.read(path)
            if raw is not None:
                try:
                    gpu[str(path)] = int(raw)
                except ValueError:
                    pass
        gpu_monotonic = time.monotonic()
        pressure = {name: parse_pressure(self.read(self.proc / "pressure" / name) or "") for name in ("cpu", "memory", "io")}
        disks = parse_diskstats(disk_raw or "", self.disk_devices)
        snapshot = {"timestamp": timestamp, "monotonic": monotonic,
                "disk_monotonic": disk_monotonic, "disk_timestamp": disk_timestamp,
                "diskstats_available": disk_raw is not None,
                "disks": disks, "missing_disk_devices": sorted(self.disk_devices - disks.keys()),
                "cpu": cpu, "cpu_monotonic": cpu_monotonic,
                "vmstat": {k: v for k, v in vm.items() if k in VM_FIELDS}, "memory_monotonic": memory_monotonic,
                "meminfo_kib": {k: v for k, v in mem.items() if k in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Dirty", "Writeback", "Cached", "Buffers"}},
                "gpu_sysfs_raw": gpu, "gpu_monotonic": gpu_monotonic, "pressure": pressure, "processes": processes,
                "cgroups": [self.collect_cgroup(path) for path in sorted(groups)], "errors": self.errors}
        snapshot["collection_finished_monotonic"] = time.monotonic()
        snapshot["collection_seconds"] = snapshot["collection_finished_monotonic"] - monotonic
        return snapshot


def positive_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("必须为有限正数")
    return value


def root_argument(raw: str) -> tuple[str, int]:
    try:
        name, value = raw.rsplit("=", 1)
        pid = int(value)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or pid <= 0:
            raise ValueError
        return name, pid
    except ValueError as error:
        raise argparse.ArgumentTypeError("使用 name=正整数PID") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=positive_number, default=180.0)
    parser.add_argument("--interval", type=positive_number, default=1.0)
    parser.add_argument("--pipeline-path", type=Path, action="append")
    parser.add_argument("--root", type=root_argument, action="append", default=[])
    parser.add_argument("--disk", action="append", default=[], help="额外采样的 /proc/diskstats 设备名，可重复；默认含 sda、mmcblk0")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.!-]*", name) for name in args.disk):
        parser.error("--disk 需要 /proc/diskstats 中的设备名，例如 nvme0n1，不接受 /dev/ 路径")
    disk_devices = {"sda", "mmcblk0", *args.disk}
    roots = dict(args.root)
    if len(roots) != len(args.root):
        parser.error("root名称不能重复")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("metadata.json", "resources.jsonl", "pipeline.jsonl", "summary.json")):
        parser.error("输出目录已经包含采集结果，请使用新目录以免混合会话")
    observer = IOObserver(roots, disk_devices=disk_devices) if args.disk else IOObserver(roots)
    tails = [JsonlTail(path.resolve()) for path in (args.pipeline_path or [Path("logs/pipeline_stats.jsonl")])]
    started = time.monotonic()
    metadata = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": args.seconds,
                "interval_seconds": args.interval, "requested_roots": roots,
                "roots": observer.root_identities, "initial_errors": observer.errors, "requested_disk_devices": sorted(disk_devices),
                "pipeline_paths": [str(tail.path) for tail in tails], "page_size": os.sysconf("SC_PAGE_SIZE"),
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"), "process_tree_refresh_seconds": 10,
                "read_only": True, "calls_docker": False, "reads_environment": False,
                "notes": ["磁盘为整机计数，cgroup含所有system.slice/docker-*.scope及指定PID所属组。", "cgroup/PID写字节不等于fsync延迟；父子cgroup不可直接相加。", "gpu load保留sysfs原值，Jetson load通常为千分比；cur_freq为Hz。", "图像完成计数可能包含pano，不代表异步输出持久化完成。", "采集文件本身也产生少量I/O；1秒flush Python缓冲但不执行fsync。"]}
    with (output / "metadata.json").open("x") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    samples = pipeline_samples = 0
    interrupted = False
    previous = None
    first_resource = last_resource = None
    first_disk = last_disk = None
    pipeline_bounds: dict[str, list[dict]] = {}
    print(json.dumps({"output": str(output), "seconds": args.seconds, "root_count": len(roots)}), flush=True)
    with (output / "resources.jsonl").open("x") as resources, (output / "pipeline.jsonl").open("x") as pipeline:
        try:
            deadline = started + args.seconds
            next_sample = started
            while True:
                now = time.monotonic()
                try:
                    snapshot = observer.sample(now)
                except (OSError, ValueError, IndexError) as error:
                    snapshot = {"timestamp": time.time(), "monotonic": now, "errors": [{"type": type(error).__name__, "scope": "sample"}]}
                if first_resource is None:
                    first_resource = snapshot["monotonic"]
                last_resource = snapshot["monotonic"]
                if snapshot.get("diskstats_available"):
                    if first_disk is None:
                        first_disk = snapshot["disk_monotonic"]
                    last_disk = snapshot["disk_monotonic"]
                if previous and "disks" in snapshot and "disks" in previous:
                    snapshot["disk_interval"] = {name: disk_delta(previous["disks"][name], values, snapshot["disk_monotonic"] - previous["disk_monotonic"])
                                                  for name, values in snapshot["disks"].items() if name in previous["disks"]}
                previous = snapshot
                snapshot["pipeline_tail"] = []
                for tail in tails:
                    records, status = tail.poll()
                    snapshot["pipeline_tail"].append(status)
                    for record in records:
                        key = str(tail.path)
                        if key not in pipeline_bounds:
                            pipeline_bounds[key] = [record, record]
                        else:
                            pipeline_bounds[key][1] = record
                        pipeline.write(json.dumps({"observed_at": time.time(), "source_path": str(tail.path), "snapshot": record}, ensure_ascii=False) + "\n")
                        pipeline_samples += 1
                resources.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
                resources.flush()
                pipeline.flush()
                samples += 1
                if now >= deadline:
                    break
                next_sample = max(next_sample + args.interval, time.monotonic())
                time.sleep(max(0, min(next_sample, deadline) - time.monotonic()))
        except KeyboardInterrupt:
            interrupted = True
    summary = {"finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
               "resource_samples": samples, "pipeline_samples": pipeline_samples, "interrupted": interrupted,
               "resource_window_seconds": last_resource - first_resource if samples > 1 else None,
               "disk_window_seconds": last_disk - first_disk if first_disk is not None and last_disk > first_disk else None,
               "pipeline_windows": {path: pipeline_window(bounds[0], bounds[1]) for path, bounds in pipeline_bounds.items()},
               "window_note": "elapsed_seconds仅采集器生命周期，不能作为吞吐分母；disk_interval使用实际磁盘monotonic时间差，pipeline使用自身快照窗口，跨来源比较须另行对齐。",
               "service_signals_sent": False}
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
