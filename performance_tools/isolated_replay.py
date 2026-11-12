"""为当前重构版本准备隔离回放。plan 离线；其他子命令显式连接服务。"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_INPUT = PROJECT_ROOT / "performance_tools/results/qps6-snapshot/snapshot.jsonl"
OUTPUT_ENV = {
    "face_detection": "KAFKA_FACE_DETECTION_RECORD_TOPIC",
    "face_recognition": "KAFKA_FACE_RECOGNITION_RECORD_TOPIC",
    "person_detection": "KAFKA_PERSON_DETECTION_RECORD_TOPIC",
    "person_recognition": "KAFKA_PERSON_RECOGNITION_RECORD_TOPIC",
    "vector_record": "KAFKA_VECTOR_RECORD_TOPIC",
}
COLLECTIONS = {"face": "alogFace", "picture": "alogPictureClip", "sbs": "alogSBS", "tbps": "alogTBPS"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Any) -> bytes:
    def plain(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(type(item).__name__)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=plain) + "\n").encode()


def save_new(path: Path, value: Any, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encode(value))


def isolation_names(namespace: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", namespace):
        raise ValueError("namespace 必须是 1..48 位小写字母/数字/下划线，首位为字母")
    prefix = f"alogbench_{namespace}_"
    return {
        "input_topic": f"alogbench.{namespace}.input",
        "output_topics": {name: f"alogbench.{namespace}.{name}" for name in OUTPUT_ENV},
        "group_id": f"alogbench.{namespace}.worker",
        "collection_prefix": prefix,
        "collections": {name: prefix + suffix for name, suffix in COLLECTIONS.items()},
        "storage_prefix": f"alogbench/{namespace}",
    }


def isolation_env(manifest: dict[str, Any], directory: Path) -> dict[str, str]:
    names = isolation_names(manifest["namespace"])
    if manifest.get("schema") != 1 or manifest.get("names") != names:
        raise ValueError("隔离manifest命名不匹配，拒绝接受外部Topic/图库覆盖")
    return {
        "KAFKA_INPUT_TOPICS": names["input_topic"], "KAFKA_GROUP_ID": names["group_id"],
        "KAFKA_AUTO_OFFSET_RESET": "earliest", "MILVUS_COLLECTION_PREFIX": names["collection_prefix"],
        "STORAGE_OUTPUT_PREFIX": names["storage_prefix"],
        "PIPELINE_STATS_ENABLED": "true", "PIPELINE_STATS_PATH": str(directory.resolve() / "pipeline_stats.jsonl"),
        **{env: names["output_topics"][name] for name, env in OUTPUT_ENV.items()},
    }


def load_manifest(path: Path, *, prepared: bool = False) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    isolation_env(manifest, path.parent)
    if prepared:
        proof = json.loads((path.parent / "prepared.json").read_text())
        if proof.get("manifest_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("prepared.json 与隔离manifest不匹配")
    return manifest


def input_summary(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get("payload"), dict) for row in rows):
        raise ValueError("输入快照必须包含有效的payload对象")
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)) if path.resolve().is_relative_to(PROJECT_ROOT) else str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(), "messages": len(rows),
        "pano_messages": sum(bool(row["payload"].get("panoKeyframeUrl")) for row in rows),
        "image_availability_verified": False,
    }


def plan(args) -> None:
    namespace = args.namespace or f"r{datetime.now(timezone.utc):%Y%m%d}_{uuid.uuid4().hex[:10]}"
    manifest = {"schema": 1, "namespace": namespace, "created_at": now(),
                "names": isolation_names(namespace), "input": input_summary(args.input),
                "notes": ["仅当前重构版本；原图对象key和事件时间保留", "未验证历史图片可下载；未连接任何服务"]}
    directory = args.output_dir / namespace
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "manifest.json"
    save_new(path, manifest)
    print(json.dumps({"manifest": str(path), "namespace": namespace, "names": manifest["names"], "input": manifest["input"]}, ensure_ascii=False, indent=2))


def capture_gallery(client, name: str, limit: int) -> dict[str, Any]:
    """有限Strong读取；超过上限拒绝截断成不完整暖图库。"""
    description = client.describe_collection(collection_name=name)
    fields = description.get("fields", [])
    primary = next((field for field in fields if field.get("is_primary")), None)
    if primary is None or description.get("auto_id") or primary.get("auto_id"):
        raise ValueError("暖图库需要当前手工主键schema")
    rows = client.query(collection_name=name, filter="", output_fields=[field["name"] for field in fields],
                        limit=limit + 1, consistency_level="Strong", timeout=30)
    if len(rows) > limit:
        raise ValueError("源图库超过读取上限；请核对后显式提高limit，不能静默截断")
    keys = [row.get(primary["name"]) for row in rows]
    if any(key is None for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("源图库主键缺失或重复")
    return {"source_collection": name, "primary_key": primary["name"],
            "fields": fields, "row_count": len(rows), "rows": rows,
            "rows_sha256": hashlib.sha256(encode(rows)).hexdigest()}


def snapshot_gallery(args) -> None:
    from core.config import settings
    from pymilvus import MilvusClient
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.face_source == args.sbs_source:
        raise ValueError("Face和SBS源集合不能相同")
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    started = now()
    try:
        galleries = {"face": capture_gallery(client, args.face_source, args.face_limit),
                     "sbs": capture_gallery(client, args.sbs_source, args.sbs_limit)}
    finally:
        client.close()
    snapshot = {"schema": 1, "started_at": started, "finished_at": now(), "galleries": galleries,
                "notes": ["两次Strong有限查询，不提供跨集合事务快照", "原主键、身份、向量、图片key和事件时间均保留"]}
    save_new(args.output, snapshot, private=True)
    print(json.dumps({"snapshot": str(args.output), "counts": {name: item["row_count"] for name, item in galleries.items()},
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False))


def load_warm_snapshot(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("暖图库快照超过32MiB上限")
    snapshot = json.loads(path.read_text())
    if snapshot.get("schema") != 1 or set(snapshot.get("galleries", {})) != {"face", "sbs"}:
        raise ValueError("不兼容的暖图库快照")
    for item in snapshot["galleries"].values():
        if item.get("row_count") != len(item.get("rows", [])):
            raise ValueError("暖图库快照行数不匹配")
        if item.get("rows_sha256") != hashlib.sha256(encode(item["rows"])).hexdigest():
            raise ValueError("暖图库快照内容校验失败")
    return snapshot


def seed_galleries(databases, snapshot: dict[str, Any] | None, names: dict[str, Any]) -> dict[str, int]:
    if snapshot is None:
        return {"face": 0, "sbs": 0}
    counts = {}
    for kind, item in snapshot["galleries"].items():
        target = getattr(databases, kind)
        destination = names["collections"][kind]
        if target.collection_name != destination or not destination.startswith(names["collection_prefix"]):
            raise ValueError("目标数据库不属于本轮隔离前缀")
        if destination == item["source_collection"]:
            raise ValueError("禁止向源图库写入")
        rows, primary = item["rows"], item["primary_key"]
        if target.client.query(collection_name=destination, filter="", output_fields=[primary], limit=1, consistency_level="Strong"):
            raise ValueError("隔离目标图库必须为空；禁止覆盖或混入上一轮")
        if rows:
            result = target.client.insert(collection_name=destination, data=rows)
            if result.get("insert_count") != len(rows):
                raise RuntimeError("暖图库写入回执行数不匹配")
            target.client.flush(collection_name=destination)
        observed = target.client.query(collection_name=destination, filter="", output_fields=[primary], limit=len(rows) + 1, consistency_level="Strong")
        if {row[primary] for row in observed} != {row[primary] for row in rows} or len(observed) != len(rows):
            raise RuntimeError("暖图库主键验证失败")
        counts[kind] = len(rows)
    return counts


async def create_topics(config, names: dict[str, Any], *, partitions: int, replication_factor: int, admin_factory=None) -> None:
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    admin = (admin_factory or AIOKafkaAdminClient)(**config.kafka_client_config)
    topics = [names["input_topic"], *names["output_topics"].values()]
    try:
        await admin.start()
        if set(topics) & set(await admin.list_topics()):
            raise ValueError("隔离Topic已存在；请使用新的namespace，不复用历史输入")
        response = await admin.create_topics([NewTopic(topic, partitions, replication_factor) for topic in topics])
        if any(row[1] for row in getattr(response, "topic_errors", [])):
            raise RuntimeError("Kafka拒绝创建部分隔离Topic")
        if not set(topics).issubset(await admin.list_topics()):
            raise RuntimeError("未找到全部隔离Topic")
    finally:
        await admin.close()


def apply_manifest_environment(path: Path, *, prepared: bool = False) -> tuple[dict[str, Any], Any]:
    manifest = load_manifest(path, prepared=prepared)
    # 在首次导入settings之前设置隔离项，避免已加载单例保留生产输出配置。
    if "core.config" in sys.modules:
        raise RuntimeError("隔离入口必须在新的Python进程中执行")
    os.environ.update(isolation_env(manifest, path.parent))
    from core.config import settings
    return manifest, settings


def prepare(args) -> None:
    manifest, config = apply_manifest_environment(args.manifest)
    proof_path = args.manifest.parent / "prepared.json"
    if proof_path.exists():
        raise FileExistsError(proof_path)
    warm = load_warm_snapshot(args.warm_snapshot)
    from pymilvus import MilvusClient
    client = MilvusClient(uri=config.milvus_uri, token=config.milvus_token)
    try:
        if any(client.has_collection(collection_name=name) for name in manifest["names"]["collections"].values()):
            raise ValueError("隔离集合已存在；请使用新的namespace，不重置现有图库")
    finally:
        client.close()
    asyncio.run(create_topics(config, manifest["names"], partitions=args.partitions, replication_factor=args.replication_factor))
    counts = asyncio.run(_prepare_galleries(config, warm, manifest["names"]))
    proof = {"schema": 1, "prepared_at": now(), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
             "warm_rows": counts, "warm_snapshot_sha256": hashlib.sha256(args.warm_snapshot.read_bytes()).hexdigest() if args.warm_snapshot else None,
             "partitions": args.partitions, "replication_factor": args.replication_factor,
             "notes": ["仅创建新命名空间；不删除、不覆盖、不修改生产offset", "失败可能留下本轮部分资源，请换新namespace，脚本不自动删库"]}
    save_new(proof_path, proof)
    print(json.dumps(proof, ensure_ascii=False, indent=2))


async def _prepare_galleries(config, warm, names):
    from core.execution import OwnerExecutionResources
    from db.zilliz import DatabaseServices

    databases = None
    try:
        async with OwnerExecutionResources(search_workers=1, write_workers=1) as resources:
            databases = DatabaseServices.create(config, resources)
            return seed_galleries(databases, warm, names)
    finally:
        # async with 先排空查询协调器和执行池，随后才关闭它们使用的客户端。
        if databases is not None:
            databases.close()


def worker(args) -> None:
    manifest = load_manifest(args.manifest, prepared=True)
    environment = {**os.environ, **isolation_env(manifest, args.manifest.parent)}
    # main的常规日志是相对路径；独立cwd使生产日志也不被第二Worker混入。
    directory = args.manifest.parent.resolve()
    print(json.dumps({"worker_directory": str(directory), "group": manifest["names"]["group_id"], "input_topic": manifest["names"]["input_topic"]}), flush=True)
    os.chdir(directory)
    os.execvpe(sys.executable, [sys.executable, str(PROJECT_ROOT / "main.py")], environment)


def replay(args) -> None:
    manifest, _config = apply_manifest_environment(args.manifest, prepared=True)
    input_path = Path(manifest["input"]["path"])
    input_path = input_path if input_path.is_absolute() else PROJECT_ROOT / input_path
    if input_summary(input_path)["sha256"] != manifest["input"]["sha256"]:
        raise ValueError("输入快照内容已变化，拒绝使用不同输入进行本轮比较")
    from performance_tools.kafka_replay import build_parser as replay_parser, replay as send
    directory = args.manifest.parent.resolve()
    command = ["replay", "--input", str(input_path), "--topic", manifest["names"]["input_topic"],
               "--count", str(args.count), "--qps", str(args.qps), "--wait-seconds", str(args.wait_seconds),
               "--max-inflight", str(args.max_inflight), "--trace-completion", "--event-time", "preserve",
               "--log-file", str(directory / "logs/ALog.log"), "--output-dir", str(directory / "runs")]
    for metric in args.metrics:
        command.extend(["--metrics", metric])
    asyncio.run(send(replay_parser().parse_args(command)))


def positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("plan", help="离线生成隔离命名与输入哈希，不连接服务")
    offline.add_argument("--namespace")
    offline.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    offline.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "performance_tools/results/isolated")
    offline.set_defaults(handler=plan)
    capture = commands.add_parser("snapshot-gallery", help="只读有限导出生产Face/SBS，含业务数据的本地文件权限0600")
    capture.add_argument("--face-source", default="alogFace")
    capture.add_argument("--sbs-source", default="alogSBS")
    capture.add_argument("--face-limit", type=positive_int, default=20)
    capture.add_argument("--sbs-limit", type=positive_int, default=1)
    capture.add_argument("--output", type=Path, required=True)
    capture.set_defaults(handler=snapshot_gallery)
    provisioning = commands.add_parser("prepare", help="显式创建6个新Topic和4个隔离集合，可恢复暖图库")
    provisioning.add_argument("--manifest", type=Path, required=True)
    provisioning.add_argument("--warm-snapshot", type=Path)
    provisioning.add_argument("--partitions", type=positive_int, default=1)
    provisioning.add_argument("--replication-factor", type=positive_int, default=1)
    provisioning.set_defaults(handler=prepare)
    start = commands.add_parser("worker", help="在当前进程启动隔离Worker；需先prepare")
    start.add_argument("--manifest", type=Path, required=True)
    start.set_defaults(handler=worker)
    send = commands.add_parser("replay", help="显式限量投递到已prepare的隔离输入；需另行启动Worker")
    send.add_argument("--manifest", type=Path, required=True)
    send.add_argument("--count", type=positive_int, default=10)
    send.add_argument("--qps", type=float, default=1)
    send.add_argument("--wait-seconds", type=float, default=60)
    send.add_argument("--max-inflight", type=positive_int, default=20)
    send.add_argument("--metrics", action="append", default=[])
    send.set_defaults(handler=replay)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"隔离工具失败: {type(error).__name__}；未自动删除任何资源。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
