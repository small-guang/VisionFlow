"""隔离回放工具的纯本地/fake验证；不创建真实Topic或集合。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from performance_tools.isolated_replay import (
    _prepare_galleries, capture_gallery, create_topics, encode, input_summary, isolation_env,
    isolation_names, load_manifest, load_warm_snapshot, plan, save_new, seed_galleries,
)


class FakeMilvus:
    def __init__(self, collections):
        self.collections = deepcopy(collections)
        self.calls = []

    def describe_collection(self, *, collection_name):
        self.calls.append(("describe", collection_name))
        return {"auto_id": False, "fields": [
            {"name": "id", "is_primary": True, "auto_id": False},
            {"name": "embedding"}, {"name": "faceId"},
        ]}

    def query(self, *, collection_name, filter, output_fields, limit, consistency_level, **kwargs):
        self.calls.append(("query", collection_name))
        return [{key: row[key] for key in output_fields} for row in self.collections[collection_name][:limit]]

    def insert(self, *, collection_name, data):
        self.calls.append(("insert", collection_name))
        self.collections[collection_name].extend(deepcopy(data))
        return {"insert_count": len(data)}

    def flush(self, *, collection_name):
        self.calls.append(("flush", collection_name))


class IsolatedReplayTests(unittest.TestCase):
    def test_namespace_defines_all_isolation_boundaries_and_rejects_path_components(self):
        names = isolation_names("case_01")
        manifest = {"schema": 1, "namespace": "case_01", "names": names}
        env = isolation_env(manifest, Path("/tmp/isolated"))
        self.assertEqual(len(names["output_topics"]), 5)
        self.assertEqual(len(names["collections"]), 4)
        self.assertEqual(env["KAFKA_INPUT_TOPICS"], "alogbench.case_01.input")
        self.assertEqual(env["MILVUS_COLLECTION_PREFIX"], "alogbench_case_01_")
        self.assertEqual(env["STORAGE_OUTPUT_PREFIX"], "alogbench/case_01")
        self.assertEqual(env["KAFKA_GROUP_ID"], "alogbench.case_01.worker")
        for invalid in ("", "../prod", "prod/name", "UPPER", "x;echo", "x" * 49):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                isolation_names(invalid)

    def test_manifest_cannot_override_derived_output_topic_or_collection(self):
        for path in ("input_topic", "collections"):
            names = isolation_names("tamper")
            names[path] = "production" if path == "input_topic" else {"face": "alogFace"}
            with self.subTest(path=path), self.assertRaises(ValueError):
                isolation_env({"schema": 1, "namespace": "tamper", "names": names}, Path("/tmp/test"))

    def test_input_summary_counts_real_snapshot_without_exposing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(json.dumps({"payload": {"keyframeUrl": "private-key", "panoKeyframeUrl": "pano"}}) + "\n")
            summary = input_summary(path)
            self.assertEqual(summary["messages"], 1)
            self.assertEqual(summary["pano_messages"], 1)
            self.assertNotIn("private-key", json.dumps(summary))
            self.assertFalse(summary["image_availability_verified"])
            path.write_text("")
            with self.assertRaises(ValueError):
                input_summary(path)

    def test_warm_capture_is_read_only_and_never_truncates_silently(self):
        rows = [{"id": 1, "embedding": [0.1], "faceId": "identity"}, {"id": 2, "embedding": [0.2], "faceId": "identity"}]
        client = FakeMilvus({"alogFace": rows})
        snapshot = capture_gallery(client, "alogFace", 2)
        self.assertEqual(snapshot["rows"], rows)
        self.assertEqual(snapshot["rows_sha256"], hashlib.sha256(encode(rows)).hexdigest())
        self.assertTrue(all(call[0] in {"query", "describe"} for call in client.calls))
        self.assertEqual(client.collections["alogFace"], rows)
        with self.assertRaisesRegex(ValueError, "超过"):
            capture_gallery(client, "alogFace", 1)

    def test_seed_preserves_identity_and_ids_and_writes_only_empty_isolated_targets(self):
        names = isolation_names("warm")
        source_rows = [{"id": 7, "embedding": [0.1], "faceId": "same-face"}]
        client = FakeMilvus({"alogFace": source_rows, "alogSBS": source_rows,
                            names["collections"]["face"]: [], names["collections"]["sbs"]: []})
        snapshot = {"schema": 1, "galleries": {kind: capture_gallery(client, source, 1) for kind, source in (("face", "alogFace"), ("sbs", "alogSBS"))}}
        databases = SimpleNamespace(**{kind: SimpleNamespace(client=client, collection_name=names["collections"][kind]) for kind in ("face", "sbs")})
        self.assertEqual(seed_galleries(databases, snapshot, names), {"face": 1, "sbs": 1})
        self.assertEqual(client.collections[names["collections"]["face"]], source_rows)
        self.assertEqual(client.collections["alogFace"], source_rows)
        self.assertTrue(all(name.startswith("alogbench_warm_") for method, name in client.calls if method in {"insert", "flush"}))
        with self.assertRaisesRegex(ValueError, "必须为空"):
            seed_galleries(databases, snapshot, names)
        databases.face.collection_name = "alogFace"
        with self.assertRaisesRegex(ValueError, "不属于"):
            seed_galleries(databases, snapshot, names)

    def test_snapshot_hash_and_prepared_manifest_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = {"row_count": 0, "rows": [], "rows_sha256": hashlib.sha256(encode([])).hexdigest()}
            snapshot = {"schema": 1, "galleries": {"face": item, "sbs": item}}
            path = root / "warm.json"
            save_new(path, snapshot, private=True)
            self.assertEqual(load_warm_snapshot(path), snapshot)
            broken = deepcopy(snapshot)
            broken["galleries"]["face"]["row_count"] = 1
            path.write_bytes(encode(broken))
            with self.assertRaises(ValueError):
                load_warm_snapshot(path)
            manifest = {"schema": 1, "namespace": "proof", "names": isolation_names("proof")}
            manifest_path = root / "manifest.json"
            save_new(manifest_path, manifest)
            save_new(root / "prepared.json", {"manifest_sha256": "wrong"})
            with self.assertRaises(ValueError):
                load_manifest(manifest_path, prepared=True)

    def test_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            save_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                save_new(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})


class IsolatedGalleryPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_preparation_closes_query_and_execution_resources_before_database(self):
        for failure in (None, ValueError("seed failed")):
            with self.subTest(failure=failure):
                resources_seen = []

                def create(config, resources):
                    resources_seen.append(resources)
                    self.assertIsNotNone(resources.vector_queries)
                    return databases

                def close():
                    self.assertTrue(all(item["closed"] for item in resources_seen[0].queue_snapshot().values()))

                databases = SimpleNamespace(close=Mock(side_effect=close))
                factory = SimpleNamespace(DatabaseServices=SimpleNamespace(create=create))
                with patch.dict("sys.modules", {"db.zilliz": factory}), patch(
                    "performance_tools.isolated_replay.seed_galleries", return_value={"face": 1}, side_effect=failure,
                ):
                    if failure is None:
                        self.assertEqual(await _prepare_galleries(object(), object(), object()), {"face": 1})
                    else:
                        with self.assertRaises(ValueError) as caught:
                            await _prepare_galleries(object(), object(), object())
                        self.assertIs(caught.exception, failure)
                databases.close.assert_called_once_with()


class IsolatedTopicTests(unittest.IsolatedAsyncioTestCase):
    async def test_creation_is_limited_to_six_fresh_topics_and_always_closes_admin(self):
        class Admin:
            def __init__(self):
                self.topics = set()
                self.created = []
                self.closed = 0
            async def start(self): pass
            async def list_topics(self): return list(self.topics)
            async def create_topics(self, definitions):
                self.created.extend(definitions)
                self.topics.update(item.name for item in definitions)
                return SimpleNamespace(topic_errors=[])
            async def close(self): self.closed += 1
        admin = Admin()
        names = isolation_names("topics")
        await create_topics(SimpleNamespace(kafka_client_config={}), names, partitions=1, replication_factor=1, admin_factory=lambda **kwargs: admin)
        self.assertEqual(len(admin.created), 6)
        self.assertTrue(all(item.name.startswith("alogbench.topics.") for item in admin.created))
        self.assertEqual(admin.closed, 1)
        with self.assertRaisesRegex(ValueError, "已存在"):
            await create_topics(SimpleNamespace(kafka_client_config={}), names, partitions=1, replication_factor=1, admin_factory=lambda **kwargs: admin)
        self.assertEqual(len(admin.created), 6)
        self.assertEqual(admin.closed, 2)
