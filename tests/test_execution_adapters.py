"""资源注入测试只替换外部 SDK，绝不创建真实业务连接。"""

import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.execution import ExecutionResources


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ExecutionAdapterTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        package_name = "execution_gallery_under_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "db/zilliz")]
        config = types.ModuleType("core.config")
        config.settings = types.SimpleNamespace(milvus_uri="unused", milvus_token="unused")
        pymilvus = types.ModuleType("pymilvus")
        pymilvus.DataType = types.SimpleNamespace(VARCHAR="VARCHAR")
        pymilvus.MilvusClient = Mock(side_effect=AssertionError("unexpected SDK construction"))
        with patch.dict(sys.modules, {package_name: package, "core.config": config, "pymilvus": pymilvus}):
            cls.modules = {
                name: load_module(f"{package_name}.{name}", ROOT / f"db/zilliz/{name}.py")
                for name in ("picture_feature", "person_feature", "face_feature", "SBS_feature", "TBPS_feature")
            }

    async def asyncSetUp(self):
        self.resources = ExecutionResources(
            image_workers=1, storage_workers=1, search_workers=2, write_workers=1,
        )
        self.addAsyncCleanup(self.resources.close)

    def test_gallery_config_and_collection_overrides_keep_default_names(self):
        picture = self.modules["picture_feature"]
        config = types.SimpleNamespace(milvus_uri="http://isolated.test:19530", milvus_token="test")
        constructors = (
            (picture.MilvusVectorDB, "image_collection"),
            (self.modules["face_feature"].FaceGallery, "alogFace"),
            (self.modules["SBS_feature"].SbsReidGallery, "alogSBS"),
            (self.modules["TBPS_feature"].TbpsGallery, "alogTBPS"),
        )
        for gallery_type, default in constructors:
            with self.subTest(gallery=gallery_type.__name__), patch.object(
                picture, "MilvusClient", return_value=object(),
            ) as constructor, patch.object(gallery_type, "_create_schema"), patch.object(
                gallery_type, "load_collection",
            ):
                for name in (None, "replay_" + default):
                    kwargs = {"collection_name": name} if name is not None else {}
                    gallery = gallery_type(
                        **kwargs, config=config,
                        search_executor=self.resources.vector_search,
                        write_executor=self.resources.vector_write,
                    )
                    self.assertEqual(gallery.collection_name, default if name is None else name)
                    self.assertIs(gallery.search_executor, self.resources.vector_search)
                    self.assertIs(gallery.write_executor, self.resources.vector_write)
                constructor.assert_called_with(uri=config.milvus_uri, token=config.milvus_token)

    def test_failed_constructor_closes_sdk_and_preserves_schema_error(self):
        picture = self.modules["picture_feature"]
        client = types.SimpleNamespace(close=Mock())
        failure = ValueError("incompatible collection")
        for step in ("_create_schema", "load_collection"):
            with self.subTest(step=step), patch.object(picture, "MilvusClient", return_value=client), patch.object(
                picture.MilvusVectorDB, "_create_schema",
            ) as schema, patch.object(picture.MilvusVectorDB, "load_collection") as load:
                client.close.reset_mock()
                (schema if step == "_create_schema" else load).side_effect = failure
                with self.assertRaises(ValueError) as raised:
                    picture.MilvusVectorDB()
                self.assertIs(raised.exception, failure)
                client.close.assert_called_once_with()

    async def test_gallery_reads_and_writes_use_their_own_resource_domains(self):
        def thread_name(*args, **kwargs):
            return threading.current_thread().name

        picture = object.__new__(self.modules["picture_feature"].MilvusVectorDB)
        face = object.__new__(self.modules["face_feature"].FaceGallery)
        sbs = object.__new__(self.modules["SBS_feature"].SbsReidGallery)
        tbps = object.__new__(self.modules["TBPS_feature"].TbpsGallery)
        for gallery in (picture, face, sbs, tbps):
            gallery.search_executor = self.resources.vector_search
            gallery.write_executor = self.resources.vector_write
            gallery.insert_embeddings = thread_name
        face.query_coordinator = self.resources.vector_queries
        face.collection_name = "face"
        search_threads = []

        def search(**kwargs):
            search_threads.append(threading.current_thread().name)
            return [[{"entity": {"faceId": "face-1", "sampleId": "sample-1"},
                      "distance": 0.95}] for _ in kwargs["data"]]

        face.client = types.SimpleNamespace(search=search)
        face.search_face_samples = Mock(side_effect=AssertionError("sync query bypassed coordinator"))
        face.search_similar_faces_with_hit_info = Mock(side_effect=AssertionError("sync query bypassed coordinator"))
        sbs.apply_prototype_updates = thread_name
        for operation in (
            lambda: picture.ainsert_embeddings(*range(5)),
            lambda: face.ainsert_embeddings(*range(11)),
            lambda: tbps.ainsert_embeddings(*range(8)),
            lambda: sbs.aapply_prototype_updates([]),
        ):
            self.assertTrue((await operation()).startswith("alog-vector-write"))
        self.assertEqual((await face.asearch_face_samples(np.array([1.0])))[0].face_id, "face-1")
        self.assertEqual(await face.asearch_similar_faces_with_hit_info(np.array([1.0])),
                         ("face-1", 0.95, "face-1"))
        self.assertEqual(len(search_threads), 2)
        self.assertTrue(all(name.startswith("alog-vector-search") for name in search_threads))

    async def test_sbs_grouped_queries_use_search_executor_and_keep_input_order(self):
        module = self.modules["SBS_feature"]
        gallery = object.__new__(module.SbsReidGallery)
        gallery.search_executor = self.resources.vector_search
        gallery._build_person_reid_base_filter = lambda *args: "filter"
        gallery._search_single_gallery_body_type = lambda _, __, body, ___: (
            body, threading.current_thread().name,
        )
        gallery._merge_person_reid_group_results = lambda results: results
        with patch.object(module, "gallery_body_types_for_query", return_value=["upper", "lower"]):
            results = await gallery.asearch_similar_person_candidates_with_hit_info([1.0], "full")
        self.assertEqual([item[0] for item in results], ["upper", "lower"])
        self.assertTrue(all(item[1].startswith("alog-vector-search") for item in results))

    async def test_storage_sdk_calls_including_response_read_use_storage_pool(self):
        observed = []

        def record(name):
            observed.append((name, threading.current_thread().name))

        class Response:
            def read(self):
                record("read")
                return b"image"

            def close(self):
                record("close")

            def release_conn(self):
                record("release")

        class Sdk:
            def get_object(self, *args):
                record("get")
                return Response()

            def put_object(self, *args, **kwargs):
                record("put")
                return types.SimpleNamespace(status=200)

        sdk = Sdk()
        config = types.ModuleType("core.config")
        config.settings = types.SimpleNamespace()
        oss2 = types.ModuleType("oss2")
        oss2.Auth = lambda *args: object()
        oss2.Bucket = lambda *args: sdk
        oss2.exceptions = types.SimpleNamespace(NoSuchKey=KeyError)
        minio = types.ModuleType("minio")
        minio.Minio = lambda *args, **kwargs: sdk
        minio_error = types.ModuleType("minio.error")
        minio_error.S3Error = type("S3Error", (Exception,), {})
        with patch.dict(sys.modules, {
            "core.config": config, "oss2": oss2, "minio": minio, "minio.error": minio_error,
        }):
            oss = load_module("execution_oss_under_test", ROOT / "utils/storage/oss_util.py")
            minio_module = load_module("execution_minio_under_test", ROOT / "utils/storage/minio_util.py")
            clients = (
                oss.AsyncOSSClient("key", "secret", "endpoint", "bucket", execution=self.resources.storage_io),
                minio_module.AsyncMinIOClient(
                    "endpoint", "key", "secret", "bucket", False, execution=self.resources.storage_io,
                ),
            )
            for client in clients:
                self.assertEqual(await client.get_object("source.jpg"), b"image")
                self.assertEqual(await client.upload_object(b"image", "crop.jpg"), "crop.jpg")
        self.assertEqual([name for name, _ in observed], [
            "get", "read", "put", "get", "read", "close", "release", "put",
        ])
        self.assertTrue(all(name.startswith("alog-storage-io") for _, name in observed))


if __name__ == "__main__":
    unittest.main()
