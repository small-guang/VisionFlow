"""SDK 启动预热不访问外部服务，并保留可选依赖与错误语义。"""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MilvusSdkRuntimeTests(unittest.TestCase):
    def load_runtime(self):
        name = "milvus_sdk_runtime_under_test"
        with patch.dict(sys.modules):
            return load_module(name, ROOT / "db/zilliz/sdk_runtime.py")

    def test_warmup_uses_sdk_type_check_without_creating_client(self):
        runtime = self.load_runtime()
        helper = SimpleNamespace(is_scipy_sparse=Mock(return_value=False))
        client = Mock(side_effect=AssertionError("must not connect"))
        with patch.dict(sys.modules, {
            "pymilvus": SimpleNamespace(MilvusClient=client),
            "pymilvus.client.utils": SimpleNamespace(SciPyHelper=helper),
        }):
            runtime.warm_search_dependencies()
        helper.is_scipy_sparse.assert_called_once_with([])
        client.assert_not_called()

    def test_dependency_initialization_failure_is_not_swallowed(self):
        runtime = self.load_runtime()
        for failure in (ImportError("broken dependency"), AttributeError("SDK changed")):
            with self.subTest(error=type(failure).__name__):
                helper = SimpleNamespace(is_scipy_sparse=Mock(side_effect=failure))
                with patch.dict(sys.modules, {
                    "pymilvus.client.utils": SimpleNamespace(SciPyHelper=helper),
                }), self.assertRaises(type(failure)) as caught:
                    runtime.warm_search_dependencies()
                self.assertIs(caught.exception, failure)

    def load_factory(self, warmup, events):
        name = "milvus_factory_under_test"
        modules = {}
        for filename, classname in (
            ("face_feature", "FaceGallery"), ("picture_feature", "MilvusVectorDB"),
            ("SBS_feature", "SbsReidGallery"), ("TBPS_feature", "TbpsGallery"),
        ):
            module = ModuleType(f"{name}.{filename}")

            def create(*, collection_name, **kwargs):
                events.append(collection_name)
                return SimpleNamespace(client=SimpleNamespace(close=Mock()), **kwargs)

            setattr(module, classname, create)
            modules[module.__name__] = module
        modules[f"{name}.sdk_runtime"] = SimpleNamespace(warm_search_dependencies=warmup)
        with patch.dict(sys.modules, modules):
            return load_module(name, ROOT / "db/zilliz/__init__.py")

    def test_factory_warms_before_any_gallery_for_each_creation(self):
        events = []
        factory = self.load_factory(lambda: events.append("warm"), events)
        config = SimpleNamespace(MILVUS_COLLECTION_PREFIX="test_")
        resources = SimpleNamespace(vector_search=object(), vector_write=object(), vector_queries=object())
        for _ in range(2):
            services = factory.DatabaseServices.create(config, resources)
            self.assertIs(services.face.query_coordinator, resources.vector_queries)
            services.close()
        self.assertEqual(events, [
            "warm", "test_alogFace", "test_alogPictureClip", "test_alogSBS", "test_alogTBPS",
        ] * 2)

    def test_factory_rejects_missing_query_resource_before_opening_galleries(self):
        events = []
        factory = self.load_factory(lambda: events.append("warm"), events)
        config = SimpleNamespace(MILVUS_COLLECTION_PREFIX="test_")
        for extra, error in (({}, AttributeError), ({"vector_queries": None}, ValueError)):
            resources = SimpleNamespace(vector_search=object(), vector_write=object(), **extra)
            with self.subTest(extra=extra), self.assertRaises(error):
                factory.DatabaseServices.create(config, resources)
        self.assertEqual(events, ["warm", "warm"])

    def test_factory_does_not_open_galleries_after_warmup_failure(self):
        events = []
        failure = RuntimeError("dependency initialization failed")
        factory = self.load_factory(Mock(side_effect=failure), events)
        with self.assertRaises(RuntimeError) as caught:
            factory.DatabaseServices.create(None, None)
        self.assertIs(caught.exception, failure)
        self.assertEqual(events, [])

    def test_pinned_sdk_preserves_absent_scipy_and_reuses_initialized_fastpath(self):
        if importlib.util.find_spec("pymilvus") is None:
            self.skipTest("pymilvus is not installed")
        # 冷进程中运行真实 SDK，不改当前测试进程的模块/初始化状态。
        code = """
import runpy, sys
from unittest.mock import patch
from pymilvus.client.utils import SciPyHelper
warm = runpy.run_path(sys.argv[1])["warm_search_dependencies"]
assert not SciPyHelper._checked
with patch("pymilvus.client.utils.importlib.util.find_spec", return_value=None) as find:
    warm()
    warm()
    assert SciPyHelper._checked
    assert not SciPyHelper._matrix_available
    assert not SciPyHelper._array_available
    assert find.call_count == 1
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(ROOT / "db/zilliz/sdk_runtime.py")],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
