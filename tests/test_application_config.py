import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class ApplicationConfigTests(unittest.TestCase):
    required = {
        "PGSQL_USER": "test", "PGSQL_PSWD": "test", "PGSQL_HOST": "localhost",
        "PGSQL_PORT": "5432", "PGSQL_DB": "test", "KAFKA_BROKER_ADDR": "localhost",
        "KAFKA_BROKER_PORT": "9092", "KAFKA_GROUP_ID": "test",
        "TRITON_SERVER_URL": "localhost:8001", "MILVUS_URI": "http://localhost:19530",
        "MILVUS_TOKEN": "test", "STORAGE_BACKEND": "oss",
        "OSS_ACCESS_KEY_ID": "test", "OSS_ACCESS_KEY_SECRET": "test",
        "OSS_ENDPOINT": "oss.example.com", "OSS_BUCKET_NAME": "test",
    }

    def config(self, values=None):
        with patch.dict(os.environ, self.required, clear=True):
            from core.config import Settings
        with patch.dict(os.environ, {**self.required, **(values or {})}, clear=True):
            return Settings(_env_file=None)

    def test_resources_are_separate_and_existing_names_are_default(self):
        config = self.config()
        self.assertEqual((config.IMAGE_CPU_WORKERS, config.STORAGE_IO_WORKERS,
                          config.MILVUS_SEARCH_WORKERS, config.MILVUS_WRITE_WORKERS), (4, 64, 16, 4))
        self.assertEqual(config.MILVUS_COLLECTION_PREFIX, "")
        self.assertEqual(config.STORAGE_OUTPUT_PREFIX, "")
        self.assertEqual(config.KAFKA_INPUT_TOPICS,
                         ("PanoramicPersonDetSync", "PtzPersonDetSync", "PtzFaceDetSync"))

    def test_worker_counts_reject_invalid_runtime_values(self):
        for name in ("IMAGE_CPU_WORKERS", "STORAGE_IO_WORKERS", "MILVUS_SEARCH_WORKERS", "MILVUS_WRITE_WORKERS"):
            invalid = ("-1", "0.5", "true", "NaN", "", "0")
            for value in invalid:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.config({name: value})

    def test_retired_pool_settings_do_not_control_resource_budgets(self):
        config = self.config({"IO_THREAD_WORKERS": "32", "IMAGE_DECODE_WORKERS": "20",
                              "PERSON_PREPARATION_WORKERS": "20"})
        self.assertEqual(config.STORAGE_IO_WORKERS, 64)
        self.assertEqual(config.IMAGE_CPU_WORKERS, 4)
        for name in ("IO_THREAD_WORKERS", "IMAGE_DECODE_WORKERS", "PERSON_PREPARATION_WORKERS",
                     "storage_io_workers"):
            self.assertFalse(hasattr(config, name))

    def test_storage_capacity_respects_environment_and_dotenv_priority(self):
        from core.config import Settings
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("IO_THREAD_WORKERS=32\nSTORAGE_IO_WORKERS=12\n", encoding="utf-8")
            with patch.dict(os.environ, self.required, clear=True):
                self.assertEqual(Settings(_env_file=env_file).STORAGE_IO_WORKERS, 12)
            with patch.dict(os.environ, {**self.required, "STORAGE_IO_WORKERS": "24"}, clear=True):
                self.assertEqual(Settings(_env_file=env_file).STORAGE_IO_WORKERS, 24)
            with patch.dict(os.environ, {**self.required, "STORAGE_IO_WORKERS": "0"}, clear=True):
                with self.assertRaisesRegex(ValueError, "STORAGE_IO_WORKERS"):
                    Settings(_env_file=env_file)

    def test_runtime_resource_factory_uses_effective_storage_capacity(self):
        from core.application import _resources
        with patch("core.execution.ExecutionResources") as factory:
            _resources(self.config({"STORAGE_IO_WORKERS": "32"}))
        self.assertEqual(factory.call_args.kwargs["storage_workers"], 32)

    def test_frame_memory_reservation_must_fit_before_admission(self):
        with self.assertRaisesRegex(ValueError, "FRAME_MEMORY_BUDGET_BYTES"):
            self.config({"FRAME_MEMORY_BUDGET_BYTES": "1024", "FRAME_RESERVATION_BYTES": "2048"})
        self.assertEqual(self.config({"FRAME_MEMORY_BUDGET_BYTES": "0"}).FRAME_MEMORY_BUDGET_BYTES, 0)

    def test_isolated_namespaces_preserve_explicit_topic_order(self):
        config = self.config({"KAFKA_INPUT_TOPICS": "test-a,test-b", "MILVUS_COLLECTION_PREFIX": "bench_",
                              "STORAGE_OUTPUT_PREFIX": "/bench/run/"})
        self.assertEqual(config.KAFKA_INPUT_TOPICS, ("test-a", "test-b"))
        self.assertEqual(config.MILVUS_COLLECTION_PREFIX, "bench_")
        self.assertEqual(config.STORAGE_OUTPUT_PREFIX, "bench/run")

    def test_invalid_namespaces_fail_early(self):
        cases = {
            "KAFKA_INPUT_TOPICS": ("", "a,a", "a,", "a/b", ".."),
            "MILVUS_COLLECTION_PREFIX": ("../prod", "1prefix", "a-b"),
            "STORAGE_OUTPUT_PREFIX": ("a/../b", "a//b", "a b", "a\\b"),
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.config({name: value})


if __name__ == "__main__":
    unittest.main()
