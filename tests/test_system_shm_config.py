import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class SystemSharedMemoryConfigTests(unittest.TestCase):
    def settings(self, **overrides):
        from core.config import Settings

        required = {
            "PGSQL_USER": "test", "PGSQL_PSWD": "test", "PGSQL_HOST": "localhost",
            "PGSQL_PORT": 5432, "PGSQL_DB": "test", "KAFKA_BROKER_ADDR": "localhost",
            "KAFKA_BROKER_PORT": 9092, "KAFKA_GROUP_ID": "test",
            "TRITON_SERVER_URL": "localhost:8001", "MILVUS_URI": "http://localhost:19530",
            "MILVUS_TOKEN": "test", "STORAGE_BACKEND": "oss",
            "OSS_ACCESS_KEY_ID": "test", "OSS_ACCESS_KEY_SECRET": "test",
            "OSS_ENDPOINT": "oss.example.com", "OSS_BUCKET_NAME": "test",
        }
        required.update(overrides)
        required.setdefault("_env_file", None)
        return Settings(**required)

    def test_opt_in_defaults_and_environment_over_file(self):
        with patch.dict(os.environ, {}, clear=True), TemporaryDirectory() as directory:
            settings = self.settings()
            self.assertEqual(settings.TRITON_SYSTEM_SHM_SLOTS, 0)
            self.assertEqual(settings.TRITON_SYSTEM_SHM_SLOT_BYTES, 32 * 1024 * 1024)
            env_file = Path(directory) / "shm.env"
            env_file.write_text("TRITON_SYSTEM_SHM_SLOTS=8\n", encoding="utf-8")
            self.assertEqual(self.settings(_env_file=env_file).TRITON_SYSTEM_SHM_SLOTS, 8)
            os.environ["TRITON_SYSTEM_SHM_SLOTS"] = "4"
            self.assertEqual(self.settings(_env_file=env_file).TRITON_SYSTEM_SHM_SLOTS, 4)

    def test_resource_limits_are_validated_before_startup(self):
        with patch.dict(os.environ, {}, clear=True):
            valid = self.settings(TRITON_SYSTEM_SHM_SLOTS=16)
            self.assertEqual(valid.TRITON_SYSTEM_SHM_SLOTS * valid.TRITON_SYSTEM_SHM_SLOT_BYTES,
                             512 * 1024 * 1024)
            for overrides in (
                {"TRITON_SYSTEM_SHM_SLOTS": -1},
                {"TRITON_SYSTEM_SHM_SLOTS": 65},
                {"TRITON_SYSTEM_SHM_SLOT_BYTES": 1024},
                {"TRITON_SYSTEM_SHM_SLOTS": 17},
            ):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    self.settings(**overrides)


if __name__ == "__main__":
    unittest.main()
