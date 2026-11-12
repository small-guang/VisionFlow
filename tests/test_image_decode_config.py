import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class ImageDecodeConfigTests(unittest.TestCase):
    def test_default_environment_and_file_precedence_and_positive_validation(self):
        from core.config import Settings

        required = {
            "VISUAL_WORKER_PROCESSES": 0,
            "PGSQL_USER": "test", "PGSQL_PSWD": "test", "PGSQL_HOST": "localhost",
            "PGSQL_PORT": 5432, "PGSQL_DB": "test", "KAFKA_BROKER_ADDR": "localhost",
            "KAFKA_BROKER_PORT": 9092, "KAFKA_GROUP_ID": "test",
            "TRITON_SERVER_URL": "localhost:8001", "MILVUS_URI": "http://localhost:19530",
            "MILVUS_TOKEN": "test", "STORAGE_BACKEND": "oss",
            "OSS_ACCESS_KEY_ID": "test", "OSS_ACCESS_KEY_SECRET": "test",
            "OSS_ENDPOINT": "oss.example.com", "OSS_BUCKET_NAME": "test",
        }
        for name, default in (
            ("IMAGE_CPU_WORKERS", 4),
        ):
            with self.subTest(setting=name), patch.dict(os.environ), TemporaryDirectory() as directory:
                os.environ.pop(name, None)
                self.assertEqual(getattr(Settings(_env_file=None, **required), name), default)
                env_path = Path(directory) / "cpu-pool.env"
                env_path.write_text(f"{name}=5\n", encoding="utf-8")
                self.assertEqual(getattr(Settings(_env_file=env_path, **required), name), 5)
                os.environ[name] = "3"
                self.assertEqual(getattr(Settings(_env_file=env_path, **required), name), 3)
                invalid_values = ("-1", "bad", "1.5", "", "0")
                for invalid in invalid_values:
                    with self.subTest(value=invalid):
                        os.environ[name] = invalid
                        with self.assertRaises(ValueError):
                            Settings(_env_file=None, **required)


if __name__ == "__main__":
    unittest.main()
