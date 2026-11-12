"""Public deployment defaults remain isolated from old private environments."""

import os
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ENV = {
    "PGSQL_USER": "demo", "PGSQL_PSWD": "demo", "PGSQL_HOST": "localhost",
    "PGSQL_PORT": "5432", "PGSQL_DB": "demo", "KAFKA_BROKER_ADDR": "localhost",
    "KAFKA_BROKER_PORT": "9092", "KAFKA_GROUP_ID": "demo", "KAFKA_SECURITY_PROTOCOL": "PLAINTEXT",
    "TRITON_SERVER_URL": "localhost:8001", "MILVUS_URI": "http://localhost:19530",
    "MILVUS_TOKEN": "demo", "STORAGE_BACKEND": "minio", "MINIO_ENDPOINT": "localhost:9000",
    "MINIO_ACCESS_KEY": "demo", "MINIO_SECRET_KEY": "demo", "MINIO_BUCKET_NAME": "demo",
    "VISUAL_WORKER_PROCESSES": "0",
}


class PublicDefaultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        config = Path(cls.temp.name) / "core" / "config.py"
        config.parent.mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1] / "core" / "config.py", config)
        with patch.dict(os.environ, ENV, clear=True):
            cls.Settings = runpy.run_path(str(config))["Settings"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_public_topics_and_no_device_allowlist_by_default(self):
        with patch.dict(os.environ, ENV, clear=True):
            settings = self.Settings(_env_file=None)
        self.assertEqual(settings.person_upright_device_sns, frozenset())
        self.assertEqual(settings.KAFKA_FACE_DETECTION_RECORD_TOPIC, "visionflow-face-detection-record")
        self.assertEqual(settings.KAFKA_FACE_RECOGNITION_RECORD_TOPIC, "visionflow-face-recognition-record")
        self.assertEqual(settings.KAFKA_PERSON_DETECTION_RECORD_TOPIC, "visionflow-person-detection-record")
        self.assertEqual(settings.KAFKA_PERSON_RECOGNITION_RECORD_TOPIC, "visionflow-person-recognition-record")
        self.assertEqual(settings.KAFKA_VECTOR_RECORD_TOPIC, "visionflow-vector-record")

    def test_existing_topics_and_device_list_are_configurable(self):
        overrides = {**ENV, "KAFKA_FACE_DETECTION_RECORD_TOPIC": "legacy-face-topic",
                     "PERSON_UPRIGHT_DEVICE_SNS": "camera-a, camera-b"}
        with patch.dict(os.environ, overrides, clear=True):
            settings = self.Settings(_env_file=None)
        self.assertEqual(settings.KAFKA_FACE_DETECTION_RECORD_TOPIC, "legacy-face-topic")
        self.assertEqual(settings.person_upright_device_sns, frozenset({"camera-a", "camera-b"}))


if __name__ == "__main__":
    unittest.main()
