import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class PipelineIsolationConfigTests(unittest.TestCase):
    required = {
        "PGSQL_USER": "test", "PGSQL_PSWD": "test", "PGSQL_HOST": "localhost",
        "PGSQL_PORT": "5432", "PGSQL_DB": "test", "KAFKA_BROKER_ADDR": "localhost",
        "KAFKA_BROKER_PORT": "9092", "KAFKA_GROUP_ID": "test",
        "TRITON_SERVER_URL": "localhost:8001", "MILVUS_URI": "http://localhost:19530",
        "MILVUS_TOKEN": "test", "STORAGE_BACKEND": "oss",
        "OSS_ACCESS_KEY_ID": "test", "OSS_ACCESS_KEY_SECRET": "test",
        "OSS_ENDPOINT": "oss.example.com", "OSS_BUCKET_NAME": "test",
    }
    nonnegative_fields = (
        "ALOG_MAX_INFLIGHT_MESSAGES", "FACE_PREPARE_LOOKAHEAD", "PERSON_PREPARE_LOOKAHEAD",
    )

    def setUp(self):
        environment = patch.dict(os.environ, self.required, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    @staticmethod
    def settings(*, env_file=None):
        from core.config import Settings

        return Settings(_env_file=env_file)

    def test_defaults(self):
        config = self.settings()
        self.assertEqual(config.ALOG_MAX_INFLIGHT_MESSAGES, 100)
        self.assertEqual(config.ALOG_MAX_INFLIGHT_FRAMES, 100)
        self.assertEqual(config.FACE_GLOBAL_CONCURRENCY, 100)
        self.assertEqual(config.PERSON_GLOBAL_CONCURRENCY, 100)
        self.assertEqual(config.FACE_PREPARE_LOOKAHEAD, 2)
        self.assertEqual(config.PERSON_PREPARE_LOOKAHEAD, 2)
        self.assertTrue(config.FACE_SEARCH_PREFETCH_ENABLED)

    def test_runtime_environment_overrides_file_and_file_overrides_defaults(self):
        file_values = {
            "ALOG_MAX_INFLIGHT_MESSAGES": "17",
            "FACE_PREPARE_LOOKAHEAD": "3",
            "PERSON_PREPARE_LOOKAHEAD": "4",
            "FACE_SEARCH_PREFETCH_ENABLED": "false",
        }
        runtime_values = {
            "ALOG_MAX_INFLIGHT_MESSAGES": "8",
            "FACE_PREPARE_LOOKAHEAD": "0",
            "PERSON_PREPARE_LOOKAHEAD": "1",
            "FACE_SEARCH_PREFETCH_ENABLED": "true",
        }
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / "isolation.env"
            env_file.write_text(
                "\n".join(f"{key}={value}" for key, value in file_values.items()),
                encoding="utf-8",
            )
            config = self.settings(env_file=env_file)
            for name in self.nonnegative_fields:
                self.assertEqual(getattr(config, name), int(file_values[name]))
            self.assertFalse(config.FACE_SEARCH_PREFETCH_ENABLED)
            with patch.dict(os.environ, runtime_values):
                config = self.settings(env_file=env_file)
            for name in self.nonnegative_fields:
                self.assertEqual(getattr(config, name), int(runtime_values[name]))
            self.assertTrue(config.FACE_SEARCH_PREFETCH_ENABLED)

    def test_message_limit_and_lookahead_reject_negative_fractional_and_invalid_values(self):
        for name in self.nonnegative_fields:
            for value in ("-1", "0.5", "NaN", "Infinity", "true", "false", "", "bad"):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    with self.assertRaisesRegex(ValueError, name):
                        self.settings()

    def test_zero_lookahead_and_single_preparation_slot_are_valid(self):
        with patch.dict(os.environ, {
            "ALOG_MAX_INFLIGHT_MESSAGES": "0",
            "FACE_FRAME_CONCURRENCY": "1", "FACE_PREPARE_LOOKAHEAD": "0",
            "PERSON_FRAME_CONCURRENCY": "1", "PERSON_PREPARE_LOOKAHEAD": "0",
        }):
            config = self.settings()
        self.assertEqual(config.ALOG_MAX_INFLIGHT_MESSAGES, 0)
        self.assertEqual((config.FACE_FRAME_CONCURRENCY, config.FACE_PREPARE_LOOKAHEAD), (1, 0))
        self.assertEqual((config.PERSON_FRAME_CONCURRENCY, config.PERSON_PREPARE_LOOKAHEAD), (1, 0))

    def test_explicit_message_limit_is_independent_of_dispatch_worker_count(self):
        for limit in (1, 2, 5):
            with self.subTest(limit=limit), patch.dict(os.environ, {
                "ALOG_WORKER_COUNT": "2", "ALOG_MAX_INFLIGHT_MESSAGES": str(limit),
            }):
                config = self.settings()
                self.assertEqual(config.ALOG_MAX_INFLIGHT_MESSAGES, limit)
                self.assertEqual(config.ALOG_WORKER_COUNT, 2)

    def test_prefetch_switch_accepts_supported_boolean_forms(self):
        for value, expected in (("true", True), ("false", False), ("1", True), ("0", False)):
            with self.subTest(value=value), patch.dict(os.environ, {"FACE_SEARCH_PREFETCH_ENABLED": value}):
                self.assertIs(self.settings().FACE_SEARCH_PREFETCH_ENABLED, expected)

    def test_prefetch_switch_rejects_invalid_boolean_values(self):
        for value in ("", "2", "sometimes", "NaN"):
            with self.subTest(value=value), patch.dict(os.environ, {"FACE_SEARCH_PREFETCH_ENABLED": value}):
                with self.assertRaisesRegex(ValueError, "FACE_SEARCH_PREFETCH_ENABLED"):
                    self.settings()

    def test_invalid_environment_override_is_not_hidden_by_valid_file(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / "isolation.env"
            env_file.write_text("FACE_PREPARE_LOOKAHEAD=2\n", encoding="utf-8")
            with patch.dict(os.environ, {"FACE_PREPARE_LOOKAHEAD": "-1"}):
                with self.assertRaisesRegex(ValueError, "FACE_PREPARE_LOOKAHEAD"):
                    self.settings(env_file=env_file)

    def test_disabling_prefetch_does_not_bypass_other_or_required_field_validation(self):
        os.environ["FACE_SEARCH_PREFETCH_ENABLED"] = "false"
        with patch.dict(os.environ, {"PERSON_PREPARE_LOOKAHEAD": "-1"}):
            with self.assertRaisesRegex(ValueError, "PERSON_PREPARE_LOOKAHEAD"):
                self.settings()
        os.environ.pop("PGSQL_USER")
        with self.assertRaisesRegex(ValueError, "PGSQL_USER"):
            self.settings()


if __name__ == "__main__":
    unittest.main()
