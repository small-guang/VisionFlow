import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class PipelineConfigTests(unittest.TestCase):
    required = {
        "PGSQL_USER": "test", "PGSQL_PSWD": "test", "PGSQL_HOST": "localhost",
        "PGSQL_PORT": "5432", "PGSQL_DB": "test", "KAFKA_BROKER_ADDR": "localhost",
        "KAFKA_BROKER_PORT": "9092", "KAFKA_GROUP_ID": "test",
        "TRITON_SERVER_URL": "localhost:8001", "MILVUS_URI": "http://localhost:19530",
        "MILVUS_TOKEN": "test", "STORAGE_BACKEND": "oss",
        "OSS_ACCESS_KEY_ID": "test", "OSS_ACCESS_KEY_SECRET": "test",
        "OSS_ENDPOINT": "oss.example.com", "OSS_BUCKET_NAME": "test",
    }

    def setUp(self):
        environment = patch.dict(os.environ, self.required, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def settings(self, *, env_file=None, **overrides):
        from core.config import Settings

        return Settings(_env_file=env_file, **overrides)

    def test_defaults_without_env_file(self):
        config = self.settings()
        self.assertTrue(config.PIPELINE_STATS_ENABLED)
        self.assertEqual(config.PIPELINE_STATS_PATH, "logs/pipeline_stats.jsonl")
        self.assertEqual(config.PIPELINE_STATS_INTERVAL_SECONDS, 5.0)
        self.assertEqual(config.OUTPUT_BACKLOG_HIGH_WATERMARK, 2000)
        self.assertEqual(config.OUTPUT_BACKLOG_LOW_WATERMARK, 1000)

    def test_runtime_environment_overrides_file_and_file_overrides_defaults(self):
        file_values = {
            "PIPELINE_STATS_ENABLED": "false",
            "PIPELINE_STATS_PATH": "logs/file-stats.jsonl",
            "PIPELINE_STATS_INTERVAL_SECONDS": "2.5",
            "OUTPUT_BACKLOG_HIGH_WATERMARK": "12",
            "OUTPUT_BACKLOG_LOW_WATERMARK": "4",
        }
        runtime_values = {
            "PIPELINE_STATS_ENABLED": "true",
            "PIPELINE_STATS_PATH": "logs/runtime-stats.jsonl",
            "PIPELINE_STATS_INTERVAL_SECONDS": "0.25",
            "OUTPUT_BACKLOG_HIGH_WATERMARK": "8",
            "OUTPUT_BACKLOG_LOW_WATERMARK": "0",
        }
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / "pipeline.env"
            env_file.write_text("\n".join(f"{key}={value}" for key, value in file_values.items()), encoding="utf-8")
            from_file = self.settings(env_file=env_file)
            self.assertFalse(from_file.PIPELINE_STATS_ENABLED)
            self.assertEqual(from_file.PIPELINE_STATS_PATH, file_values["PIPELINE_STATS_PATH"])
            self.assertEqual(from_file.PIPELINE_STATS_INTERVAL_SECONDS, 2.5)
            self.assertEqual(from_file.OUTPUT_BACKLOG_HIGH_WATERMARK, 12)
            self.assertEqual(from_file.OUTPUT_BACKLOG_LOW_WATERMARK, 4)
            os.environ.update(runtime_values)
            from_runtime = self.settings(env_file=env_file)
            self.assertTrue(from_runtime.PIPELINE_STATS_ENABLED)
            self.assertEqual(from_runtime.PIPELINE_STATS_PATH, runtime_values["PIPELINE_STATS_PATH"])
            self.assertEqual(from_runtime.PIPELINE_STATS_INTERVAL_SECONDS, 0.25)
            self.assertEqual(from_runtime.OUTPUT_BACKLOG_HIGH_WATERMARK, 8)
            self.assertEqual(from_runtime.OUTPUT_BACKLOG_LOW_WATERMARK, 0)

    def test_enabled_requires_a_valid_boolean(self):
        for value, expected in (("false", False), ("0", False), ("true", True), ("1", True)):
            with self.subTest(value=value):
                os.environ["PIPELINE_STATS_ENABLED"] = value
                self.assertIs(self.settings().PIPELINE_STATS_ENABLED, expected)
        for value in ("", "sometimes", "2"):
            with self.subTest(value=value):
                os.environ["PIPELINE_STATS_ENABLED"] = value
                with self.assertRaisesRegex(ValueError, "PIPELINE_STATS_ENABLED"):
                    self.settings()

    def test_log_path_rejects_blank_and_preserves_valid_paths(self):
        for value in ("", "   ", "\t"):
            with self.subTest(value=value):
                os.environ["PIPELINE_STATS_PATH"] = value
                with self.assertRaisesRegex(ValueError, "PIPELINE_STATS_PATH"):
                    self.settings()
        os.environ["PIPELINE_STATS_PATH"] = "  /tmp/alog-stats/pipeline.jsonl  "
        self.assertEqual(self.settings().PIPELINE_STATS_PATH, "/tmp/alog-stats/pipeline.jsonl")

    def test_interval_must_be_positive_and_finite(self):
        for value in ("0", "-0.5", "NaN", "Infinity", "-Infinity", "", "bad"):
            with self.subTest(value=value):
                os.environ["PIPELINE_STATS_INTERVAL_SECONDS"] = value
                with self.assertRaisesRegex(ValueError, "PIPELINE_STATS_INTERVAL_SECONDS"):
                    self.settings()
        os.environ["PIPELINE_STATS_INTERVAL_SECONDS"] = "0.125"
        self.assertEqual(self.settings().PIPELINE_STATS_INTERVAL_SECONDS, 0.125)

    def test_watermarks_require_integer_ranges_and_strict_order(self):
        for name, invalid_values in (
            ("OUTPUT_BACKLOG_HIGH_WATERMARK", ("0", "-1", "1.5", "bad", "")),
            ("OUTPUT_BACKLOG_LOW_WATERMARK", ("-1", "1.5", "bad", "")),
        ):
            for value in invalid_values:
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    with self.assertRaisesRegex(ValueError, name):
                        self.settings()
        for high, low in ((1, 1), (2, 3)):
            with self.subTest(high=high, low=low), patch.dict(os.environ, {
                "OUTPUT_BACKLOG_HIGH_WATERMARK": str(high),
                "OUTPUT_BACKLOG_LOW_WATERMARK": str(low),
            }):
                with self.assertRaisesRegex(ValueError, "OUTPUT_BACKLOG_LOW_WATERMARK.*lower"):
                    self.settings()
        config = self.settings(OUTPUT_BACKLOG_HIGH_WATERMARK=1, OUTPUT_BACKLOG_LOW_WATERMARK=0)
        self.assertEqual((config.OUTPUT_BACKLOG_HIGH_WATERMARK, config.OUTPUT_BACKLOG_LOW_WATERMARK), (1, 0))

    def test_environment_override_is_validated_against_file_watermark(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / "pipeline.env"
            env_file.write_text("OUTPUT_BACKLOG_HIGH_WATERMARK=10\nOUTPUT_BACKLOG_LOW_WATERMARK=5\n", encoding="utf-8")
            os.environ["OUTPUT_BACKLOG_HIGH_WATERMARK"] = "4"
            with self.assertRaisesRegex(ValueError, "OUTPUT_BACKLOG_LOW_WATERMARK.*lower"):
                self.settings(env_file=env_file)

    def test_disabling_statistics_keeps_configuration_and_required_field_validation(self):
        os.environ["PIPELINE_STATS_ENABLED"] = "false"
        with patch.dict(os.environ, {"PIPELINE_STATS_INTERVAL_SECONDS": "NaN"}):
            with self.assertRaisesRegex(ValueError, "PIPELINE_STATS_INTERVAL_SECONDS"):
                self.settings()
        os.environ.pop("PGSQL_USER")
        with self.assertRaisesRegex(ValueError, "PGSQL_USER"):
            self.settings()


if __name__ == "__main__":
    unittest.main()
