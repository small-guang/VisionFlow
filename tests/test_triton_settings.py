import os
import subprocess
import sys
import unittest
from unittest.mock import patch


class TritonSettingsTests(unittest.TestCase):
    required_environment = {
        "PGSQL_USER": "user",
        "PGSQL_PSWD": "password",
        "PGSQL_HOST": "localhost",
        "PGSQL_PORT": "5432",
        "PGSQL_DB": "alog",
        "KAFKA_BROKER_ADDR": "localhost",
        "KAFKA_BROKER_PORT": "9092",
        "KAFKA_GROUP_ID": "alog-tests",
        "TRITON_SERVER_URL": "localhost:8001",
        "MILVUS_URI": "http://localhost:19530",
        "MILVUS_TOKEN": "token",
        "OSS_ACCESS_KEY_ID": "access-key",
        "OSS_ACCESS_KEY_SECRET": "secret",
        "OSS_ENDPOINT": "oss.example.com",
        "OSS_BUCKET_NAME": "bucket",
        "STORAGE_BACKEND": "oss",
    }
    triton_baseline_environment = {
        "VISUAL_WORKER_PROCESSES": "0",
        "ALOG_WORKER_COUNT": "16",
        "TRITON_BATCH_TARGET_DEFAULT": "1",
        "TRITON_BATCH_WAIT_MS_DEFAULT": "0",
        "TRITON_BATCH_TARGETS": "{}",
        "TRITON_BATCH_WAIT_MS": "{}",
        "TRITON_QUEUE_TIMEOUT_SECONDS": "60",
        "TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS": "5",
        "TRITON_INFER_TIMEOUT_SECONDS": "30",
        "TRITON_CONFIG_TIMEOUT_SECONDS": "5",
        "TRITON_CLIENT_POOL_SIZE": "16",
        "TRITON_EXECUTOR_CONCURRENCY": "8",
        "TRITON_EXECUTOR_QUEUE_CAPACITY": "64",
        "TRITON_TRANSIENT_RETRIES": "0",
    }

    def _run_config(self, code, overrides):
        environment = {
            name: value for name, value in os.environ.items()
            if name in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL"}
        }
        environment.update(self.required_environment)
        environment.update(self.triton_baseline_environment)
        for name, value in overrides.items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        script = (
            "from unittest.mock import patch\n"
            "with patch('pydantic_settings.sources.DotEnvSettingsSource.__call__', return_value={}):\n"
            f"    exec({code!r})\n"
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=environment,
            capture_output=True,
            text=True,
        )

    @patch.object(subprocess, "run")
    def test_helper_always_supplies_explicit_triton_baseline(self, run):
        self._run_config("import core.config", {})

        environment = run.call_args.kwargs["env"]
        self.assertEqual(
            {
                name: environment[name]
                for name in self.triton_baseline_environment
            },
            self.triton_baseline_environment,
        )

    def test_field_declarations_have_safe_triton_defaults(self):
        result = self._run_config(
            "from core.config import Settings; "
            "fields = Settings.model_fields; "
            "assert fields['ALOG_WORKER_COUNT'].default == 32; "
            "assert fields['TRITON_BATCH_TARGET_DEFAULT'].default == 4; "
            "assert fields['TRITON_BATCH_WAIT_MS_DEFAULT'].default == 2.0; "
            "assert fields['TRITON_BATCH_TARGETS'].default_factory() == {}; "
            "assert fields['TRITON_BATCH_WAIT_MS'].default_factory() == {}; "
            "assert fields['TRITON_QUEUE_TIMEOUT_SECONDS'].default == 60.0; "
            "assert fields['TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS'].default == 5.0; "
            "assert fields['TRITON_INFER_TIMEOUT_SECONDS'].default == 30.0; "
            "assert fields['TRITON_CONFIG_TIMEOUT_SECONDS'].default == 5.0; "
            "assert fields['TRITON_CLIENT_POOL_SIZE'].default == 100; "
            "assert fields['TRITON_EXECUTOR_CONCURRENCY'].default == 80; "
            "assert fields['TRITON_EXECUTOR_QUEUE_CAPACITY'].default == 256; "
            "assert fields['TRITON_TRANSIENT_RETRIES'].default == 0",
            {},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_singleton_settings_loads_explicit_triton_baseline(self):
        result = self._run_config(
            "from core.config import settings; "
            "assert settings.ALOG_WORKER_COUNT == 16; "
            "assert settings.TRITON_BATCH_TARGET_DEFAULT == 1; "
            "assert settings.TRITON_BATCH_WAIT_MS_DEFAULT == 0.0; "
            "assert settings.TRITON_BATCH_TARGETS == {}; "
            "assert settings.TRITON_BATCH_WAIT_MS == {}; "
            "assert settings.TRITON_QUEUE_TIMEOUT_SECONDS == 60.0; "
            "assert settings.TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS == 5.0; "
            "assert settings.TRITON_INFER_TIMEOUT_SECONDS == 30.0; "
            "assert settings.TRITON_CONFIG_TIMEOUT_SECONDS == 5.0; "
            "assert settings.TRITON_CLIENT_POOL_SIZE == 16; "
            "assert settings.TRITON_EXECUTOR_CONCURRENCY == 8; "
            "assert settings.TRITON_EXECUTOR_QUEUE_CAPACITY == 64; "
            "assert settings.TRITON_TRANSIENT_RETRIES == 0",
            {},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_batch_above_one_requires_positive_wait(self):
        result = self._run_config(
            "import core.config",
            {"TRITON_BATCH_TARGETS": '{"person_pose": 4}'},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("person_pose", result.stderr)

    def test_person_concurrency_can_be_configured_independently_of_rpc_slots(self):
        result = self._run_config(
            "from core.config import settings; "
            "assert settings.PERSON_FRAME_CONCURRENCY == 3; "
            "assert settings.PERSON_GLOBAL_CONCURRENCY == 2; "
            "assert settings.TRITON_EXECUTOR_CONCURRENCY == 8",
            {"PERSON_FRAME_CONCURRENCY": "3", "PERSON_GLOBAL_CONCURRENCY": "2"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_person_concurrency_rejects_non_positive_and_fractional_limits(self):
        for name in ("PERSON_FRAME_CONCURRENCY", "PERSON_GLOBAL_CONCURRENCY"):
            for value in ("0", "-1", "1.5"):
                with self.subTest(name=name, value=value):
                    result = self._run_config("import core.config", {name: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(name, result.stderr)

    def test_face_concurrency_can_be_configured_independently_of_rpc_slots(self):
        result = self._run_config(
            "from core.config import settings; "
            "assert settings.FACE_FRAME_CONCURRENCY == 3; "
            "assert settings.FACE_GLOBAL_CONCURRENCY == 2; "
            "assert settings.TRITON_EXECUTOR_CONCURRENCY == 8",
            {"FACE_FRAME_CONCURRENCY": "3", "FACE_GLOBAL_CONCURRENCY": "2"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_face_concurrency_rejects_non_positive_and_fractional_limits(self):
        for name in ("FACE_FRAME_CONCURRENCY", "FACE_GLOBAL_CONCURRENCY"):
            for value in ("0", "-1", "1.5"):
                with self.subTest(name=name, value=value):
                    result = self._run_config("import core.config", {name: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(name, result.stderr)

    def test_unknown_business_key_is_rejected(self):
        result = self._run_config(
            "import core.config",
            {
                "TRITON_BATCH_TARGETS": '{"unknown_model": 2}',
                "TRITON_BATCH_WAIT_MS": '{"unknown_model": 5}',
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown_model", result.stderr)

    def test_json_maps_parse_and_per_model_batch_can_be_enabled_safely(self):
        result = self._run_config(
            "from core.config import settings; "
            "assert settings.TRITON_BATCH_TARGETS == {'person_pose': 4}; "
            "assert settings.TRITON_BATCH_WAIT_MS == {'person_pose': 5.0}",
            {
                "TRITON_BATCH_TARGETS": '{"person_pose": 4}',
                "TRITON_BATCH_WAIT_MS": '{"person_pose": 5}',
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_targets_must_be_at_least_one(self):
        result = self._run_config(
            "import core.config", {"TRITON_BATCH_TARGET_DEFAULT": "0"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TRITON_BATCH_TARGET_DEFAULT", result.stderr)

        result = self._run_config(
            "import core.config",
            {"TRITON_BATCH_TARGETS": '{"face_det": 0}'},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("face_det", result.stderr)

    def test_wait_must_be_non_negative(self):
        result = self._run_config(
            "import core.config", {"TRITON_BATCH_WAIT_MS_DEFAULT": "-1"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TRITON_BATCH_WAIT_MS_DEFAULT", result.stderr)

        result = self._run_config(
            "import core.config",
            {"TRITON_BATCH_WAIT_MS": '{"face_det": -1}'},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("face_det", result.stderr)

    def test_per_model_wait_rejects_non_finite_numbers(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                result = self._run_config(
                    "import core.config",
                    {"TRITON_BATCH_WAIT_MS": f'{{"face_det": {value}}}'},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("face_det", result.stderr)
                self.assertIn("finite", result.stderr.lower())

    def test_json_map_booleans_are_not_accepted_as_numbers(self):
        for name in ("TRITON_BATCH_TARGETS", "TRITON_BATCH_WAIT_MS"):
            with self.subTest(name=name):
                result = self._run_config(
                    "import core.config", {name: '{"face_det": true}'}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("face_det", result.stderr)

    def test_timeout_settings_reject_non_finite_numbers(self):
        for name in (
            "TRITON_QUEUE_TIMEOUT_SECONDS",
            "TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS",
            "TRITON_INFER_TIMEOUT_SECONDS",
            "TRITON_CONFIG_TIMEOUT_SECONDS",
        ):
            for value in ("NaN", "Infinity"):
                with self.subTest(name=name, value=value):
                    result = self._run_config("import core.config", {name: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(name, result.stderr)
                    self.assertIn("finite", result.stderr.lower())

    def test_default_wait_rejects_non_finite_numbers(self):
        for value in ("NaN", "Infinity"):
            with self.subTest(value=value):
                result = self._run_config(
                    "import core.config",
                    {"TRITON_BATCH_WAIT_MS_DEFAULT": value},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("TRITON_BATCH_WAIT_MS_DEFAULT", result.stderr)
                self.assertIn("finite", result.stderr.lower())

    def test_default_target_rejects_boolean_text(self):
        result = self._run_config(
            "import core.config", {"TRITON_BATCH_TARGET_DEFAULT": "true"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TRITON_BATCH_TARGET_DEFAULT", result.stderr)

    def test_wait_must_be_shorter_than_queue_timeout(self):
        result = self._run_config(
            "import core.config",
            {
                "TRITON_BATCH_TARGETS": '{"person_seg": 2}',
                "TRITON_BATCH_WAIT_MS": '{"person_seg": 5000}',
                "TRITON_QUEUE_TIMEOUT_SECONDS": "5",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("person_seg", result.stderr)

    def test_timeouts_and_pool_size_must_be_positive(self):
        for name in (
            "TRITON_QUEUE_TIMEOUT_SECONDS",
            "TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS",
            "TRITON_INFER_TIMEOUT_SECONDS",
            "TRITON_CONFIG_TIMEOUT_SECONDS",
            "TRITON_CLIENT_POOL_SIZE",
        ):
            with self.subTest(name=name):
                result = self._run_config("import core.config", {name: "0"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)

    def test_runtime_concurrency_counts_must_be_positive(self):
        for name in (
            "ALOG_WORKER_COUNT",
            "TRITON_EXECUTOR_CONCURRENCY",
            "TRITON_EXECUTOR_QUEUE_CAPACITY",
        ):
            with self.subTest(name=name):
                result = self._run_config("import core.config", {name: "0"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)

    def test_executor_concurrency_cannot_exceed_client_pool(self):
        result = self._run_config(
            "import core.config",
            {
                "TRITON_EXECUTOR_CONCURRENCY": "17",
                "TRITON_CLIENT_POOL_SIZE": "16",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TRITON_EXECUTOR_CONCURRENCY", result.stderr)
        self.assertIn("TRITON_CLIENT_POOL_SIZE", result.stderr)

    def test_client_acquire_timeout_must_be_positive_and_finite(self):
        for value in ("0", "NaN", "Infinity"):
            with self.subTest(value=value):
                result = self._run_config(
                    "import core.config",
                    {"TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS": value},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS", result.stderr
                )

    def test_transient_retries_accepts_only_zero_or_one(self):
        for value in ("-1", "2"):
            with self.subTest(value=value):
                result = self._run_config(
                    "import core.config", {"TRITON_TRANSIENT_RETRIES": value}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("TRITON_TRANSIENT_RETRIES", result.stderr)

    def test_root_validator_errors_have_no_empty_field_prefix(self):
        result = self._run_config(
            "import core.config",
            {"TRITON_BATCH_TARGETS": '{"unknown_model": 2}'},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("variables: Value error", result.stderr)
        self.assertNotIn("variables: : Value error", result.stderr)

    def test_error_taxonomy_has_expected_base_types(self):
        from core.infer.errors import (
            InferenceInputError,
            InferenceQueueTimeout,
            InferenceResponseError,
            InferenceServerTimeout,
            InferenceUnavailable,
            TritonContractError,
            TritonInferenceError,
        )

        for error_type in (
            TritonContractError,
            InferenceInputError,
            InferenceQueueTimeout,
            InferenceServerTimeout,
            InferenceUnavailable,
            InferenceResponseError,
        ):
            with self.subTest(error_type=error_type):
                self.assertTrue(issubclass(error_type, TritonInferenceError))
        self.assertTrue(issubclass(InferenceInputError, ValueError))
        self.assertTrue(issubclass(InferenceQueueTimeout, TimeoutError))
        self.assertTrue(issubclass(InferenceServerTimeout, TimeoutError))


if __name__ == "__main__":
    unittest.main()
