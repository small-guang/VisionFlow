import os
import subprocess
import sys
import unittest


class StorageConfigTests(unittest.TestCase):
    environment = {
        "PGSQL_USER": "user", "PGSQL_PSWD": "password",
        "PGSQL_HOST": "localhost", "PGSQL_PORT": "5432", "PGSQL_DB": "alog",
        "KAFKA_BROKER_ADDR": "localhost", "KAFKA_BROKER_PORT": "9092",
        "KAFKA_GROUP_ID": "storage-tests", "KAFKA_SECURITY_PROTOCOL": "PLAINTEXT",
        "TRITON_SERVER_URL": "localhost:8001", "MILVUS_URI": "http://localhost:19530",
        "MILVUS_TOKEN": "token", "STORAGE_BACKEND": "oss",
        "OSS_ACCESS_KEY_ID": "test-access", "OSS_ACCESS_KEY_SECRET": "test-secret",
        "OSS_ENDPOINT": "https://oss.example.com", "OSS_BUCKET_NAME": "test-images",
        "MINIO_ENDPOINT": "", "MINIO_ACCESS_KEY": "", "MINIO_SECRET_KEY": "",
        "MINIO_BUCKET_NAME": "", "MINIO_SECURE": "false",
    }
    minio_environment = {
        "STORAGE_BACKEND": "minio", "MINIO_ENDPOINT": "minio.example.com:9000",
        "MINIO_ACCESS_KEY": "minio-test-access", "MINIO_SECRET_KEY": "private-test-value",
        "MINIO_BUCKET_NAME": "camtells-images",
        "OSS_ACCESS_KEY_ID": "", "OSS_ACCESS_KEY_SECRET": "",
        "OSS_ENDPOINT": "", "OSS_BUCKET_NAME": "",
    }

    def run_config(self, overrides=None, code="from core.config import settings; print(settings.STORAGE_BACKEND)"):
        # 只保留解释器启动所需环境，配置完全来自本用例的显式输入。
        environment = {
            name: value for name, value in os.environ.items()
            if name in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL"}
        }
        environment.update(self.environment)
        for name, value in (overrides or {}).items():
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
            [sys.executable, "-c", script], env=environment,
            capture_output=True, text=True,
        )

    def test_unset_backend_keeps_existing_oss_deployment(self):
        result = self.run_config({"STORAGE_BACKEND": None},
            "from core.config import Settings; print(Settings(_env_file=None).STORAGE_BACKEND)")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "oss")

    def test_minio_can_start_without_oss_credentials(self):
        result = self.run_config(self.minio_environment,
            "from core.config import settings; print(settings.STORAGE_BACKEND); print(settings.MINIO_SECURE)")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["minio", "False"])

    def test_oss_ignores_blank_inactive_minio_credentials(self):
        result = self.run_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "oss")

    def test_oss_does_not_validate_inactive_minio_tls_option(self):
        for value in ("", "garbage"):
            with self.subTest(value=value):
                result = self.run_config({"MINIO_SECURE": value})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "oss")

    def test_minio_requires_a_valid_tls_option(self):
        for value in ("", "garbage"):
            with self.subTest(value=value):
                result = self.run_config({**self.minio_environment, "MINIO_SECURE": value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("MINIO_SECURE", result.stderr)

    def test_selected_backend_requires_each_connection_setting(self):
        for backend, values, required in (
            ("oss", {}, ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET_NAME")),
            ("minio", self.minio_environment, ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_NAME")),
        ):
            for name in required:
                with self.subTest(backend=backend, setting=name):
                    result = self.run_config({**values, name: "   "})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(name, result.stderr)
                    self.assertNotIn("private-test-value", result.stderr)

    def test_invalid_or_empty_backend_fails_instead_of_falling_back(self):
        for backend in ("s3", "MINIO", "", "   "):
            with self.subTest(backend=backend):
                result = self.run_config({"STORAGE_BACKEND": backend})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("STORAGE_BACKEND", result.stderr)

    def test_minio_endpoint_rejects_scheme_path_credentials_and_invalid_port(self):
        for endpoint in ("http://localhost:9000", "localhost:9000/bucket",
                         "user:private-test-value@localhost:9000", "localhost:0",
                         "localhost:65536", "local host:9000", "localhost:bad"):
            with self.subTest(endpoint=endpoint):
                result = self.run_config({**self.minio_environment, "MINIO_ENDPOINT": endpoint})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("MINIO_ENDPOINT", result.stderr)
                self.assertNotIn("private-test-value", result.stderr)

    def test_minio_supports_tls_and_bracketed_ipv6_endpoint(self):
        result = self.run_config({**self.minio_environment,
            "MINIO_ENDPOINT": "[::1]:9000", "MINIO_SECURE": "true"},
            "from core.config import settings; print(settings.MINIO_ENDPOINT); print(settings.MINIO_SECURE)")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["[::1]:9000", "True"])


if __name__ == "__main__":
    unittest.main()
