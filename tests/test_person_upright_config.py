import os
import subprocess
import sys
import unittest
from pathlib import Path


class PersonUprightConfigTests(unittest.TestCase):
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
    }

    def _run_config(self, code, overrides):
        environment = os.environ.copy()
        environment.update(self.required_environment)
        for name, value in overrides.items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=environment,
            capture_output=True,
            text=True,
        )

    def test_device_sns_are_trimmed_filtered_and_deduplicated(self):
        result = self._run_config(
            "from core.config import settings; "
            "print(','.join(sorted(settings.person_upright_device_sns)))",
            {"PERSON_UPRIGHT_DEVICE_SNS": " CAM-A, CAM-B ,,CAM-A "},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "CAM-A,CAM-B")

    def test_device_sns_are_exposed_as_one_direct_typed_field(self):
        code = (
            "from core.config import Settings; "
            "assert 'person_upright_device_sns' in type(Settings()).model_fields; "
            "assert 'PERSON_UPRIGHT_DEVICE_SNS' not in type(Settings()).model_fields; "
            "settings = Settings(person_upright_device_sns=['LOCAL-A', ' LOCAL-B ', 'LOCAL-A']); "
            "assert isinstance(settings.person_upright_device_sns, frozenset); "
            "assert settings.person_upright_device_sns == frozenset({'LOCAL-A', 'LOCAL-B'})"
        )
        result = self._run_config(code, {"PERSON_UPRIGHT_DEVICE_SNS": ""})

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_blank_device_sns_start_with_an_empty_frozenset(self):
        code = (
            "from core.config import settings; "
            "assert isinstance(settings.person_upright_device_sns, frozenset); "
            "assert not settings.person_upright_device_sns"
        )
        for value in ("", "   "):
            with self.subTest(value=repr(value)):
                result = self._run_config(
                    code,
                    {"PERSON_UPRIGHT_DEVICE_SNS": value},
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_unset_device_sns_keep_the_current_defaults(self):
        expected = {
            "541288EPS11C0013998A1ECF",
            "541288EPS11C0013B64023C7",
            "541288EPS11C0013C5722941",
            "541288EPS11C0014DE1A6344",
            "541288EPS11C00139B2E1216",
            "541288EPS11C00138A6DF15D",
            "541288EPS11C0013E581980E",
            "541288EPS11C00139764BD37",
        }
        code = (
            "from core.config import Settings; "
            "settings = Settings(_env_file=None); "
            f"assert settings.person_upright_device_sns == {expected!r}"
        )
        result = self._run_config(
            code,
            {"PERSON_UPRIGHT_DEVICE_SNS": None},
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_blank_optional_kafka_credentials_remain_none(self):
        code = (
            "from core.config import settings; "
            "assert settings.KAFKA_USERNAME is None; "
            "assert settings.KAFKA_PASSWORD is None"
        )
        result = self._run_config(
            code,
            {"KAFKA_USERNAME": "", "KAFKA_PASSWORD": "   "},
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_device_sns_reject_non_string_scalar(self):
        result = self._run_config(
            "from core.config import Settings; "
            "Settings(person_upright_device_sns=123)",
            {"PERSON_UPRIGHT_DEVICE_SNS": ""},
        )

        self.assertNotEqual(result.returncode, 0)

    def test_device_sns_reject_iterables_with_non_string_items(self):
        result = self._run_config(
            "from core.config import Settings; "
            "Settings(person_upright_device_sns=[1, None, True])",
            {"PERSON_UPRIGHT_DEVICE_SNS": ""},
        )

        self.assertNotEqual(result.returncode, 0)

    def test_device_sns_reject_bytes(self):
        result = self._run_config(
            "from core.config import Settings; "
            "Settings(person_upright_device_sns=b'CAM-A,CAM-B')",
            {"PERSON_UPRIGHT_DEVICE_SNS": ""},
        )

        self.assertNotEqual(result.returncode, 0)

    def test_device_sns_reject_explicit_none(self):
        result = self._run_config(
            "from core.config import Settings; "
            "Settings(person_upright_device_sns=None)",
            {"PERSON_UPRIGHT_DEVICE_SNS": ""},
        )

        self.assertNotEqual(result.returncode, 0)

    def test_required_string_setting_rejects_blank_values(self):
        result = self._run_config(
            "import core.config",
            {"PGSQL_USER": "   "},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PGSQL_USER", result.stderr)
        self.assertIn("must not be empty", result.stderr)

    def test_pydantic_settings_requires_nodecode_compatible_version(self):
        requirements = (Path(__file__).parents[1] / "requirements.txt").read_text()
        pydantic_settings_lines = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip().lower().startswith("pydantic-settings")
        ]

        self.assertEqual(len(pydantic_settings_lines), 1)
        self.assertRegex(
            pydantic_settings_lines[0], r"^pydantic-settings==\d+\.\d+\.\d+$"
        )
        pinned_version = tuple(
            int(part) for part in pydantic_settings_lines[0].split("==")[1].split(".")
        )
        self.assertGreaterEqual(pinned_version, (2, 7, 0))


if __name__ == "__main__":
    unittest.main()
