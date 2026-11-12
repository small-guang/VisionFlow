import os
import subprocess
import sys
import unittest


class PersonFaceReuseConfigTests(unittest.TestCase):
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

    def test_explicit_reuse_mode_is_exposed_and_legacy_stage_is_ignored(self):
        code = (
            "from core.config import settings; "
            "print(settings.person_face_reuse_mode); "
            "print(hasattr(settings, 'person_face_reuse_max_stage'))"
        )
        result = self._run_config(
            code,
            {
                "PERSON_FACE_REUSE_MODE": "off",
                "PERSON_FACE_REUSE_MAX_STAGE": "analysis",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["off", "False"])

    def test_unset_reuse_mode_defaults_to_enabled(self):
        code = (
            "from core.config import Settings; "
            "settings = Settings(_env_file=None); "
            "print(settings.person_face_reuse_mode)"
        )
        result = self._run_config(
            code,
            {
                "PERSON_FACE_REUSE_MODE": None,
                "PERSON_FACE_REUSE_MAX_STAGE": None,
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["on"])

    def test_reuse_mode_rejects_observe_unsupported_blank_and_uppercase_values(self):
        cases = (
            ("PERSON_FACE_REUSE_MODE", "enabled"),
            ("PERSON_FACE_REUSE_MODE", "observe"),
            ("PERSON_FACE_REUSE_MODE", ""),
            ("PERSON_FACE_REUSE_MODE", "ON"),
        )
        for name, value in cases:
            with self.subTest(name=name, value=repr(value)):
                result = self._run_config(
                    "import core.config",
                    {
                        "PERSON_FACE_REUSE_MODE": None,
                        "PERSON_FACE_REUSE_MAX_STAGE": None,
                        name: value,
                    },
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)


if __name__ == "__main__":
    unittest.main()
