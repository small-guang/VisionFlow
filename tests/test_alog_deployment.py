import json
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_ROOT = ROOT / "deploy" / "alog"


def _env_keys(path: Path) -> set[str]:
    keys = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0].strip())
    return keys


class ALogDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("docker") is None:
            raise unittest.SkipTest("Docker CLI is required for deployment checks")

        result = subprocess.run(
            ["docker", "compose", "version"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise unittest.SkipTest("Docker Compose is required for deployment checks")

    def test_compose_uses_a_published_image_without_source_mounts(self):
        compose_source = DEPLOYMENT_ROOT / "compose.yaml"
        self.assertTrue(compose_source.is_file(), "release compose file is missing")

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            shutil.copyfile(compose_source, project / "compose.yaml")
            (project / ".env").write_text(
                "ALOG_IMAGE=registry.example.test/camtellsvi-alog:1.2.3\n"
                "DEPLOYMENT_SENTINEL=from-env-file\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "--project-directory",
                    str(project),
                    "--env-file",
                    str(project / ".env"),
                    "-f",
                    str(project / "compose.yaml"),
                    "config",
                    "--format",
                    "json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        service = config["services"]["alog"]

        self.assertEqual(
            service["image"],
            "registry.example.test/camtellsvi-alog:1.2.3",
        )
        self.assertNotIn("build", service)
        self.assertNotIn("platform", service)
        self.assertNotIn("ports", service)
        self.assertEqual(service["network_mode"], "host")
        self.assertEqual(service["restart"], "unless-stopped")
        self.assertIs(service["init"], True)
        self.assertEqual(service["stop_grace_period"], "5m0s")
        self.assertEqual(
            service["logging"],
            {
                "driver": "json-file",
                "options": {"max-file": "5", "max-size": "100m"},
            },
        )
        self.assertEqual(service["environment"]["DEPLOYMENT_SENTINEL"], "from-env-file")

        mounts = service["volumes"]
        self.assertFalse(any(mount["type"] == "bind" for mount in mounts))
        self.assertTrue(
            any(
                mount["type"] == "volume"
                and mount["source"] == "alog-logs"
                and mount["target"] == "/app/logs"
                for mount in mounts
            )
        )

    def test_release_env_template_tracks_application_configuration(self):
        release_template = DEPLOYMENT_ROOT / ".env.example"
        self.assertTrue(release_template.is_file(), "release environment template is missing")

        application_keys = _env_keys(ROOT / ".env.example")
        release_keys = _env_keys(release_template)

        self.assertEqual(release_keys, application_keys | {"ALOG_IMAGE"})

    def test_both_images_package_the_default_algorithm_profile(self):
        for dockerfile in (ROOT / "Dockerfile-ALog", DEPLOYMENT_ROOT / "Dockerfile"):
            with self.subTest(dockerfile=dockerfile):
                self.assertIn("COPY config/ ./config/", dockerfile.read_text(encoding="utf-8"))

    def test_env_templates_explain_optional_algorithm_profile_without_threshold_keys(self):
        algorithm = tomllib.loads(
            (ROOT / "config" / "algorithm.toml").read_text(encoding="utf-8")
        )
        leaf_keys = set()

        def collect_leaf_keys(section):
            for key, value in section.items():
                if isinstance(value, dict):
                    collect_leaf_keys(value)
                else:
                    leaf_keys.add(key.upper())

        collect_leaf_keys(algorithm)
        for template in (ROOT / ".env.example", DEPLOYMENT_ROOT / ".env.example"):
            with self.subTest(template=template):
                contents = template.read_text(encoding="utf-8")
                self.assertIn("ALGORITHM_CONFIG_PATH", contents)
                self.assertFalse(leaf_keys & _env_keys(template))

    def test_release_dockerfile_passes_multi_platform_build_checks(self):
        dockerfile = DEPLOYMENT_ROOT / "Dockerfile"
        self.assertTrue(dockerfile.is_file(), "release Dockerfile is missing")

        for platform in ("linux/amd64", "linux/arm64"):
            with self.subTest(platform=platform):
                result = subprocess.run(
                    [
                        "docker",
                        "buildx",
                        "build",
                        "--check",
                        "--platform",
                        platform,
                        "-f",
                        str(dockerfile),
                        str(ROOT),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
