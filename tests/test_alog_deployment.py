import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _env_keys(path: Path) -> set[str]:
    keys = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0].strip())
    return keys


class VisionFlowDeploymentTests(unittest.TestCase):
    def test_dockerfile_packages_the_application_and_default_profile(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG PYTHON_IMAGE=python:3.12-slim-bookworm", dockerfile)
        for instruction in ("COPY main.py ./", "COPY config/ ./config/",
                            "COPY core/ ./core/", "COPY db/ ./db/", "COPY utils/ ./utils/"):
            self.assertIn(instruction, dockerfile)
        self.assertIn('CMD ["python", "main.py"]', dockerfile)
        for path in ("main.py", "core", "db", "utils", "config/algorithm.toml"):
            self.assertTrue((ROOT / path).exists(), path)
        self.assertIn(".env", (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    def test_env_template_explains_optional_algorithm_profile_without_threshold_keys(self):
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
        template = ROOT / ".env.example"
        self.assertIn("ALGORITHM_CONFIG_PATH", template.read_text(encoding="utf-8"))
        self.assertFalse(leaf_keys & _env_keys(template))

    def test_compose_renders_a_self_contained_service(self):
        if shutil.which("docker") is None:
            self.skipTest("Docker Compose CLI is not available")
        version = subprocess.run(
            ["docker", "compose", "version"], cwd=ROOT, capture_output=True, text=True
        )
        if version.returncode != 0:
            self.skipTest("Docker Compose CLI is not available")

        # A temporary project avoids reading or exposing a developer's real .env.
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            shutil.copyfile(ROOT / "docker-compose.yaml", project / "compose.yaml")
            shutil.copyfile(ROOT / "Dockerfile", project / "Dockerfile")
            (project / ".env").write_text(
                "DEPLOYMENT_SENTINEL=from-env-file\nALOG_IPC_MODE=host\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            for key in ("PYTHON_IMAGE", "ALOG_IPC_MODE", "COMPOSE_PROJECT_NAME"):
                environment.pop(key, None)
            result = subprocess.run(
                ["docker", "compose", "--project-directory", str(project),
                 "-f", str(project / "compose.yaml"), "config", "--format", "json"],
                cwd=project, env=environment, capture_output=True, text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        self.assertEqual(config["name"], "visionflow")
        service = config["services"]["visionflow"]
        self.assertEqual(service["build"]["dockerfile"], "Dockerfile")
        self.assertEqual(service["build"]["args"]["PYTHON_IMAGE"], "python:3.12-slim-bookworm")
        self.assertEqual(service["network_mode"], "host")
        self.assertEqual(service["ipc"], "host")
        self.assertEqual(service["restart"], "unless-stopped")
        self.assertEqual(service["environment"]["DEPLOYMENT_SENTINEL"], "from-env-file")
        self.assertFalse(any(mount["type"] == "bind" for mount in service["volumes"]))
        self.assertTrue(any(
            mount["type"] == "volume" and mount["source"] == "visionflow-logs"
            and mount["target"] == "/app/logs"
            for mount in service["volumes"]
        ))


if __name__ == "__main__":
    unittest.main()
