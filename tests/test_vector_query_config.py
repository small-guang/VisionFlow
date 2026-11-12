import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.application import _resources
from core.multiprocess.assembly import owner_resources, visual_resources, make_visual_config
from tests import test_application_config as config_helpers
from tests.test_multiprocess_resources import settings


class VectorQueryConfigTests(unittest.TestCase):
    def test_default_opt_out_and_invalid_limits(self):
        self.assertEqual(settings().MILVUS_QUERY_BATCH_SIZE, 4)
        self.assertEqual(settings().MILVUS_QUERY_CAPACITY, 128)
        for name, values in {
            "MILVUS_QUERY_BATCH_SIZE": (0, 33, True, 1.5),
            "MILVUS_QUERY_WAIT_MS": (-1, 11, True, float("nan"), float("inf")),
            "MILVUS_QUERY_CAPACITY": (0, True, 1.5),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    settings(**{name: value})
        with self.assertRaisesRegex(ValueError, "CAPACITY"):
            settings(MILVUS_QUERY_BATCH_SIZE=8, MILVUS_QUERY_CAPACITY=7)

    def test_environment_priority_and_owner_only_assembly(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("MILVUS_QUERY_BATCH_SIZE=4\nMILVUS_QUERY_WAIT_MS=0.5\n", encoding="utf-8")
            with patch.dict(os.environ, config_helpers.ApplicationConfigTests.required, clear=True):
                from core.config import Settings
                self.assertEqual(Settings(_env_file=path).MILVUS_QUERY_BATCH_SIZE, 4)
                with patch.dict(os.environ, {"MILVUS_QUERY_BATCH_SIZE": "8"}):
                    self.assertEqual(Settings(_env_file=path).MILVUS_QUERY_BATCH_SIZE, 8)
        config = settings(VISUAL_WORKER_PROCESSES=4, MILVUS_QUERY_BATCH_SIZE=8,
                          MILVUS_QUERY_WAIT_MS=0.5, MILVUS_QUERY_CAPACITY=64)
        for name, factory in (("ExecutionResources", _resources), ("OwnerExecutionResources", owner_resources)):
            with patch(f"core.execution.{name}") as mocked:
                factory(config)
            self.assertEqual(mocked.call_args.kwargs["query_batch_size"], 8)
            self.assertEqual(mocked.call_args.kwargs["query_wait_ms"], 0.5)
            self.assertEqual(mocked.call_args.kwargs["query_capacity"], 64)
        with patch("core.execution.VisualExecutionResources") as mocked:
            visual_resources(make_visual_config(config, 0))
        self.assertNotIn("query_batch_size", mocked.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
