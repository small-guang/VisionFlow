import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.application import _resources
from core.multiprocess.assembly import make_visual_config, visual_resources
from tests import test_application_config as config_helpers
from tests.test_multiprocess_resources import settings


class StorageAdmissionConfigTests(unittest.TestCase):
    def test_default_and_invalid_global_reservations(self):
        self.assertEqual(settings().STORAGE_DOWNLOAD_RESERVED_WORKERS, 4)
        for reserve in (-1, 8, 9, True, 1.5):
            with self.subTest(reserve=reserve), self.assertRaises(ValueError):
                settings(STORAGE_IO_WORKERS=8, STORAGE_DOWNLOAD_RESERVED_WORKERS=reserve)
        self.assertEqual(settings(VISUAL_WORKER_PROCESSES=0, STORAGE_IO_WORKERS=8,
                                  STORAGE_DOWNLOAD_RESERVED_WORKERS=7).STORAGE_IO_WORKERS, 8)

    def test_each_visual_has_both_directions_and_global_reserve_is_conserved(self):
        for count in (1, 2, 3, 4):
            for total in (2 * count, 2 * count + 3, 64):
                for reserve in (0, count, total - count):
                    with self.subTest(count=count, total=total, reserve=reserve):
                        config = settings(VISUAL_WORKER_PROCESSES=count, STORAGE_IO_WORKERS=total,
                                          STORAGE_DOWNLOAD_RESERVED_WORKERS=reserve)
                        views = [make_visual_config(config, index) for index in range(count)]
                        self.assertEqual(sum(view.STORAGE_IO_WORKERS for view in views), total)
                        self.assertEqual(sum(view.STORAGE_DOWNLOAD_RESERVED_WORKERS for view in views), reserve)
                        for view in views:
                            self.assertLess(view.STORAGE_DOWNLOAD_RESERVED_WORKERS, view.STORAGE_IO_WORKERS)
                            if reserve:
                                self.assertGreaterEqual(view.STORAGE_DOWNLOAD_RESERVED_WORKERS, 1)
                            self.assertEqual(pickle.loads(pickle.dumps(view)).STORAGE_DOWNLOAD_RESERVED_WORKERS,
                                             view.STORAGE_DOWNLOAD_RESERVED_WORKERS)
        for count, total, reserve in ((4, 64, 3), (4, 64, 61), (4, 7, 4)):
            with self.subTest(count=count, total=total, reserve=reserve), self.assertRaisesRegex(ValueError, "VISUAL_WORKER_PROCESSES"):
                settings(VISUAL_WORKER_PROCESSES=count, STORAGE_IO_WORKERS=total,
                         STORAGE_DOWNLOAD_RESERVED_WORKERS=reserve)

    def test_environment_priority_and_factories_use_assigned_reserve(self):
        from core.config import Settings

        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("STORAGE_IO_WORKERS=64\nSTORAGE_DOWNLOAD_RESERVED_WORKERS=4\n", encoding="utf-8")
            with patch.dict(os.environ, config_helpers.ApplicationConfigTests.required, clear=True):
                self.assertEqual(Settings(_env_file=path).STORAGE_DOWNLOAD_RESERVED_WORKERS, 4)
                with patch.dict(os.environ, {"STORAGE_DOWNLOAD_RESERVED_WORKERS": "8"}):
                    self.assertEqual(Settings(_env_file=path).STORAGE_DOWNLOAD_RESERVED_WORKERS, 8)
        config = settings(VISUAL_WORKER_PROCESSES=4, STORAGE_IO_WORKERS=64,
                          STORAGE_DOWNLOAD_RESERVED_WORKERS=8)
        with patch("core.execution.ExecutionResources") as factory:
            _resources(config)
        self.assertEqual(factory.call_args.kwargs["storage_download_reserved"], 8)
        with patch("core.execution.VisualExecutionResources") as factory:
            visual_resources(make_visual_config(config, 0))
        self.assertEqual(factory.call_args.kwargs["storage_workers"], 16)
        self.assertEqual(factory.call_args.kwargs["storage_download_reserved"], 2)


if __name__ == "__main__":
    unittest.main()
