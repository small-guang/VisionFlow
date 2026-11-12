"""多进程容量守恒、角色隔离和配置校验；全部使用本地 fake。"""

import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.execution import OwnerExecutionResources, VisualExecutionResources
from core.multiprocess.assembly import make_visual_config
from tests import test_application_config as config_helpers
from tests.test_system_shared_memory import ShmTestCase


def settings(**values):
    with patch.dict(os.environ, config_helpers.ApplicationConfigTests.required, clear=True):
        from core.config import Settings
        return Settings(_env_file=None, **values)


class MultiprocessConfigTests(unittest.TestCase):
    def test_default_visual_workers_have_disjoint_conserved_resources(self):
        config = settings()
        self.assertEqual(config.VISUAL_WORKER_PROCESSES, 4)
        self.assertEqual(config.IMAGE_CPU_WORKERS, 4)
        self.assertEqual(config.MULTIPROCESS_IPC_MAX_PAYLOAD_BYTES, 4 << 20)
        views = [make_visual_config(config, index) for index in range(4)]
        self.assertEqual([view.IMAGE_CPU_WORKERS for view in views], [1, 1, 1, 1])
        self.assertEqual([view.STORAGE_IO_WORKERS for view in views], [16, 16, 16, 16])
        self.assertEqual([view.STORAGE_DOWNLOAD_RESERVED_WORKERS for view in views], [1, 1, 1, 1])

    def test_environment_overrides_dotenv_for_opt_in(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("VISUAL_WORKER_PROCESSES=2\n", encoding="utf-8")
            with patch.dict(os.environ, config_helpers.ApplicationConfigTests.required, clear=True):
                from core.config import Settings
                self.assertEqual(Settings(_env_file=path).VISUAL_WORKER_PROCESSES, 2)
                with patch.dict(os.environ, {"VISUAL_WORKER_PROCESSES": "0"}):
                    self.assertEqual(Settings(_env_file=path).VISUAL_WORKER_PROCESSES, 0)

    def test_invalid_counts_and_limits_fail_before_resources_start(self):
        for value in (-1, 5, 1.5, True, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings(VISUAL_WORKER_PROCESSES=value)
        for name in ("MULTIPROCESS_START_TIMEOUT_SECONDS", "MULTIPROCESS_STOP_TIMEOUT_SECONDS"):
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    settings(**{name: value})
        with self.assertRaisesRegex(ValueError, "MAX_PAYLOAD_BYTES"):
            settings(MULTIPROCESS_IPC_MAX_BYTES=1024)

    def test_undersized_global_capacity_is_rejected_instead_of_rounded_up(self):
        for name in (
            "IMAGE_CPU_WORKERS", "STORAGE_IO_WORKERS", "TRITON_EXECUTOR_CONCURRENCY",
            "TRITON_EXECUTOR_QUEUE_CAPACITY", "FACE_GLOBAL_CONCURRENCY",
            "PERSON_GLOBAL_CONCURRENCY", "PERSON_FACE_CONCURRENCY",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "VISUAL_WORKER_PROCESSES"):
                settings(VISUAL_WORKER_PROCESSES=2, STORAGE_DOWNLOAD_RESERVED_WORKERS=0, **{name: 1})

    def test_odd_budgets_are_conserved_and_view_survives_spawn_serialization(self):
        config = settings(
            VISUAL_WORKER_PROCESSES=2, IMAGE_CPU_WORKERS=5, STORAGE_IO_WORKERS=7,
            TRITON_CLIENT_POOL_SIZE=11, TRITON_EXECUTOR_CONCURRENCY=9,
            TRITON_EXECUTOR_QUEUE_CAPACITY=5, FACE_GLOBAL_CONCURRENCY=7,
            PERSON_GLOBAL_CONCURRENCY=9, PERSON_FACE_CONCURRENCY=5,
            TRITON_SYSTEM_SHM_SLOTS=5,
        )
        views = [make_visual_config(config, index) for index in range(2)]
        for name in (
            "IMAGE_CPU_WORKERS", "STORAGE_IO_WORKERS", "TRITON_CLIENT_POOL_SIZE",
            "TRITON_EXECUTOR_CONCURRENCY", "TRITON_EXECUTOR_QUEUE_CAPACITY",
            "FACE_GLOBAL_CONCURRENCY", "PERSON_GLOBAL_CONCURRENCY",
            "PERSON_FACE_CONCURRENCY", "TRITON_SYSTEM_SHM_SLOTS",
        ):
            self.assertEqual(sum(getattr(view, name) for view in views), getattr(config, name), name)
        self.assertEqual(sum(view.TRITON_SYSTEM_SHM_COPY_WORKERS for view in views), 4)
        self.assertEqual([view.IMAGE_CPU_WORKERS for view in views], [3, 2])
        self.assertEqual([view.FACE_FRAME_CONCURRENCY for view in views], [4, 3])
        self.assertTrue(all(view.ALOG_MAX_INFLIGHT_FRAMES == config.ALOG_MAX_INFLIGHT_FRAMES for view in views))
        restored = pickle.loads(pickle.dumps(views[1]))
        self.assertEqual(restored.IMAGE_CPU_WORKERS, 2)
        self.assertEqual(restored.TRITON_SERVER_URL, config.TRITON_SERVER_URL)
        self.assertEqual(restored.MP_WORKER_INDEX, 1)
        self.assertEqual(config.IMAGE_CPU_WORKERS, 5)
        with self.assertRaisesRegex(ValueError, "already"):
            make_visual_config(views[0], 0)

    def test_explicit_storage_and_face_budgets_are_divided_once(self):
        config = settings(VISUAL_WORKER_PROCESSES=2, STORAGE_IO_WORKERS=7,
                          PERSON_GLOBAL_CONCURRENCY=5, TRITON_SYSTEM_SHM_SLOTS=1)
        first, second = (make_visual_config(config, index) for index in range(2))
        self.assertEqual((first.STORAGE_IO_WORKERS, second.STORAGE_IO_WORKERS), (4, 3))
        self.assertEqual((first.PERSON_FACE_CONCURRENCY, second.PERSON_FACE_CONCURRENCY), (3, 2))
        self.assertEqual((first.TRITON_SYSTEM_SHM_SLOTS, second.TRITON_SYSTEM_SHM_SLOTS), (1, 0))
        self.assertEqual((first.TRITON_SYSTEM_SHM_COPY_WORKERS, second.TRITON_SYSTEM_SHM_COPY_WORKERS), (1, None))

    def test_all_supported_worker_counts_conserve_odd_budgets_and_frame_authority(self):
        fields = {
            "IMAGE_CPU_WORKERS": 5, "STORAGE_IO_WORKERS": 7,
            "TRITON_CLIENT_POOL_SIZE": 11, "TRITON_EXECUTOR_CONCURRENCY": 9,
            "TRITON_EXECUTOR_QUEUE_CAPACITY": 5, "FACE_GLOBAL_CONCURRENCY": 7,
            "PERSON_GLOBAL_CONCURRENCY": 9, "PERSON_FACE_CONCURRENCY": 5,
        }
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                config = settings(VISUAL_WORKER_PROCESSES=count, STORAGE_DOWNLOAD_RESERVED_WORKERS=0, **fields)
                views = [make_visual_config(config, index) for index in range(count)]
                for name, total in fields.items():
                    shares = [getattr(view, name) for view in views]
                    self.assertEqual(sum(shares), total, name)
                    self.assertLessEqual(max(shares) - min(shares), 1, name)
                    self.assertGreaterEqual(min(shares), 1, name)
                for index, view in enumerate(views):
                    restored = pickle.loads(pickle.dumps(view))
                    self.assertEqual(restored.MP_WORKER_INDEX, index)
                    self.assertEqual(restored.MP_WORKER_COUNT, count)
                    self.assertEqual(restored.STORAGE_IO_WORKERS, view.STORAGE_IO_WORKERS)
                    self.assertEqual(restored.ALOG_MAX_INFLIGHT_FRAMES, config.ALOG_MAX_INFLIGHT_FRAMES)
                    self.assertEqual(restored.FRAME_MEMORY_BUDGET_BYTES, config.FRAME_MEMORY_BUDGET_BYTES)
                    for prefix in ("FACE", "PERSON"):
                        self.assertEqual(getattr(view, f"{prefix}_FRAME_CONCURRENCY"), min(
                            getattr(config, f"{prefix}_FRAME_CONCURRENCY"),
                            getattr(view, f"{prefix}_GLOBAL_CONCURRENCY"),
                        ))

    def test_small_shm_budgets_do_not_create_unassigned_copy_threads(self):
        for count in (1, 2, 3, 4):
            for slots in (0, 1, 3, 5, 16):
                with self.subTest(count=count, slots=slots):
                    config = settings(VISUAL_WORKER_PROCESSES=count, TRITON_SYSTEM_SHM_SLOTS=slots)
                    views = [make_visual_config(config, index) for index in range(count)]
                    self.assertEqual(sum(view.TRITON_SYSTEM_SHM_SLOTS for view in views), slots)
                    self.assertEqual(sum(view.TRITON_SYSTEM_SHM_COPY_WORKERS or 0 for view in views), min(4, slots))
                    self.assertTrue(all(view.TRITON_SYSTEM_SHM_SLOT_BYTES == config.TRITON_SYSTEM_SHM_SLOT_BYTES for view in views))
                    for view in views:
                        if view.TRITON_SYSTEM_SHM_SLOTS:
                            self.assertGreaterEqual(view.TRITON_SYSTEM_SHM_COPY_WORKERS, 1)
                            self.assertLessEqual(view.TRITON_SYSTEM_SHM_COPY_WORKERS, min(4, view.TRITON_SYSTEM_SHM_SLOTS))
                        else:
                            self.assertIsNone(view.TRITON_SYSTEM_SHM_COPY_WORKERS)

    def test_worker_counts_and_minimum_budgets_cannot_bypass_the_config_boundary(self):
        config = settings()
        for count in (0, 5, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "between 1 and 4"):
                make_visual_config(config, 0, worker_count=count)
        for count in (2, 3, 4):
            for name in ("IMAGE_CPU_WORKERS", "STORAGE_IO_WORKERS", "TRITON_CLIENT_POOL_SIZE",
                         "TRITON_EXECUTOR_CONCURRENCY", "TRITON_EXECUTOR_QUEUE_CAPACITY",
                         "FACE_GLOBAL_CONCURRENCY", "PERSON_GLOBAL_CONCURRENCY", "PERSON_FACE_CONCURRENCY"):
                with self.subTest(count=count, name=name), self.assertRaisesRegex(ValueError, "VISUAL_WORKER_PROCESSES"):
                    settings(VISUAL_WORKER_PROCESSES=count, STORAGE_DOWNLOAD_RESERVED_WORKERS=0,
                             **{name: count - 1})


class MultiprocessRoleResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_roles_create_only_their_resource_domains_and_close_them(self):
        visual = VisualExecutionResources(image_workers=2, storage_workers=3)
        owner = OwnerExecutionResources(search_workers=4, write_workers=2)
        try:
            self.assertEqual(set(visual.queue_snapshot()), {"image_cpu", "storage_io"})
            self.assertEqual(set(owner.queue_snapshot()), {"vector_search", "vector_write", "vector_queries"})
            self.assertFalse(hasattr(visual, "vector_write"))
            self.assertFalse(hasattr(owner, "image_cpu"))
            self.assertEqual(await visual.image_cpu.run(pow, 3, 2), 9)
            self.assertEqual(await owner.vector_search.run(pow, 2, 3), 8)
        finally:
            await visual.close()
            await owner.close()
        self.assertTrue(all(item["closed"] for item in visual.queue_snapshot().values()))
        self.assertTrue(all(item["closed"] for item in owner.queue_snapshot().values()))


class MultiprocessShmCopyBudgetTests(ShmTestCase):
    async def test_transport_uses_assigned_copy_capacity_with_more_regions(self):
        transport = self.make_transport(shm_slots=8, shm_copy_workers=2)
        await transport.start()
        self.assertEqual(transport._shm_pool._copy_pool.workers, 2)
        self.assertEqual(len(self.state.registered), 8)
        await self.infer(transport)

    async def test_invalid_copy_budget_is_rejected_before_allocating_regions(self):
        for value in (0, 5, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_pool(slots=8, copy_workers=value)
        self.assertEqual(self.state.registered, [])

    async def test_one_three_and_four_transports_keep_the_total_copy_pool_capacity(self):
        for count in (1, 3, 4):
            with self.subTest(count=count):
                config = settings(VISUAL_WORKER_PROCESSES=count, TRITON_SYSTEM_SHM_SLOTS=16)
                transports = []
                before_registered = len(self.state.registered)
                try:
                    for index in range(count):
                        view = make_visual_config(config, index)
                        transport = self.make_transport(
                            shm_slots=view.TRITON_SYSTEM_SHM_SLOTS,
                            shm_copy_workers=view.TRITON_SYSTEM_SHM_COPY_WORKERS,
                        )
                        transports.append(transport)
                        await transport.start()
                    self.assertEqual(sum(item._shm_pool._copy_pool.workers for item in transports), 4)
                    self.assertEqual(len(self.state.registered) - before_registered, 16)
                finally:
                    for transport in transports:
                        await transport.close()


if __name__ == "__main__":
    unittest.main()
