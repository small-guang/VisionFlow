"""SBS 预取状态必须覆盖真实写线程和 pending 消失窗口。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import AsyncMock, patch

from utils.buffers.sbs_feature_buffer import SbsFeatureBuffer


class GalleryCandidateRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package_name = "sbs_candidate_revision_under_test"
        package = types.ModuleType(cls.package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "db" / "zilliz")]
        pymilvus = types.ModuleType("pymilvus")
        pymilvus.DataType = types.SimpleNamespace(VARCHAR="VARCHAR")
        pymilvus.MilvusClient = object
        config = types.ModuleType("core.config")
        config.settings = None
        module_name = f"{cls.package_name}.SBS_feature"
        spec = importlib.util.spec_from_file_location(
            module_name, Path(package.__path__[0]) / "SBS_feature.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[cls.package_name] = package
        sys.modules[module_name] = cls.module
        with patch.dict(sys.modules, {"pymilvus": pymilvus, "core.config": config}):
            spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == cls.package_name or name.startswith(cls.package_name + "."):
                sys.modules.pop(name, None)

    def make_gallery(self):
        with patch.object(self.module.PersonVectorGalleryBase, "__init__", return_value=None):
            gallery = self.module.SbsReidGallery()
        gallery.collection_name = "alogSBS"
        gallery.client = types.SimpleNamespace(
            query=lambda **kwargs: [],
            insert=lambda **kwargs: {"insert_count": 1},
            delete=lambda **kwargs: None,
        )
        return gallery

    @staticmethod
    def item():
        return {
            "prototypeId": "new-prototype",
            "reId": "reid-1",
            "faceId": "face-1",
            "bodyCompleteness": "full_body",
            "embedding": [1.0, 0.0],
            "deviceId": "device-1",
            "pictureVectorId": 1,
            "timestamp": 1000,
            "qualityScore": 9.0,
        }

    @staticmethod
    def insert_args():
        return {
            "faceId": "face-1",
            "reId": "reid-1",
            "bodyCompleteness": "full_body",
            "cropUrl": "crop.jpg",
            "embedding": [1.0, 0.0],
            "deviceId": "device-1",
            "trackId": 0,
            "keyframeId": 1,
            "personRect": {},
            "timestamp": 1000,
            "qualityScore": 9.0,
        }

    def plan(self, *, replace=False):
        return types.SimpleNamespace(
            action=(self.module.PROTOTYPE_ACTION_REPLACE if replace else "insert"),
            reason="test",
            writes_vector=True,
            replace_id=("old-prototype" if replace else None),
        )

    def test_every_sync_write_entry_tracks_the_actual_writer(self):
        entries = (
            ("apply_prototype_updates", lambda gallery: gallery.apply_prototype_updates([self.item()])),
            ("_apply_prototype_update", lambda gallery: gallery._apply_prototype_update(self.item())),
            ("_insert_prototype", lambda gallery: gallery._insert_prototype(**self.insert_args())),
            ("_insert_person_columns", lambda gallery: gallery._insert_person_columns(["reId"], ["reid-1"])),
            ("insert_embeddings", lambda gallery: gallery.insert_embeddings(1, [1.0, 0.0], "device-1", "image.jpg", 1000)),
        )
        for name, call in entries:
            with self.subTest(entry=name):
                gallery = self.make_gallery()
                started = threading.Event()
                release = threading.Event()

                def insert(**kwargs):
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release insert")
                    return {"insert_count": 1}

                gallery.client.insert = insert
                self.assertEqual(gallery.candidate_revision(), (0, 0))
                with patch.object(self.module, "plan_prototype_update", return_value=self.plan()):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(call, gallery)
                        try:
                            self.assertTrue(started.wait(5))
                            # 嵌套 batch→update→insert 仍然只计一个真实线程。
                            self.assertEqual(gallery.candidate_revision(), (1, 1))
                        finally:
                            release.set()
                        future.result(timeout=5)
                self.assertEqual(gallery.candidate_revision(), (2, 0))

    def test_replacement_stays_active_across_delete_and_insert(self):
        gallery = self.make_gallery()
        delete_started = threading.Event()
        insert_started = threading.Event()
        release_delete = threading.Event()
        release_insert = threading.Event()

        def delete(**kwargs):
            delete_started.set()
            if not release_delete.wait(5):
                raise TimeoutError("test did not release delete")

        def insert(**kwargs):
            insert_started.set()
            if not release_insert.wait(5):
                raise TimeoutError("test did not release insert")
            return {"insert_count": 1}

        gallery.client.delete = delete
        gallery.client.insert = insert
        with patch.object(self.module, "plan_prototype_update", return_value=self.plan(replace=True)):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(gallery.apply_prototype_updates, [self.item()])
                try:
                    self.assertTrue(delete_started.wait(5))
                    during_delete = gallery.candidate_revision()
                    release_delete.set()
                    self.assertTrue(insert_started.wait(5))
                    self.assertEqual(during_delete, (1, 1))
                    self.assertEqual(gallery.candidate_revision(), during_delete)
                finally:
                    release_delete.set()
                    release_insert.set()
                self.assertTrue(future.result(timeout=5)[0]["persisted"])
        self.assertEqual(gallery.candidate_revision(), (2, 0))

    def test_waiting_writer_is_active_and_snapshot_does_not_wait_for_io_lock(self):
        gallery = self.make_gallery()
        observed = threading.Event()
        snapshots = []

        # 在写线程完成状态注册之后、等待 prototype lock 时读取快照。
        class ObservedLock:
            def __enter__(self):
                observed.set()
                original_lock.acquire()

            def __exit__(self, *args):
                original_lock.release()

        original_lock = threading.Lock()
        original_lock.acquire()
        gallery._prototype_update_lock = ObservedLock()
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(gallery.apply_prototype_updates, [])
            try:
                self.assertTrue(observed.wait(5))
                snapshots.append(
                    executor.submit(gallery.candidate_revision).result(timeout=2)
                )
                observed.clear()
                second = executor.submit(gallery.apply_prototype_updates, [])
                self.assertTrue(observed.wait(5))
                snapshots.append(
                    executor.submit(gallery.candidate_revision).result(timeout=2)
                )
            finally:
                original_lock.release()
            first.result(timeout=5)
            second.result(timeout=5)
        self.assertEqual(snapshots, [(1, 1), (2, 2)])
        self.assertEqual(gallery.candidate_revision(), (4, 0))

    def test_failed_or_aborted_write_always_invalidates_and_releases_state(self):
        gallery = self.make_gallery()
        with patch.object(gallery, "_apply_prototype_update", side_effect=RuntimeError("failed")):
            result = gallery.apply_prototype_updates([self.item()])
        self.assertTrue(result[0]["retry"])
        self.assertEqual(gallery.candidate_revision(), (2, 0))
        with patch.object(gallery, "_apply_prototype_update", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                gallery.apply_prototype_updates([self.item()])
        self.assertEqual(gallery.candidate_revision(), (4, 0))

    def test_cancelled_async_waiter_keeps_real_write_thread_active(self):
        gallery = self.make_gallery()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def update(item):
            started.set()
            try:
                if not release.wait(5):
                    raise TimeoutError("test did not release update")
                return {"item": item, "persisted": True}
            finally:
                finished.set()

        async def run():
            with patch.object(gallery, "_apply_prototype_update", side_effect=update):
                task = asyncio.create_task(gallery.aapply_prototype_updates([self.item()]))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 5))
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertFalse(finished.is_set())
                    self.assertEqual(gallery.candidate_revision(), (1, 1))
                finally:
                    release.set()
                self.assertTrue(await asyncio.to_thread(finished.wait, 5))

        asyncio.run(run())  # 也等待 default executor 的真实线程退出。
        self.assertEqual(gallery.candidate_revision(), (2, 0))

    def test_candidate_search_explicitly_requires_strong_consistency(self):
        gallery = self.make_gallery()
        calls = []

        def search(**kwargs):
            calls.append(kwargs)
            return [[]]

        gallery.client.search = search
        gallery._search_single_gallery_body_type(
            [1.0, 0.0], "full_body", "full_body", "timestamp >= 0"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["consistency_level"], "Strong")
        self.assertEqual(gallery.candidate_revision(), (0, 0))

    def test_new_only_gallery_initializes_a_shared_revision_state(self):
        gallery = self.module.SbsReidGallery.__new__(self.module.SbsReidGallery)
        with ThreadPoolExecutor(max_workers=4) as executor:
            states = list(executor.map(lambda _: gallery._candidate_write_state(), range(16)))
        self.assertTrue(all(state is states[0] for state in states))
        self.assertEqual(gallery.candidate_revision(), (0, 0))


class PendingRemovalRevisionTests(unittest.IsolatedAsyncioTestCase):
    def make_buffer(self, vector_db=None):
        return SbsFeatureBuffer(
            vector_db=vector_db or types.SimpleNamespace(aapply_prototype_updates=AsyncMock()),
            max_batch_size=100,
            on_vectors_inserted=AsyncMock(),
        )

    async def test_pending_candidates_keep_durable_prototype_id_for_deduplication(self):
        buffer = self.make_buffer()
        common = {
            "faceId": "face-1",
            "bodyCompleteness": "full_body",
            "embedding": [1.0, 0.0],
            "deviceId": "device-1",
            "timestamp": 1000,
        }
        durable_id = "e7bef3cb-0239-4d73-a613-dbe73ff047cf"
        first = {**common, "reId": "reid-1", "prototypeId": durable_id}
        legacy = {**common, "reId": "reid-2"}
        await buffer._add_pending(first)
        await buffer._add_pending(legacy)
        candidates, *_ = await buffer.asearch_candidates_with_hit_info(
            [1.0, 0.0], "full_body", query_timestamp=1000
        )
        self.assertEqual(
            [candidate.prototype_id for candidate in candidates],
            [durable_id, legacy["__pendingKey"]],
        )
        self.assertEqual(buffer.pending_removal_revision(), 0)

    async def test_only_actual_removal_changes_revision(self):
        buffer = self.make_buffer()
        first = {"reId": "reid-1"}
        second = {"reId": "reid-2"}
        self.assertEqual(buffer.pending_removal_revision(), 0)
        await buffer._add_pending(first)
        await buffer._add_pending(second)
        self.assertEqual(buffer.pending_removal_revision(), 0)
        await buffer._remove_pending([first, first, {}])
        self.assertEqual(buffer.pending_removal_revision(), 1)
        self.assertEqual(await buffer._pending_snapshot(), [second])
        await buffer._remove_pending([first, {}])
        await buffer._remove_pending([])
        self.assertEqual(buffer.pending_removal_revision(), 1)
        await buffer._remove_pending([second])
        self.assertEqual(buffer.pending_removal_revision(), 2)

    async def test_removal_and_revision_wait_for_the_same_pending_lock(self):
        buffer = self.make_buffer()
        item = {"reId": "reid-1"}
        await buffer._add_pending(item)
        async with buffer._pending_lock:
            removal = asyncio.create_task(buffer._remove_pending([item]))
            await asyncio.sleep(0)
            self.assertFalse(removal.done())
            self.assertIn(item["__pendingKey"], buffer.pending)
            self.assertEqual(buffer.pending_removal_revision(), 0)
        await removal
        self.assertNotIn(item["__pendingKey"], buffer.pending)
        self.assertEqual(buffer.pending_removal_revision(), 1)

    async def test_flush_retry_keeps_pending_until_a_completed_outcome(self):
        vector_db = types.SimpleNamespace(aapply_prototype_updates=AsyncMock())
        buffer = self.make_buffer(vector_db)
        item = {"reId": "reid-1", "faceId": "face-1"}
        await buffer._add_pending(item)
        await buffer.queue.put(item)
        vector_db.aapply_prototype_updates.return_value = [
            {"item": item, "retry": True, "persisted": False}
        ]
        await buffer.flush()
        self.assertEqual(buffer.pending_removal_revision(), 0)
        self.assertEqual(await buffer._pending_snapshot(), [item])
        vector_db.aapply_prototype_updates.return_value = [
            {"item": item, "persisted": True, "insert_result": None}
        ]
        await buffer.flush()
        self.assertEqual(buffer.pending_removal_revision(), 1)
        self.assertEqual(await buffer._pending_snapshot(), [])


if __name__ == "__main__":
    unittest.main()
