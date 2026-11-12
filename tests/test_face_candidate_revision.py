"""Face 候选版本覆盖真实写线程和 pending 移除，而非异步等待者寿命。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_milvus_records as buffer_fixtures
from tests.test_face_gallery import MemoryMilvusClient, load_gallery_modules


class FaceGalleryCandidateRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.picture_module, cls.face_module = load_gallery_modules()

    def make_gallery(self):
        with patch.object(self.face_module.MilvusVectorDB, "__init__", return_value=None):
            gallery = self.face_module.FaceGallery(max_vectors_per_face=2)
        gallery.collection_name = "alogFace"
        gallery.dim = 2
        gallery.client = MemoryMilvusClient()
        return gallery

    @staticmethod
    def record(sample="sample-1"):
        return dict(
            faceId="face-1", embedding=[1., 0.], deviceId="camera", trackId=0,
            keyframeId=1, faceRect={}, faceKeyPoints=[], faceCropPath=f"{sample}.jpg",
            timestamp=100, qualityScore=8., sampleId=sample,
        )

    def test_sync_entries_and_nested_writes_track_one_actual_writer(self):
        for entry in ("insert_embeddings", "_update_one"):
            with self.subTest(entry=entry):
                gallery = self.make_gallery()
                started = threading.Event()
                release = threading.Event()
                original_upsert = gallery.client.upsert

                def blocked_upsert(**kwargs):
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release upsert")
                    return original_upsert(**kwargs)

                gallery.client.upsert = blocked_upsert
                call = (lambda: gallery.insert_embeddings(**self.record())) if entry == "insert_embeddings" else (
                    lambda: gallery._update_one(self.record())
                )
                self.assertEqual(gallery.candidate_revision(), (0, 0))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(call)
                    try:
                        self.assertTrue(started.wait(5))
                        self.assertEqual(gallery.candidate_revision(), (1, 1))
                    finally:
                        release.set()
                    future.result(timeout=5)
                self.assertEqual(gallery.candidate_revision(), (2, 0))

    def test_waiting_writers_are_active_without_blocking_revision_reads(self):
        gallery = self.make_gallery()
        original_lock = threading.Lock()
        observed = threading.Event()

        class ObservedLock:
            def __enter__(self):
                observed.set()
                original_lock.acquire()

            def __exit__(self, *args):
                original_lock.release()

        gallery._quality_update_lock = ObservedLock()
        original_lock.acquire()
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(gallery.insert_embeddings, **self.record())
            try:
                self.assertTrue(observed.wait(5))
                self.assertEqual(executor.submit(gallery.candidate_revision).result(timeout=2), (1, 1))
                observed.clear()
                second = executor.submit(gallery.insert_embeddings, **self.record("sample-2"))
                self.assertTrue(observed.wait(5))
                self.assertEqual(executor.submit(gallery.candidate_revision).result(timeout=2), (2, 2))
            finally:
                original_lock.release()
            first.result(timeout=5)
            second.result(timeout=5)
        self.assertEqual(gallery.candidate_revision(), (4, 0))

    def test_partial_write_failure_and_base_exception_invalidate_and_release_state(self):
        gallery = self.make_gallery()
        gallery.client.lose_write_reply = True
        with self.assertRaisesRegex(RuntimeError, "reply lost"):
            gallery.insert_embeddings(**self.record())
        self.assertEqual(len(gallery.client.rows), 1)
        self.assertEqual(gallery.candidate_revision(), (2, 0))
        with patch.object(gallery.client, "query_iterator", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                gallery._update_one(self.record("sample-2"))
        self.assertEqual(gallery.candidate_revision(), (4, 0))

    def test_capacity_delete_and_upsert_share_one_active_revision(self):
        gallery = self.make_gallery()
        for index in range(3):
            gallery.client.rows[index] = dict(self.record(f"old-{index}"), id=index)
        states = []
        original_delete = gallery.client.delete
        original_upsert = gallery.client.upsert

        def delete(**kwargs):
            states.append(gallery.candidate_revision())
            return original_delete(**kwargs)

        def upsert(**kwargs):
            states.append(gallery.candidate_revision())
            return original_upsert(**kwargs)

        gallery.client.delete = delete
        gallery.client.upsert = upsert
        gallery.insert_embeddings(**dict(self.record(), qualityScore=9.))
        self.assertEqual(states, [(1, 1), (1, 1)])
        self.assertEqual(gallery.candidate_revision(), (2, 0))

    def test_cancelled_async_waiter_does_not_mark_actual_thread_idle(self):
        gallery = self.make_gallery()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        original_upsert = gallery.client.upsert

        def blocked_upsert(**kwargs):
            started.set()
            try:
                if not release.wait(5):
                    raise TimeoutError("test did not release upsert")
                return original_upsert(**kwargs)
            finally:
                finished.set()

        gallery.client.upsert = blocked_upsert

        async def run():
            task = asyncio.create_task(gallery.ainsert_embeddings(**self.record()))
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

        asyncio.run(run())
        self.assertEqual(gallery.candidate_revision(), (2, 0))

    def test_search_preserves_strong_consistency_and_does_not_change_revision(self):
        gallery = self.make_gallery()
        with patch.object(gallery.client, "search", return_value=[[]], create=True) as search:
            self.assertEqual(gallery.search_face_samples(np.asarray([1., 0.])), [])
        self.assertEqual(search.call_args.kwargs["consistency_level"], "Strong")
        self.assertEqual(gallery.candidate_revision(), (0, 0))

    def test_new_only_gallery_initializes_one_shared_revision_state(self):
        gallery = self.face_module.FaceGallery.__new__(self.face_module.FaceGallery)
        with ThreadPoolExecutor(max_workers=4) as executor:
            states = list(executor.map(lambda _: gallery._candidate_write_state(), range(16)))
        self.assertTrue(all(state is states[0] for state in states))
        self.assertEqual(gallery.candidate_revision(), (0, 0))


class FacePendingRemovalRevisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = buffer_fixtures.FeatureBufferRoutingTests()
        self.fixture.setUp()
        self.buffer = self.fixture.buffer_module.FeatureBuffer()

    def tearDown(self):
        self.fixture.tearDown()

    async def test_only_actual_face_pending_removal_changes_revision(self):
        buffer = self.buffer
        first, second = {"faceId": "face-1"}, {"faceId": "face-2"}
        self.assertEqual(buffer.pending_face_removal_revision(), 0)
        await buffer._add_pending(buffer.pending_faces, first)
        await buffer._add_pending(buffer.pending_faces, second)
        self.assertEqual(buffer.pending_face_removal_revision(), 0)
        await buffer._remove_pending(buffer.pending_faces, [first, first, {}])
        self.assertEqual(buffer.pending_face_removal_revision(), 1)
        self.assertEqual(await buffer._pending_snapshot(buffer.pending_faces), [second])
        await buffer._remove_pending(buffer.pending_faces, [first, {}])
        await buffer._remove_pending(buffer.pending_faces, [])
        self.assertEqual(buffer.pending_face_removal_revision(), 1)
        other_pending = {1: {}}
        await buffer._remove_pending(other_pending, [{"__pendingKey": 1}])
        self.assertEqual(buffer.pending_face_removal_revision(), 1)
        await buffer._remove_pending(buffer.pending_faces, [second])
        self.assertEqual(buffer.pending_face_removal_revision(), 2)

    async def test_removal_and_revision_share_pending_lock(self):
        buffer = self.buffer
        item = {"faceId": "face-1"}
        await buffer._add_pending(buffer.pending_faces, item)
        async with buffer._pending_lock:
            removal = asyncio.create_task(buffer._remove_pending(buffer.pending_faces, [item]))
            await asyncio.sleep(0)
            self.assertFalse(removal.done())
            self.assertEqual(buffer.pending_face_removal_revision(), 0)
            self.assertIn(item["__pendingKey"], buffer.pending_faces)
        await removal
        self.assertEqual(buffer.pending_face_removal_revision(), 1)
        self.assertEqual(buffer.pending_faces, {})

    async def test_failed_flush_keeps_revision_until_durable_outcome(self):
        buffer = self.buffer
        await buffer.add_face_feature(
            face_id="face-1", embedding=[.1, .2], device_id="CAM001",
            keyframe_id="frame-1", picture_vector_id=42, keyframe_url="frame.jpg",
            face_rect={}, face_key_points=[], face_crop_path="crop.jpg",
            timestamp=171, ai_id="ai-1", quality_score=8.25,
            new_face_recognition_record={"id": "recognition-1"},
        )
        with patch.object(self.fixture.face_db, "ainsert_embeddings", side_effect=RuntimeError("failed")):
            await buffer._flush_faces()
        self.assertEqual(buffer.pending_face_removal_revision(), 0)
        self.assertEqual(len(buffer.pending_faces), 1)
        self.assertTrue(buffer.face_recognition_record_buffer.empty())
        await buffer._flush_faces()
        self.assertEqual(buffer.pending_face_removal_revision(), 1)
        self.assertEqual(buffer.pending_faces, {})
        self.assertEqual(buffer.face_recognition_record_buffer.qsize(), 1)


if __name__ == "__main__":
    unittest.main()
