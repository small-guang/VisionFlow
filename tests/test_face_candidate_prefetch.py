"""Face 远程预取必须在提交时复核，并读取最新 pending。"""

import asyncio
from types import SimpleNamespace
import unittest

import numpy as np

from utils.reid.face_matching import FaceSampleMatch
from utils.reid.face_search import FaceSearchService


class ControlledGallery:
    def __init__(self):
        self.revision = 0
        self.active_writers = 0
        self.samples = []
        self.calls = []
        self.search_hook = None

    def candidate_revision(self):
        return self.revision, self.active_writers

    async def asearch_face_samples(self, embedding, *, device_ids=None):
        self.calls.append(device_ids)
        samples = list(self.samples)
        if self.search_hook is not None:
            await self.search_hook()
        return samples


class ControlledPending:
    def __init__(self):
        self.revision = 0
        self.samples = []
        self.calls = []
        self.search_hook = None

    def pending_face_removal_revision(self):
        return self.revision

    async def asearch_pending_face_samples(self, embedding, *, device_ids=None):
        self.calls.append(device_ids)
        samples = list(self.samples)
        if self.search_hook is not None:
            await self.search_hook()
        return samples


class FaceCandidatePrefetchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.gallery = ControlledGallery()
        self.pending = ControlledPending()
        self.service = FaceSearchService(
            face_embedding_db=self.gallery, feature_buffer=self.pending,
        )
        self.embedding = np.asarray([1., 0.], dtype=np.float32)

    @staticmethod
    def sample(face="face-1", score=.9, sample="sample-1", crop="crop-1.jpg"):
        return FaceSampleMatch(face, score, sample, crop)

    async def prefetch(self, **kwargs):
        state = self.service.snapshot_candidate_state()
        self.assertIsNotNone(state)
        return await self.service.prefetch(self.embedding, state=state, **kwargs)

    async def resolve(self, prefetched, **kwargs):
        return await self.service.resolve_prefetched(
            prefetched, self.embedding, threshold=.45, **kwargs,
        )

    async def test_stable_prefetch_merges_latest_pending_without_a_second_remote_search(self):
        prefetched = await self.prefetch()
        self.pending.samples.append(self.sample())
        result = await self.resolve(prefetched)
        self.assertEqual(result, ("face-1", .9, "face-1"))
        self.assertEqual(len(self.gallery.calls), 1)
        self.assertEqual(len(self.pending.calls), 1)

    async def test_concurrent_empty_prefetches_see_registration_under_shared_identity_lock(self):
        first, second = await asyncio.gather(self.prefetch(), self.prefetch())
        identity_lock = asyncio.Lock()
        created = []

        async def commit(prefetched):
            async with identity_lock:
                result = await self.resolve(prefetched)
                self.assertIsNotNone(result)
                face_id, *_ = result
                if face_id is None:
                    face_id = f"created-{len(created) + 1}"
                    created.append(face_id)
                    self.pending.samples.append(self.sample(face=face_id))
                return face_id

        self.assertEqual(await asyncio.gather(commit(first), commit(second)),
                         ["created-1", "created-1"])
        self.assertEqual(created, ["created-1"])
        self.assertEqual(len(self.gallery.calls), 2)

    async def test_normal_miss_is_a_tuple_and_not_invalid_prefetch(self):
        result = await self.resolve(await self.prefetch())
        self.assertEqual(result, (None, 0., ""))
        self.assertIsNotNone(result)

    async def test_missing_revision_capability_and_active_writers_disable_prefetch(self):
        for gallery, pending in ((SimpleNamespace(), self.pending),
                                 (self.gallery, SimpleNamespace())):
            with self.subTest(gallery=type(gallery).__name__, pending=type(pending).__name__):
                service = FaceSearchService(face_embedding_db=gallery, feature_buffer=pending)
                self.assertIsNone(service.snapshot_candidate_state())
        self.gallery.active_writers = 1
        self.assertIsNone(self.service.snapshot_candidate_state())

    async def test_gallery_changes_or_active_writer_invalidate_before_pending_search(self):
        prefetched = await self.prefetch()
        self.gallery.revision += 2
        self.assertIsNone(await self.resolve(prefetched))
        self.assertEqual(self.pending.calls, [])
        prefetched = await self.prefetch()
        self.gallery.active_writers = 1
        self.assertIsNone(await self.resolve(prefetched))
        self.assertEqual(self.pending.calls, [])

    async def test_pending_removal_invalidates_before_pending_search(self):
        prefetched = await self.prefetch()
        self.pending.revision += 1
        self.assertIsNone(await self.resolve(prefetched))
        self.assertEqual(self.pending.calls, [])

    async def test_change_while_awaiting_pending_snapshot_invalidates_after_read(self):
        for change in ("gallery", "pending", "active_writer"):
            with self.subTest(change=change):
                prefetched = await self.prefetch()

                async def mutate():
                    await asyncio.sleep(0)
                    if change == "gallery":
                        self.gallery.revision += 2
                    elif change == "pending":
                        self.pending.revision += 1
                    else:
                        self.gallery.active_writers = 1

                self.pending.search_hook = mutate
                self.assertIsNone(await self.resolve(prefetched))
                self.gallery.active_writers = 0

    async def test_pending_flush_during_prefetch_requires_full_search_and_retains_identity(self):
        self.pending.samples.append(self.sample())

        async def flush_during_search():
            self.gallery.samples.extend(self.pending.samples)
            self.pending.samples.clear()
            self.gallery.revision += 2
            self.pending.revision += 1

        self.gallery.search_hook = flush_during_search
        prefetched = await self.prefetch()
        self.assertEqual(prefetched.milvus, ())
        self.gallery.search_hook = None
        self.assertIsNone(await self.resolve(prefetched))
        self.assertEqual(await self.service.search(self.embedding, .45),
                         ("face-1", .9, "face-1"))
        self.assertEqual(len(self.gallery.calls), 2)

    async def test_query_parameter_or_embedding_change_rejects_reuse(self):
        devices = ["CAM001"]
        prefetched = await self.prefetch(device_ids=devices)
        devices.append("CAM002")
        self.assertEqual(prefetched.device_ids, ("CAM001",))
        self.assertEqual(self.gallery.calls, [["CAM001"]])
        self.assertIsNone(await self.resolve(prefetched, device_ids=devices))
        self.assertIsNone(await self.service.resolve_prefetched(
            prefetched, self.embedding.copy(), threshold=.45, device_ids=["CAM001"],
        ))
        self.assertEqual(self.pending.calls, [])

    async def test_gray_band_requires_two_independent_samples_across_sources(self):
        self.gallery.samples = [self.sample(score=.5)]
        prefetched = await self.prefetch()
        self.pending.samples = [self.sample(score=.49)]
        self.assertEqual(await self.resolve(prefetched), (None, .5, "face-1"))
        self.pending.samples = [self.sample(score=.49, sample="sample-2")]
        self.assertEqual(await self.resolve(prefetched), (None, .5, "face-1"))
        self.pending.samples = [self.sample(score=.49, sample="sample-2", crop="crop-2.jpg")]
        self.assertEqual(await self.resolve(prefetched), ("face-1", .5, "face-1"))

    async def test_equal_scores_preserve_milvus_before_pending_tie_order(self):
        self.gallery.samples = [self.sample(face="milvus-face")]
        self.pending.samples = [self.sample(face="pending-face")]
        self.assertEqual(await self.resolve(await self.prefetch()),
                         ("milvus-face", .9, "milvus-face"))

    async def test_prefetch_failure_propagates_instead_of_returning_a_miss(self):
        async def fail():
            raise RuntimeError("Milvus unavailable")

        self.gallery.search_hook = fail
        with self.assertRaisesRegex(RuntimeError, "Milvus unavailable"):
            await self.prefetch()
        self.assertEqual(self.pending.calls, [])

    async def test_full_search_reads_pending_first_and_survives_flush_during_remote_search(self):
        events = []
        self.pending.samples = [self.sample()]

        async def record_pending():
            events.append("pending")

        async def flush_during_remote():
            events.append("milvus")
            self.pending.samples.clear()
            self.pending.revision += 1

        self.pending.search_hook = record_pending
        self.gallery.search_hook = flush_during_remote
        self.assertEqual(await self.service.search(self.embedding, .45),
                         ("face-1", .9, "face-1"))
        self.assertEqual(events, ["pending", "milvus"])

    async def test_full_search_supports_legacy_buffer_without_pending_search(self):
        self.service.feature_buffer = SimpleNamespace()
        self.gallery.samples = [self.sample()]
        self.assertEqual(await self.service.search(self.embedding, .45),
                         ("face-1", .9, "face-1"))


if __name__ == "__main__":
    unittest.main()
