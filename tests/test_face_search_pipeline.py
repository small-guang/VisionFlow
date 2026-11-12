import asyncio
from types import SimpleNamespace
import unittest

import numpy as np

from core.face_keyframe_processor import FaceKeyframeProcessor, _PreparedFace
from core.face_observation import FrameFaceObservation
from core.face_search_pipeline import prefetch_face_search, resolve_face_search
from core.keyframe_processor import KeyframeProcessor
from tests.test_face_candidate_prefetch import ControlledGallery, ControlledPending
from tests.test_keyframe_flow import DummyLogger
from utils.keyframe.detection_helpers import KeyframeContext
from utils.keyframe.keyframe_flow_helpers import QUALITY_LEVEL_ENROLL
from utils.reid.face_matching import FaceSampleMatch
from utils.reid.face_search import FaceSearchService


class PendingBuffer(ControlledPending):
    def __init__(self):
        super().__init__()
        self.features = []
        self.records = []
        self.withheld = []

    async def add_face_detection_record(self, record):
        pass

    async def add_face_recognition_record(self, record):
        self.records.append(record)

    async def add_face_feature(self, **feature):
        self.features.append(feature)
        self.samples.append(FaceSampleMatch(
            feature['face_id'], .99, feature['ai_id'], feature['face_crop_path'],
        ))
        if 'new_face_recognition_record' in feature:
            self.withheld.append(feature['new_face_recognition_record'])


class FaceSearchPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_remote_reads_then_atomic_registration_and_newborn_gate(self):
        gallery, buffer, lock = ControlledGallery(), PendingBuffer(), asyncio.Lock()
        both_reading, release = asyncio.Event(), asyncio.Event()
        async def remote_hook():
            self.assertFalse(lock.locked())
            if len(gallery.calls) == 2:
                both_reading.set()
            await release.wait()
        gallery.search_hook = remote_hook
        service = FaceSearchService(face_embedding_db=gallery, feature_buffer=buffer)
        processor = FaceKeyframeProcessor(
            triton_client=None, feature_buffer=buffer, face_identity_lock=lock,
            search_faces_with_pending=service.search, logger=DummyLogger(),
            face_search_service=service,
        )
        embedding = np.array([1., 0.], dtype=np.float32)
        ctx = KeyframeContext(sn='camera', timestamp=1, object_name='image.jpg', keyframe_id='frame')
        def prepared(index):
            return _PreparedFace(
                ai_id=str(index), detect_time=1, confidence=.9,
                face_rect={'x':0,'y':0,'w':10,'h':10}, detection_info='',
                crop_path=f'crop-{index}.jpg', quality_score=8., quality_level=QUALITY_LEVEL_ENROLL,
                observation=FrameFaceObservation(source_rect=(0,0,10,10),confidence=.9),
                recognition_embedding=embedding,
            )
        tasks = [asyncio.create_task(processor._commit_face(ctx, str(i), i, 'image.jpg', prepared(i))) for i in range(2)]
        try:
            await asyncio.wait_for(both_reading.wait(), 1)
            release.set()
            result = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(result[0].assigned_face_id, result[1].assigned_face_id)
        self.assertEqual(len(buffer.withheld), 1)
        self.assertEqual([r['faceStatus'] for r in buffer.records], [2])
        self.assertEqual(len(gallery.calls), 2)

    async def test_person_fallback_remote_query_uses_same_lock_free_path(self):
        gallery, buffer, lock = ControlledGallery(), PendingBuffer(), asyncio.Lock()
        async def assert_unlocked():
            self.assertFalse(lock.locked())
        gallery.search_hook = assert_unlocked
        processor = KeyframeProcessor.__new__(KeyframeProcessor)
        processor._face_identity_lock = lock
        processor.face_embedding_db = gallery
        processor.feature_buffer = buffer
        processor._face_search_service = FaceSearchService(face_embedding_db=gallery, feature_buffer=buffer)
        self.assertEqual(await processor._search_faces_with_pending_locked(np.array([1.,0.]), .45, None), (None, 0., ''))
        self.assertEqual(len(gallery.calls), 1)

    async def test_prefetch_failure_falls_back_under_lock_and_cancellation_propagates(self):
        lock = asyncio.Lock()
        async def failure(*args, **kwargs):
            raise OSError('query unavailable')
        service = SimpleNamespace(snapshot_candidate_state=lambda: object(), prefetch=failure)
        value = np.array([1.,0.])
        self.assertIsNone(await prefetch_face_search(service, lock, value))
        async def fallback(*args, **kwargs):
            self.assertTrue(lock.locked())
            return 'existing', .9, 'existing'
        async with lock:
            self.assertEqual(await resolve_face_search(service, None, value, threshold=.45, device_ids=None, fallback=fallback), ('existing',.9,'existing'))
        async def cancelled(*args, **kwargs):
            raise asyncio.CancelledError
        service.prefetch = cancelled
        with self.assertRaises(asyncio.CancelledError):
            await prefetch_face_search(service, lock, value)
        self.assertFalse(lock.locked())
