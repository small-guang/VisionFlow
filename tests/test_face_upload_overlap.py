import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_face_pipeline_concurrency as face_fixtures


class FaceUploadOverlapTests(unittest.IsolatedAsyncioTestCase):
    make_flow = face_fixtures.FacePipelineConcurrencyTests.make_flow
    run_flow = face_fixtures.FacePipelineConcurrencyTests.run_flow
    reached = face_fixtures.FacePipelineConcurrencyTests.reached

    def setUp(self):
        quality = patch(
            "core.face_keyframe_processor.meets_pre_extraction_quality_criteria",
            return_value=(True, None),
        )
        quality.start()
        self.addCleanup(quality.stop)

    async def test_quality_overlaps_upload_but_features_and_records_wait(self):
        flow = self.make_flow(1)
        uploading, release, quality_done = (asyncio.Event() for _ in range(3))
        original = flow.triton.extract_face_quality

        async def upload(*args):
            uploading.set()
            await release.wait()
            return "camera/face.jpg"

        async def quality(*args):
            await uploading.wait()
            result = await original(*args)
            quality_done.set()
            return result

        flow.triton.extract_face_quality = quality
        with patch("core.face_keyframe_processor.upload_crop_bytes", upload):
            task = asyncio.create_task(self.run_flow(flow))
            try:
                await self.reached(quality_done)
                self.assertEqual(flow.triton.feature_calls, 0)
                self.assertEqual(flow.buffer.face_detection_records, [])
                self.assertFalse(task.done())
            finally:
                release.set()
                await task
        self.assertEqual(flow.triton.feature_calls, 1)
        self.assertEqual(len(flow.buffer.face_recognition_records), 1)

    async def test_quality_early_returns_still_wait_for_uploaded_crop(self):
        for reason in ("keypoints", "quality"):
            with self.subTest(reason=reason):
                flow = self.make_flow(1)
                reached, release = asyncio.Event(), asyncio.Event()

                async def upload(*args):
                    await release.wait()
                    return "camera/rejected-face.jpg"

                async def reject(*args):
                    reached.set()
                    return None if reason == "keypoints" else SimpleNamespace(score=0.0)

                if reason == "keypoints":
                    flow.triton.extract_face_keypoints = reject
                else:
                    flow.triton.extract_face_quality = reject
                with patch("core.face_keyframe_processor.upload_crop_bytes", upload):
                    task = asyncio.create_task(self.run_flow(flow))
                    try:
                        await self.reached(reached)
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        self.assertEqual(flow.buffer.face_detection_records, [])
                    finally:
                        release.set()
                        await task
                self.assertEqual(flow.triton.feature_calls, 0)
                self.assertEqual(len(flow.buffer.face_detection_records), 1)
                self.assertIn("camera/rejected-face.jpg", str(flow.buffer.face_detection_records))

    async def test_failed_upload_still_blocks_face_features(self):
        flow = self.make_flow(1)

        async def upload(*args):
            return ""

        with patch("core.face_keyframe_processor.upload_crop_bytes", upload):
            observations = await self.run_flow(flow)
        self.assertEqual(flow.triton.feature_calls, 0)
        self.assertEqual(flow.buffer.face_recognition_records, [])
        self.assertEqual(observations[0].failure_reason, "Face crop upload failed")

    async def test_unhandled_quality_failure_drains_upload_without_records(self):
        flow = self.make_flow(1)
        uploading, stopped = asyncio.Event(), asyncio.Event()
        failure = RuntimeError("invalid keypoint response")

        async def upload(*args):
            uploading.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def keypoints(*args):
            await uploading.wait()
            raise failure

        flow.triton.extract_face_keypoints = keypoints
        with patch("core.face_keyframe_processor.upload_crop_bytes", upload):
            with self.assertRaises(RuntimeError) as caught:
                await self.run_flow(flow)
        self.assertIs(caught.exception, failure)
        self.assertTrue(stopped.is_set())
        self.assertEqual(flow.buffer.face_detection_records, [])
        self.assertEqual(flow.buffer.face_recognition_records, [])
