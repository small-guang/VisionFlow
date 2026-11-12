import asyncio
from dataclasses import replace
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from core.performance import RuntimePerformance
from core.execution import AsyncWorkExecutor
from core.person_pipeline.models import PersonObservation, PreparedPerson
from core.person_pipeline.preparation import PersonPreparationService
from tests.person_quality_test_support import (
    PERSON_TEST_IMAGE_SHAPE,
    _evaluate_person_test_quality,
    _person_test_mask,
    _person_test_segment,
)
from utils.storage.image_util import get_image_by_oss_name


class QuietLogger:
    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class PersonPreparationPoolTests(unittest.IsolatedAsyncioTestCase):
    def make_executor(self, workers=2, *, metrics=None):
        executor = AsyncWorkExecutor(workers, name="image_cpu", metrics=metrics)
        self.addAsyncCleanup(executor.close)
        return executor

    async def test_crop_runs_off_loop_before_visual_and_encoding(self):
        entered, release = threading.Event(), threading.Event()
        stages = []
        result = PreparedPerson(None, SimpleNamespace(image=b"pixels"), None, "")
        owner_thread = threading.get_ident()

        class Service(PersonPreparationService):
            def _prepare_crop(self, *args):
                stages.append(("crop", threading.get_ident()))
                entered.set()
                if not release.wait(timeout=2):
                    raise TimeoutError("test did not release crop preparation")
                return result

            async def _apply_visual_quality(self, ctx, prepared):
                stages.append(("visual", threading.get_ident()))
                self_test.assertIs(prepared, result)
                return prepared

            async def _run_image_operation(self, operation, *args):
                stages.append(("encode", threading.get_ident()))
                self_test.assertEqual(args, (b"pixels",))
                return b"encoded"

            async def upload_prepared(self, ctx, storage, prepared):
                stages.append(("upload", threading.get_ident()))
                self_test.assertEqual(prepared.image_bytes, b"encoded")
                return prepared

        self_test = self
        metrics = RuntimePerformance()
        pool = self.make_executor(metrics=metrics)
        service = Service(
            logger=QuietLogger(), runtime=SimpleNamespace(encode_image=object()),
            upright_device_sns=(), image_executor=pool,
        )
        task = asyncio.create_task(service.prepare(None, None, None, None, []))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            await asyncio.wait_for(asyncio.sleep(0.005), 0.2)
            self.assertEqual([name for name, _ in stages], ["crop"])
            release.set()
            self.assertEqual(await task, replace(result, image_bytes=b"encoded"))
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual([name for name, _ in stages], ["crop", "visual", "encode", "upload"])
        self.assertNotEqual(stages[0][1], owner_thread)
        self.assertEqual([thread_id for _, thread_id in stages[1:]], [owner_thread] * 3)
        self.assertEqual(pool.workers, 2)
        self.assertEqual(metrics.snapshot()["operations"]["person_crop_prepare"]["count"], 1)

    async def test_threaded_crop_matches_original_geometry_quality_and_pixels(self):
        executor = self.make_executor()
        image = np.arange(np.prod(PERSON_TEST_IMAGE_SHAPE), dtype=np.uint8).reshape(
            PERSON_TEST_IMAGE_SHAPE
        )
        original_image = image.copy()
        mask = _person_test_mask()
        original_mask = mask.copy()
        segment = _person_test_segment(mask=mask)
        observation = PersonObservation(
            segment_index=0, segment=segment, people_rect=segment["rect"],
            mask_pixel_count=int(mask.sum()), confidence=segment["conf"],
            person_keypoints=segment["pose"]["keypoints"],
            person_orientation_result=SimpleNamespace(label="front"),
            quality_decision=_evaluate_person_test_quality(segment),
            ai_id="test", detect_time=0,
        )
        ctx = SimpleNamespace(sn="test-device", is_pano_keyframe=False)
        service = PersonPreparationService(
            logger=QuietLogger(), runtime=SimpleNamespace(encode_image=object()),
            upright_device_sns=(), image_executor=executor,
        )
        expected = service._prepare_crop(ctx, image, observation, [segment])

        async def unchanged(ctx, prepared):
            return prepared

        async def no_upload(ctx, storage, prepared):
            return prepared

        with (
            patch.object(service, "_apply_visual_quality", side_effect=unchanged),
            patch.object(service, "_run_image_operation", new=AsyncMock(return_value=b"encoded")),
            patch.object(service, "upload_prepared", side_effect=no_upload),
        ):
            actual = await service.prepare(ctx, image, None, observation, [segment])
        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual.cropped, expected.cropped)
        np.testing.assert_array_equal(actual.mask_crop, expected.mask_crop)
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(mask, original_mask)
        self.assertEqual(actual.crop_result.requested_xyxy, expected.crop_result.requested_xyxy)
        self.assertEqual(actual.quality_decision, expected.quality_decision)
        self.assertEqual(actual.crop_info, expected.crop_info)

    async def test_crop_and_decode_share_the_injected_capacity(self):
        entered, release = threading.Event(), threading.Event()
        decoded = threading.Event()

        def blocking():
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release crop preparation")

        executor = self.make_executor(1)
        storage = AsyncMock()
        storage.get_object.return_value = b"image"

        def decode(data):
            decoded.set()
            return data

        task = asyncio.create_task(executor.run(blocking))
        decode_task = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            with patch("utils.storage.image_util._decode_image", side_effect=decode):
                decode_task = asyncio.create_task(get_image_by_oss_name(
                    "key", storage, image_executor=executor,
                ))
                await asyncio.sleep(0.02)
                self.assertFalse(decoded.is_set())
                self.assertFalse(decode_task.done())
                release.set()
                self.assertEqual(await asyncio.wait_for(decode_task, 1), (b"image", b"image"))
                await task
        finally:
            release.set()
            await asyncio.gather(*([task, decode_task] if decode_task else [task]), return_exceptions=True)

    async def test_missing_executor_is_rejected_during_assembly(self):
        with self.assertRaisesRegex(ValueError, "image_executor is required"):
            PersonPreparationService(
                logger=QuietLogger(), runtime=SimpleNamespace(),
                upright_device_sns=(), image_executor=None,
            )


if __name__ == "__main__":
    unittest.main()
