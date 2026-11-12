from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.infer.adapters.person import PersonSegmentationAdapter
from core.infer.errors import InferenceResponseError
from core.execution import AsyncWorkExecutor
from core.performance import RuntimePerformance
from utils.person_mask import PersonMask
from utils.storage.image_util import get_image_by_oss_name


class ImageDecodeTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_decoding_excludes_storage_wait_and_preserves_the_image(self):
        clock = SimpleNamespace(now=0.0)
        metrics = RuntimePerformance(clock=lambda: clock.now)
        image = np.zeros((3, 4, 3), dtype=np.uint8)

        class Storage:
            async def get_object(self, name):
                clock.now += 0.200
                return b"encoded-image"

        def decode(*args):
            clock.now += 0.025
            return image

        pool = AsyncWorkExecutor(1, name="image_cpu")
        pool._pool._clock = lambda: clock.now
        try:
            with patch("utils.storage.image_util.runtime_performance", metrics), patch(
                "utils.storage.image_decode.cv2.imdecode", side_effect=decode
            ):
                result, encoded = await get_image_by_oss_name(
                    "private/object/key", Storage(), image_executor=pool,
                )
        finally:
            await pool.close()

        self.assertIs(result, image)
        self.assertEqual(encoded, b"encoded-image")
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["operations"]["image_decode"], {
            "count": 1, "errors": 0, "total_ms": 25.0,
            "avg_ms": 25.0, "max_ms": 25.0,
        })
        self.assertNotIn("private/object/key", str(snapshot))

    async def test_invalid_image_keeps_its_error_and_is_counted(self):
        class Storage:
            async def get_object(self, name):
                return b"invalid-image"

        metrics = RuntimePerformance()
        pool = AsyncWorkExecutor(1, name="image_cpu")
        try:
            with patch("utils.storage.image_util.runtime_performance", metrics), patch(
                "utils.storage.image_decode.cv2.imdecode", return_value=None
            ):
                with self.assertRaisesRegex(ValueError, "图片解码失败"):
                    await get_image_by_oss_name(
                        "private/object/key", Storage(), image_executor=pool,
                    )
        finally:
            await pool.close()
        self.assertEqual(metrics.snapshot()["operations"]["image_decode"]["errors"], 1)


class PersonSegmentationTimingTests(unittest.TestCase):
    def test_unpack_timing_covers_mask_resize_and_pixel_count(self):
        clock = SimpleNamespace(now=0.0)
        metrics = RuntimePerformance(clock=lambda: clock.now)
        adapter = PersonSegmentationAdapter()
        request = adapter.normalize(np.zeros((4, 4, 3), dtype=np.uint8))
        outputs = {
            "DETECTIONS": np.asarray([[[0, 0, 4, 4, 0.9, 0]]], dtype=np.float32),
            "NUM_DETECTIONS": np.asarray([[1]], dtype=np.int32),
            "MASK_CROPS_PACKED": np.full((1, 1, 160, 20), 255, dtype=np.uint8),
        }
        real_resize, real_count = cv2.resize, PersonMask.count

        def resize(*args, **kwargs):
            clock.now += 0.015
            return real_resize(*args, **kwargs)

        def count(mask, *args, **kwargs):
            clock.now += 0.005
            return real_count(mask, *args, **kwargs)

        with patch("core.infer.adapters.person.runtime_performance", metrics), patch(
            "core.infer.adapters.person.cv2.resize", side_effect=resize
        ), patch.object(PersonMask, "count", count):
            result = adapter.unpack_one(outputs, 0, request)

        self.assertEqual(result[0]["maskPixelCount"], 16)
        self.assertEqual(result[0]["mask"].data.shape, (4, 4))
        self.assertEqual(metrics.snapshot()["operations"]["person_seg_unpack"], {
            "count": 1, "errors": 0, "total_ms": 20.0,
            "avg_ms": 20.0, "max_ms": 20.0,
        })

    def test_unpack_failure_keeps_model_prefix_and_is_counted(self):
        metrics = RuntimePerformance()
        adapter = PersonSegmentationAdapter()
        request = adapter.normalize(np.zeros((4, 4, 3), dtype=np.uint8))
        with patch("core.infer.adapters.person.runtime_performance", metrics):
            with self.assertRaisesRegex(InferenceResponseError, "person_seg: "):
                adapter.unpack_one({}, 0, request)
        self.assertEqual(metrics.snapshot()["operations"]["person_seg_unpack"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
