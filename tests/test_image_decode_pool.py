import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np

from core.execution import AsyncWorkExecutor
from core.performance import RuntimePerformance
from utils.storage.image_util import get_image_by_oss_name


class ImageDecodePoolTests(unittest.IsolatedAsyncioTestCase):
    def make_executor(self, workers=1):
        executor = AsyncWorkExecutor(workers, name="image_cpu")
        self.addAsyncCleanup(executor.close)
        return executor

    async def test_slow_decode_keeps_loop_responsive_and_preserves_result(self):
        entered, release = threading.Event(), threading.Event()
        image = np.ones((3, 4, 3), dtype=np.uint8)
        worker_ids = []

        def decode(data):
            worker_ids.append(threading.get_ident())
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release decoder")
            return image

        storage = AsyncMock()
        storage.get_object.return_value = b"image"
        executor = self.make_executor()
        with patch("utils.storage.image_util._decode_image", side_effect=decode):
            task = asyncio.create_task(get_image_by_oss_name(
                "key", storage, image_executor=executor,
            ))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                await asyncio.wait_for(asyncio.sleep(0.005), timeout=0.2)
                self.assertFalse(task.done())
                release.set()
                decoded, encoded = await asyncio.wait_for(task, 1)
                self.assertIs(decoded, image)
                self.assertEqual(encoded, b"image")
                self.assertNotEqual(worker_ids, [threading.get_ident()])
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_real_image_roundtrip_and_original_exception(self):
        image = np.full((7, 9, 3), 42, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        storage = AsyncMock()
        storage.get_object.return_value = encoded.tobytes()
        executor = self.make_executor(2)
        metrics = RuntimePerformance()
        with patch("utils.storage.image_util.runtime_performance", metrics):
            decoded, original = await get_image_by_oss_name(
                "key", storage, image_executor=executor,
            )
            np.testing.assert_array_equal(decoded, image)
            self.assertEqual(original, encoded.tobytes())
            storage.get_object.return_value = b"invalid-image"
            with self.assertRaisesRegex(ValueError, "图片解码失败"):
                await get_image_by_oss_name("key", storage, image_executor=executor)
        self.assertEqual(metrics.snapshot()["operations"]["image_decode"]["errors"], 1)

    async def test_cancellation_retains_submission_bound_until_work_finishes(self):
        entered, release = threading.Event(), threading.Event()
        executor = self.make_executor()
        calls = []

        def decode(data):
            calls.append(data)
            entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release decoder")
            return np.zeros((1, 1, 3), dtype=np.uint8)

        storage = AsyncMock()
        storage.get_object.return_value = b"image"
        second = None
        with patch("utils.storage.image_util._decode_image", side_effect=decode):
            first = asyncio.create_task(get_image_by_oss_name(
                "first", storage, image_executor=executor,
            ))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(get_image_by_oss_name(
                    "second", storage, image_executor=executor,
                ))
                await asyncio.sleep(0.02)
                self.assertEqual(calls, [b"image"])
                release.set()
                await asyncio.wait_for(second, 1)
                self.assertEqual(calls, [b"image", b"image"])
            finally:
                release.set()
                await asyncio.gather(*([first, second] if second else [first]), return_exceptions=True)

    async def test_empty_storage_result_does_not_enter_decode_pool(self):
        storage, executor = AsyncMock(), AsyncMock()
        storage.get_object.return_value = b""
        with self.assertRaisesRegex(ValueError, "从对象存储获取图片数据为空"):
            await get_image_by_oss_name("key", storage, image_executor=executor)
        executor.run.assert_not_awaited()
        executor.run_timed.assert_not_awaited()

    async def test_missing_executor_fails_before_storage_access(self):
        storage = AsyncMock()
        with self.assertRaisesRegex(ValueError, "image_executor is required"):
            await get_image_by_oss_name("key", storage, image_executor=None)
        storage.get_object.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
