"""分类计时不能创建第二套存储准入，异常仍交给原SDK适配器。"""

import asyncio
import threading
import unittest

from core.execution import AsyncWorkExecutor
from core.performance import RuntimePerformance
from utils.storage.execution import run_storage_operation


class StorageExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_download_and_upload_share_capacity_and_add_direction_timings(self):
        metrics = RuntimePerformance()
        executor = AsyncWorkExecutor(1, name="storage_io", metrics=metrics)
        entered, release = threading.Event(), threading.Event()
        calls = []

        def upload():
            entered.set()
            release.wait(3)
            calls.append("upload")
            return "crop.jpg"

        def download():
            calls.append("download")
            return b"image"

        first = asyncio.create_task(run_storage_operation(executor, "upload", upload))
        second = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            second = asyncio.create_task(run_storage_operation(executor, "download", download))
            await asyncio.sleep(0)
            self.assertFalse(second.done())
            self.assertEqual(calls, [])
            release.set()
            self.assertEqual(await first, "crop.jpg")
            self.assertEqual(await second, b"image")
            self.assertEqual(calls, ["upload", "download"])
            operations = metrics.snapshot()["operations"]
            for suffix in ("run", "queue_wait"):
                self.assertEqual(operations[f"storage_io_{suffix}"]["count"], 2)
                for direction in ("download", "upload"):
                    self.assertEqual(operations[f"storage_{direction}_{suffix}"]["count"], 1)
        finally:
            release.set()
            await asyncio.gather(*(task for task in (first, second) if task), return_exceptions=True)
            await executor.close()

    async def test_sdk_exception_is_preserved_and_counted_once_in_each_scope(self):
        metrics = RuntimePerformance()
        executor = AsyncWorkExecutor(1, name="storage_io", metrics=metrics)
        failure = OSError("storage unavailable")

        def fail():
            raise failure

        try:
            with self.assertRaises(OSError) as raised:
                await run_storage_operation(executor, "upload", fail)
            self.assertIs(raised.exception, failure)
            operations = metrics.snapshot()["operations"]
            self.assertEqual(operations["storage_io_run"]["errors"], 1)
            self.assertEqual(operations["storage_upload_run"]["errors"], 1)
            self.assertEqual(operations["storage_download_run"]["count"], 0)
        finally:
            await executor.close()


if __name__ == "__main__":
    unittest.main()
