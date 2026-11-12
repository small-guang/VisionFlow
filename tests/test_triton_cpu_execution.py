"""推理大数组组包借用图片 CPU 池，轻量路径与结果分发留在事件循环。"""

import asyncio
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from core.execution import AsyncWorkExecutor
from core.infer.adapters.embeddings import ClipImageAdapter
from core.infer.adapters.face import FaceDetectionAdapter
from core.infer.batching import ModelBatchScheduler, PendingRequest
from core.infer.capabilities import ModelCapability
from core.infer.executor import FairInferenceExecutor
from core.infer.metrics import InferenceMetrics
from core.infer.triton_client import UnifiedTritonClient
from tests.test_triton_batching import ImmediateExecutor, RecordingLogger
from tests.test_triton_pose_client import FakeTransport, facade_settings, target_one_capabilities


class OutputTransport:
    def __init__(self):
        self.inputs = []

    async def infer(self, contract, inputs, *, request_id):
        self.inputs.append(inputs)
        batch = next(iter(inputs.values())).shape[0]
        if contract.key == "clip_image":
            return {"unnorm_image_features": np.ones((batch, 512), dtype=np.float32)}
        return {
            "DETECTIONS": np.zeros((batch, 100, 6), dtype=np.float32),
            "NUM_DETECTIONS": np.zeros((batch, 1), dtype=np.int32),
        }


class TritonCpuExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cpu = AsyncWorkExecutor(1, name="test-inference-cpu")
        self.addAsyncCleanup(self.cpu.close)

    def batch(self, adapter, values, *, cpu=None, executor=None):
        transport, logger = OutputTransport(), RecordingLogger()
        scheduler = ModelBatchScheduler(
            adapter=adapter,
            capability=ModelCapability(
                key=adapter.contract.key, model_name=adapter.contract.model_name,
                api_contract_version=adapter.contract.api_contract_version,
                server_max_batch=4, target_batch=len(values),
            ),
            transport=transport, executor=executor or ImmediateExecutor(),
            cpu_executor=self.cpu if cpu is None else cpu,
            wait_ms=0, queue_timeout=5, logger=logger,
        )
        pending = [PendingRequest(
            str(index), adapter.normalize(value), asyncio.get_running_loop().create_future(),
            time.perf_counter(),
        ) for index, value in enumerate(values)]
        self.addAsyncCleanup(scheduler.close)
        return scheduler, pending, transport, logger

    async def test_contiguous_single_image_and_encoded_bytes_avoid_thread_roundtrip(self):
        class UnusedCpu:
            async def run(self, *args, **kwargs):
                raise AssertionError("cheap pack was unnecessarily offloaded")

        image = np.zeros((768, 768, 3), dtype=np.uint8)
        cases = (
            (FaceDetectionAdapter(), [image]),
            (ClipImageAdapter(), [b"x" * (1024 * 1024), b"y" * (1024 * 1024)]),
        )
        for adapter, values in cases:
            scheduler, pending, transport, logger = self.batch(adapter, values, cpu=UnusedCpu())
            await scheduler._dispatch(pending)
            await asyncio.gather(*(item.future for item in pending))
            self.assertEqual(logger.records[0][0]["pack_queue_wait_ms"], 0.0)
            if adapter.contract.key == "face_det":
                self.assertTrue(np.shares_memory(transport.inputs[0]["RAW_IMAGE"], image))

    async def test_large_strided_copy_and_batch_stack_preserve_tensor_contract(self):
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        image = np.arange(1024 * 1024 * 3, dtype=np.uint8).reshape(1024, 1024, 3)
        cases = ([image[:768, :768]], [image[:512, :512].copy()] * 2)
        for values in cases:
            entered, release = asyncio.Event(), threading.Event()
            adapter = FaceDetectionAdapter()
            original_pack, original_validate, original_unpack = (
                adapter.pack, adapter.validate_outputs, adapter.unpack_one,
            )
            expected = original_pack([adapter.normalize(value) for value in values])
            observed = []

            def pack(requests):
                observed.append(("pack", threading.get_ident()))
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(2):
                    raise TimeoutError("test did not release CPU pack")
                return original_pack(requests)

            def validate(*args):
                observed.append(("validate", threading.get_ident()))
                return original_validate(*args)

            def unpack(*args):
                observed.append(("unpack", threading.get_ident()))
                return original_unpack(*args)

            adapter.pack, adapter.validate_outputs, adapter.unpack_one = pack, validate, unpack
            scheduler, pending, transport, logger = self.batch(adapter, values)
            task = asyncio.create_task(scheduler._dispatch(pending))
            try:
                # 组包故意阻塞时，loop 仍必须能观察线程发来的事件。
                await asyncio.wait_for(entered.wait(), 1)
                self.assertFalse(task.done())
                release.set()
                await task
                self.assertEqual(await asyncio.gather(*(item.future for item in pending)), [[]] * len(values))
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
            np.testing.assert_array_equal(transport.inputs[0]["RAW_IMAGE"], expected["RAW_IMAGE"])
            self.assertTrue(transport.inputs[0]["RAW_IMAGE"].flags.c_contiguous)
            self.assertNotEqual(observed[0][1], loop_thread)
            self.assertTrue(all(thread == loop_thread for name, thread in observed if name != "pack"))
            fields = logger.records[0][0]
            self.assertIsNotNone(fields["pack_ms"])
            self.assertGreaterEqual(fields["pack_queue_wait_ms"], 0.0)

    async def test_pack_failure_is_original_and_prevents_rpc(self):
        adapter = FaceDetectionAdapter()
        failure = ValueError("invalid mutated input")

        def fail(requests):
            raise failure

        adapter.pack = fail
        image = np.zeros((800, 900, 3), dtype=np.uint8)[:, :800]
        scheduler, pending, transport, logger = self.batch(adapter, [image])
        await scheduler._dispatch(pending)
        with self.assertRaises(ValueError) as raised:
            await pending[0].future
        self.assertIs(raised.exception, failure)
        self.assertEqual(transport.inputs, [])
        self.assertEqual(logger.records[0][0]["error_class"], "ValueError")
        self.assertIsNotNone(logger.records[0][0]["pack_ms"])

    async def test_cancelled_caller_keeps_real_pack_owned_until_executor_drains(self):
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        adapter = FaceDetectionAdapter()
        original_pack = adapter.pack

        def pack(requests):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(2):
                raise TimeoutError("test did not release CPU pack")
            return original_pack(requests)

        adapter.pack = pack
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=2)
        await executor.start()
        self.addAsyncCleanup(executor.close)
        image = np.zeros((800, 900, 3), dtype=np.uint8)[:, :800]
        scheduler, pending, transport, logger = self.batch(adapter, [image], executor=executor)
        task = asyncio.create_task(scheduler._dispatch(pending))
        closer = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(pending[0].future.cancelled())
            self.assertEqual(self.cpu.queue_snapshot()["inflight_jobs"], 1)
            self.assertEqual(executor.queue_snapshot()["active_batches"], 1)
            closer = asyncio.create_task(executor.close())
            await asyncio.sleep(0)
            self.assertFalse(closer.done())
            release.set()
            await closer
            self.assertEqual(self.cpu.queue_snapshot()["inflight_jobs"], 0)
            self.assertEqual(len(logger.records), 1)
            self.assertTrue(logger.records[0][0]["caller_cancelled"])
            self.assertEqual(logger.records[0][0]["request_bytes"], image.nbytes)
        finally:
            release.set()
            await asyncio.gather(*(item for item in (task, closer) if item), return_exceptions=True)

    async def test_facade_injects_borrowed_cpu_resource_without_closing_it(self):
        facade = UnifiedTritonClient(
            settings=facade_settings(), transport=FakeTransport(), cpu_executor=self.cpu,
        )
        with patch("core.infer.triton_client.discover_all_capabilities", return_value=target_one_capabilities()):
            await facade.initialize()
        self.assertTrue(all(scheduler.cpu_executor is self.cpu for scheduler in facade.schedulers.values()))
        await facade.close()
        self.assertEqual(await self.cpu.run(lambda: 42), 42)

    def test_pack_queue_is_separate_from_cpu_duration_and_added_to_total_queue(self):
        metrics = InferenceMetrics(pack_ms=7.0, pack_queue_wait_ms=3.0, executor_queue_wait_ms=2.0, client_wait_ms=5.0)
        fields = metrics.log_fields(11.0)
        self.assertEqual(fields["pack_ms"], 7.0)
        self.assertEqual(fields["pack_queue_wait_ms"], 3.0)
        self.assertEqual(fields["queue_total_ms"], 21.0)


if __name__ == "__main__":
    unittest.main()
