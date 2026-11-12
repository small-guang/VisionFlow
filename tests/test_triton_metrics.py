import asyncio
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.infer.adapters.embeddings import ClipImageAdapter
from core.infer.batching import ModelBatchScheduler, PendingRequest
from core.infer.capabilities import ModelCapability
from core.infer.errors import InferenceUnavailable
from core.infer.metrics import active_inference_metrics, tensor_payload_bytes
from tests.test_triton_batching import (
    ImmediateExecutor, RecordingLogger, RejectingExecutor, wait_until,
)
from core.infer.executor import FairInferenceExecutor
from tests.test_triton_transport import FakeInferResult, FakeTritonClient, make_transport


class PayloadAccountingTests(unittest.TestCase):
    def test_numeric_slices_count_logical_payload_without_the_backing_array(self):
        values = {
            "pixels": np.zeros((2, 12), dtype=np.uint8)[:, ::2],
            "points": np.ones((1, 2), dtype=np.float32),
        }
        self.assertEqual(tensor_payload_bytes(values), 20)

    def test_bytes_count_length_headers_and_preserve_embedded_and_trailing_nuls(self):
        values = np.asarray([[b"\x00a\x00"], [b"\x00"]], dtype=object)
        self.assertEqual(tensor_payload_bytes({"encoded": values}), 12)
        self.assertIsNone(tensor_payload_bytes({"malformed": None}))


class BatchMetricsTests(unittest.IsolatedAsyncioTestCase):
    def make_batch(self, values, *, executor=None, client=None, adapter=None):
        adapter = adapter or ClipImageAdapter()
        client = client or FakeTritonClient()
        client.infer_result = FakeInferResult({
            "unnorm_image_features": np.ones((len(values), 512), dtype=np.float32),
        })
        transport = make_transport(client)
        self.addAsyncCleanup(transport.close)
        log = RecordingLogger()
        scheduler = ModelBatchScheduler(
            adapter=adapter,
            capability=ModelCapability(
                key="clip_image", model_name="clip_ensemble_trt",
                api_contract_version=1, server_max_batch=4, target_batch=len(values),
            ),
            transport=transport,
            executor=executor or ImmediateExecutor(),
            wait_ms=0,
            queue_timeout=100,
            logger=log,
        )
        self.addAsyncCleanup(scheduler.close)
        pending = [PendingRequest(
            f"metric-{index}", adapter.normalize(value),
            asyncio.get_running_loop().create_future(), time.perf_counter(),
        ) for index, value in enumerate(values)]
        return scheduler, transport, client, log, pending

    async def test_batch_record_separates_payload_packing_and_all_queue_stages(self):
        # Wrong BYTES accounting, missing queue stages or overlapping pack/RPC
        # timers must fail these independent, hand-derived expectations.
        clock = SimpleNamespace(now=10.0)
        fake_time = SimpleNamespace(perf_counter=lambda: clock.now)

        class TimedExecutor(ImmediateExecutor):
            async def submit(self, model, operation, *, deadline):
                clock.now += 3
                return await operation()

        scheduler, transport, client, log, pending = self.make_batch(
            [b"a\x00", b"bcde\x00"], executor=TimedExecutor(),
        )
        for item in pending:
            item.queued_at = 1.0
        original_pack = scheduler.adapter.pack
        original_build = transport._build_inputs
        original_acquire = transport._pool.acquire

        def pack(requests):
            clock.now += 2
            return original_pack(requests)

        def build(contract, inputs, **kwargs):
            clock.now += 4
            return original_build(contract, inputs, **kwargs)

        async def acquire():
            clock.now += 5
            return await original_acquire()

        async def infer(**kwargs):
            clock.now += 7
            return client.infer_result

        with patch("core.infer.batching.time", fake_time), patch(
            "core.infer.transport.time", fake_time
        ), patch.object(scheduler.adapter, "pack", pack), patch.object(
            transport, "_build_inputs", build
        ), patch.object(transport._pool, "acquire", acquire), patch.object(
            client, "infer", infer
        ):
            await scheduler._dispatch(pending)

        self.assertEqual(len(log.records), 1)
        fields, message, args, _ = log.records[0]
        expected = {
            "model": "clip_image", "batch_size": 2,
            "request_bytes": 15, "response_bytes": 4096,
            "pack_ms": 2000.0, "serialize_ms": 4000.0,
            "batch_wait_ms": 9000.0, "executor_queue_wait_ms": 3000.0,
            "client_wait_ms": 5000.0, "queue_total_ms": 17000.0,
            "rpc_ms": 7000.0, "rpc_attempts": 1,
        }
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(fields.get(name), value)
        # The application's default file sink prints only the message.
        rendered = message.format(*args)
        for name in ("model", "batch_size", "request_bytes", "response_bytes",
                     "pack_ms", "batch_wait_ms", "executor_queue_wait_ms"):
            self.assertIn(f"{name}=", rendered)
        for item in pending:
            self.assertEqual(item.future.result().vector.shape, (512,))
        self.assertIsNone(active_inference_metrics.get())

    async def test_pack_failure_reports_duration_without_claiming_a_transfer(self):
        scheduler, _, client, log, pending = self.make_batch([b"payload"])
        with patch.object(scheduler.adapter, "pack", side_effect=ValueError("bad pack")):
            await scheduler._dispatch(pending)
        self.assertIsInstance(pending[0].future.exception(), ValueError)
        fields = log.records[0][0]
        self.assertIn("pack_ms", fields)
        self.assertGreaterEqual(fields["pack_ms"], 0.0)
        self.assertIsNone(fields.get("request_bytes"))
        self.assertIsNone(fields.get("response_bytes"))
        self.assertEqual(fields.get("rpc_attempts"), 0)
        self.assertFalse(any(name == "infer" for name, _ in client.calls))

    async def test_rejected_batch_has_no_packing_or_transfer_metrics(self):
        scheduler, _, _, log, pending = self.make_batch(
            [b"payload"], executor=RejectingExecutor(InferenceUnavailable("full")),
        )
        await scheduler._dispatch(pending)
        self.assertIsInstance(pending[0].future.exception(), InferenceUnavailable)
        fields = log.records[0][0]
        self.assertIn("pack_ms", fields)
        self.assertIsNone(fields["pack_ms"])
        self.assertIsNone(fields["request_bytes"])
        self.assertIsNone(fields["response_bytes"])
        self.assertEqual(fields["rpc_attempts"], 0)

    async def test_rpc_failure_keeps_request_size_and_unknown_response(self):
        scheduler, _, client, log, pending = self.make_batch([b"payload"])
        client.infer_effects = [InferenceUnavailable("offline")]
        await scheduler._dispatch(pending)
        self.assertIsNotNone(pending[0].future.exception())
        fields = log.records[0][0]
        self.assertEqual(fields.get("request_bytes"), 11)
        self.assertIsNone(fields.get("response_bytes"))
        self.assertEqual(fields.get("rpc_attempts"), 1)

    async def test_cancelled_rpc_logs_once_and_releases_its_metrics_context(self):
        scheduler, _, client, log, pending = self.make_batch([b"cancel"])
        started = asyncio.Event()

        async def infer(**kwargs):
            started.set()
            await asyncio.Event().wait()

        with patch.object(client, "infer", infer):
            task = asyncio.create_task(scheduler._dispatch(pending))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(pending[0].future.cancelled())
        self.assertEqual(len(log.records), 1)
        self.assertEqual(log.records[0][0].get("rpc_attempts"), 1)
        self.assertIsNone(log.records[0][0].get("response_bytes"))

        next_scheduler, _, _, next_log, next_pending = self.make_batch([b"next"])
        await next_scheduler._dispatch(next_pending)
        self.assertEqual(next_log.records[0][0].get("request_bytes"), 8)
        self.assertEqual(next_log.records[0][0].get("rpc_attempts"), 1)
        self.assertEqual(next_pending[0].future.result().vector.shape, (512,))

    async def test_cancellation_during_pool_acquisition_accounts_for_wait(self):
        scheduler, transport, _, log, pending = self.make_batch([b"wait"])
        clock = SimpleNamespace(now=10.0)
        fake_time = SimpleNamespace(perf_counter=lambda: clock.now)

        async def acquire():
            clock.now += 5
            raise asyncio.CancelledError()

        with patch("core.infer.batching.time", fake_time), patch(
            "core.infer.transport.time", fake_time
        ), patch.object(transport._pool, "acquire", acquire):
            with self.assertRaises(asyncio.CancelledError):
                await scheduler._dispatch(pending)
        self.assertEqual(log.records[0][0]["client_wait_ms"], 5000.0)
        self.assertEqual(log.records[0][0]["rpc_attempts"], 0)
        self.assertIsNone(active_inference_metrics.get())

    async def test_retry_counts_attempts_without_duplicating_logical_payload_bytes(self):
        scheduler, transport, client, log, pending = self.make_batch([b"retry"])
        transport._transient_retries = 1
        from tritonclient.utils import InferenceServerException
        client.infer_effects = [
            InferenceServerException("retry", status="StatusCode.UNAVAILABLE"),
            client.infer_result,
        ]
        await scheduler._dispatch(pending)
        self.assertEqual(pending[0].future.result().vector.shape, (512,))
        fields = log.records[0][0]
        self.assertEqual(fields["request_bytes"], 9)
        self.assertEqual(fields["response_bytes"], 2048)
        self.assertEqual(fields["rpc_attempts"], 2)

    async def test_cancelled_waiter_records_terminal_bytes_after_real_executor_finishes(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=2)
        await executor.start()
        self.addAsyncCleanup(executor.close)
        scheduler, _, client, log, _ = self.make_batch([b"cancel"], executor=executor)
        started = asyncio.Event()
        release = asyncio.Event()

        async def infer(**kwargs):
            started.set()
            await release.wait()
            return client.infer_result

        try:
            with patch.object(client, "infer", infer):
                submitted = asyncio.create_task(
                    scheduler.submit(scheduler.adapter.normalize(b"cancel"))
                )
                await asyncio.wait_for(started.wait(), 1)
                submitted.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await submitted
                await wait_until(lambda: not scheduler._dispatch_pending)
                self.assertEqual(log.records, [])
                release.set()
                await wait_until(lambda: bool(log.records))
        finally:
            release.set()

        self.assertEqual(len(log.records), 1)
        fields = log.records[0][0]
        self.assertEqual(fields["response_bytes"], 2048)
        self.assertEqual(fields["rpc_attempts"], 1)
        self.assertTrue(fields["caller_cancelled"])
        self.assertGreater(fields["infer_ms"], 0)
        self.assertGreater(fields["rpc_ms"], 0)

    async def test_rpc_failure_racing_with_caller_cancellation_retains_remote_error(self):
        from tritonclient.utils import InferenceServerException

        executor = FairInferenceExecutor(concurrency=1, queue_capacity=2)
        await executor.start()
        self.addAsyncCleanup(executor.close)
        scheduler, _, client, log, _ = self.make_batch([b"cancel"], executor=executor)
        started = asyncio.Event()
        release = asyncio.Event()

        async def infer(**kwargs):
            started.set()
            await release.wait()
            raise InferenceServerException("offline", status="StatusCode.UNAVAILABLE")

        try:
            with patch.object(client, "infer", infer):
                submitted = asyncio.create_task(
                    scheduler.submit(scheduler.adapter.normalize(b"cancel"))
                )
                await asyncio.wait_for(started.wait(), 1)
                submitted.cancel()
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await submitted
                await wait_until(lambda: bool(log.records))
        finally:
            release.set()

        self.assertEqual(len(log.records), 1)
        fields = log.records[0][0]
        self.assertEqual(fields["error_class"], "InferenceUnavailable")
        self.assertEqual(fields["request_bytes"], 10)
        self.assertIsNone(fields["response_bytes"])
        self.assertEqual(fields["rpc_attempts"], 1)
        self.assertTrue(fields["caller_cancelled"])
        self.assertGreater(fields["rpc_ms"], 0)


if __name__ == "__main__":
    unittest.main()
