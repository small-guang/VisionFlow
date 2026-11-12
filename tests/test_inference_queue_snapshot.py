import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from core.infer.batching import ModelBatchScheduler, NormalizedRequest
from core.infer.capabilities import ModelCapability
from core.infer.contracts import MODEL_CONTRACTS
from core.infer.errors import InferenceQueueTimeout, InferenceUnavailable
from core.infer.executor import FairInferenceExecutor
from core.infer.triton_client import UnifiedTritonClient


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1)


class QuietLogger:
    def bind(self, **fields):
        return self

    def info(self, *args, **kwargs):
        pass


class SampleAdapter:
    contract = MODEL_CONTRACTS["clip_image"]

    def pack(self, requests):
        return {"VALUE": np.stack([item.inputs["VALUE"] for item in requests])}

    def validate_outputs(self, outputs, batch_size):
        if outputs["VALUE"].shape[0] != batch_size:
            raise AssertionError("unexpected batch size")

    def unpack_one(self, outputs, index, request):
        return int(outputs["VALUE"][index, 0])


class GatedTransport:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def infer(self, contract, inputs, *, request_id):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return inputs


def capability(key="clip_image", target=2):
    contract = MODEL_CONTRACTS[key]
    return ModelCapability(
        key=key,
        model_name=contract.model_name,
        api_contract_version=contract.api_contract_version,
        server_max_batch=4,
        target_batch=target,
    )


def sample(value):
    return NormalizedRequest(
        inputs={"VALUE": np.array([value], dtype=np.int64)},
        compatibility_key=(1,),
        context="private-context-not-for-logs",
    )


class ExecutorQueueSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_tracks_models_capacity_cancellation_and_completion(self):
        executor = FairInferenceExecutor(
            concurrency=1, queue_capacity=2, logger=QuietLogger()
        )
        initial = executor.queue_snapshot()
        self.assertEqual(initial["state"], "new")
        self.assertEqual(initial["models"], {})
        self.assertEqual(initial["queue_capacity_batches"], 2)
        self.assertEqual(initial["concurrency_batches"], 1)
        await executor.start()
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_operation():
            started.set()
            await release.wait()
            return "finished"

        first = asyncio.create_task(
            executor.submit("a", blocked_operation, deadline=time.perf_counter() + 5)
        )
        await started.wait()
        cancelled = asyncio.create_task(
            executor.submit("b", blocked_operation, deadline=time.perf_counter() + 5)
        )
        last = asyncio.create_task(
            executor.submit("a", blocked_operation, deadline=time.perf_counter() + 5)
        )
        try:
            await wait_until(lambda: executor.queue_snapshot()["queued_batches"] == 2)
            snapshot = executor.queue_snapshot()
            self.assertEqual(snapshot["active_batches"], 1)
            self.assertEqual(snapshot["models"], {
                "a": {"queued_batches": 1, "active_batches": 1},
                "b": {"queued_batches": 1, "active_batches": 0},
            })
            snapshot["models"]["a"]["queued_batches"] = 900
            self.assertEqual(executor.queue_snapshot()["models"]["a"]["queued_batches"], 1)
            with self.assertRaises(InferenceUnavailable):
                await executor.submit("rejected", blocked_operation, deadline=time.perf_counter() + 5)
            self.assertNotIn("rejected", executor.queue_snapshot()["models"])
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            self.assertEqual(executor.queue_snapshot()["queued_batches"], 1)
            self.assertNotIn("b", executor.queue_snapshot()["models"])
        finally:
            release.set()
            await asyncio.gather(first, cancelled, last, return_exceptions=True)
            await executor.close()
        final = executor.queue_snapshot()
        self.assertEqual(final["state"], "closed")
        self.assertEqual(final["active_batches"], 0)
        self.assertEqual(final["queued_batches"], 0)
        self.assertEqual(final["models"], {})

    async def test_caller_cancellation_does_not_hide_active_transport_wait(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=2, logger=QuietLogger())
        await executor.start()
        started = asyncio.Event()
        release = asyncio.Event()

        async def waiting_for_client_or_shm():
            started.set()
            await release.wait()

        submission = asyncio.create_task(executor.submit(
            "clip_image", waiting_for_client_or_shm, deadline=time.perf_counter() + 5
        ))
        try:
            await started.wait()
            submission.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await submission
            self.assertEqual(executor.queue_snapshot()["active_batches"], 1)
            self.assertEqual(executor.queue_snapshot()["models"]["clip_image"]["active_batches"], 1)
        finally:
            release.set()
            await executor.close()
        self.assertEqual(executor.queue_snapshot()["active_batches"], 0)

    async def test_timeout_and_operation_error_leave_no_phantom_backlog(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=2, logger=QuietLogger())
        await executor.start()
        release = asyncio.Event()
        first = asyncio.create_task(executor.submit(
            "busy", release.wait, deadline=time.perf_counter() + 5
        ))
        await wait_until(lambda: executor.queue_snapshot()["active_batches"] == 1)

        async def failing_operation():
            raise ValueError("synthetic failure")

        try:
            with self.assertRaises(InferenceQueueTimeout):
                await executor.submit("expired", failing_operation, deadline=time.perf_counter() + 0.01)
            self.assertEqual(executor.queue_snapshot()["queued_batches"], 0)
            self.assertNotIn("expired", executor.queue_snapshot()["models"])
            release.set()
            await first
            with self.assertRaises(ValueError):
                await executor.submit("failed", failing_operation, deadline=time.perf_counter() + 5)
            self.assertEqual(executor.queue_snapshot()["active_batches"], 0)
            self.assertEqual(executor.queue_snapshot()["models"], {})
        finally:
            release.set()
            await first
            await executor.close()


class SchedulerQueueSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_samples_leave_batch_wait_when_dispatched_as_one_batch(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=2, logger=QuietLogger())
        transport = GatedTransport()
        scheduler = ModelBatchScheduler(
            adapter=SampleAdapter(), capability=capability(), transport=transport,
            executor=executor, wait_ms=10000, queue_timeout=20, logger=QuietLogger(),
        )
        await executor.start()
        first = asyncio.create_task(scheduler.submit(sample(1)))
        second = None
        try:
            await wait_until(lambda: scheduler.queue_snapshot()["batch_waiting_samples"] == 1)
            before = scheduler.queue_snapshot()
            self.assertEqual(before["batch_target_samples"], 2)
            self.assertEqual(before["server_max_batch_samples"], 4)
            self.assertEqual(executor.queue_snapshot()["active_batches"], 0)
            for _ in range(20):
                self.assertEqual(scheduler.queue_snapshot(), before)
            self.assertEqual(transport.calls, 0)
            second = asyncio.create_task(scheduler.submit(sample(2)))
            await transport.started.wait()
            self.assertEqual(scheduler.queue_snapshot()["batch_waiting_samples"], 0)
            self.assertEqual(executor.queue_snapshot()["active_batches"], 1)
            self.assertNotIn("private-context", json.dumps(scheduler.queue_snapshot()))
        finally:
            transport.release.set()
            results = await asyncio.gather(first, *([second] if second is not None else []))
            await scheduler.close()
            await executor.close()
        self.assertEqual(results, [1, 2])
        self.assertTrue(scheduler.queue_snapshot()["closed"])

    async def test_cancelled_samples_are_removed_from_batch_wait(self):
        scheduler = ModelBatchScheduler(
            adapter=SampleAdapter(), capability=capability(), transport=object(),
            executor=object(), wait_ms=10000, queue_timeout=20, logger=QuietLogger(),
        )
        pending = asyncio.create_task(scheduler.submit(sample(1)))
        await wait_until(lambda: scheduler.queue_snapshot()["batch_waiting_samples"] == 1)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(scheduler.queue_snapshot()["batch_waiting_samples"], 0)
        await scheduler.close()


class ClientQueueSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_facade_combines_local_snapshots_without_network_access(self):
        settings = SimpleNamespace(
            TRITON_BATCH_TARGETS={}, TRITON_BATCH_TARGET_DEFAULT=2,
            TRITON_BATCH_WAIT_MS={}, TRITON_BATCH_WAIT_MS_DEFAULT=10000,
            TRITON_QUEUE_TIMEOUT_SECONDS=20,
        )
        transport = SimpleNamespace(
            start=AsyncMock(), is_server_ready=AsyncMock(return_value=True),
            close=AsyncMock(), infer=AsyncMock(side_effect=AssertionError("no inference expected")),
        )
        executor = FairInferenceExecutor(concurrency=2, queue_capacity=3, logger=QuietLogger())
        client = UnifiedTritonClient(settings=settings, transport=transport, executor=executor)
        self.assertEqual(client.queue_snapshot()["state"], "new")
        with patch("core.infer.triton_client.discover_all_capabilities", AsyncMock(
            return_value={key: capability(key) for key in MODEL_CONTRACTS}
        )):
            await client.initialize()
        pending = asyncio.create_task(client.extract_image_features(b"private-image-payload"))
        try:
            await wait_until(lambda: client.queue_snapshot()["batch_waiting_samples"] == 1)
            snapshot = client.queue_snapshot()
            self.assertEqual(snapshot["models"]["clip_image"]["batch_waiting_samples"], 1)
            self.assertEqual(snapshot["models"]["clip_image"]["executor_queued_batches"], 0)
            self.assertEqual(snapshot["executor"]["queue_capacity_batches"], 3)
            self.assertEqual(snapshot["executor"]["concurrency_batches"], 2)
            self.assertEqual(len(snapshot["models"]), len(MODEL_CONTRACTS))
            self.assertNotIn("private-image", json.dumps(snapshot))
            for _ in range(20):
                self.assertEqual(client.queue_snapshot(), snapshot)
            transport.infer.assert_not_called()
            transport.is_server_ready.assert_awaited_once()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await client.close()
        self.assertEqual(client.queue_snapshot()["batch_waiting_samples"], 0)
        self.assertEqual(client.queue_snapshot()["state"], "closed")

    async def test_custom_executor_without_observability_is_unavailable_not_zero(self):
        client = UnifiedTritonClient(settings=object(), transport=object(), executor=object())
        self.assertIsNone(client.queue_snapshot()["executor"])


if __name__ == "__main__":
    unittest.main()
