import asyncio
import logging
import math
import unittest
from unittest.mock import patch

import grpc
import numpy as np
from tritonclient.utils import InferenceServerException

from core.infer.contracts import MODEL_CONTRACTS
from core.infer.errors import (
    InferenceInputError,
    InferenceQueueTimeout,
    InferenceResponseError,
    InferenceServerTimeout,
    InferenceUnavailable,
    TritonInferenceError,
)
from core.infer.transport import AsyncClientPool, TritonTransport


class FakePoolClient:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class AsyncClientPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_size_one_waiter_receives_released_client_and_close_closes_it(self):
        client = FakePoolClient()
        pool = AsyncClientPool(lambda: client, size=1, acquire_timeout=0.1)
        await pool.start()

        first = await pool.acquire()
        waiter = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())

        await pool.release(first)
        second = await waiter
        self.assertIs(second, client)
        await pool.release(second)
        await pool.close()

        self.assertEqual(client.close_calls, 1)

    async def test_lease_returns_client_when_body_raises(self):
        client = FakePoolClient()
        pool = AsyncClientPool(lambda: client, size=1, acquire_timeout=0.1)

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            async with pool.lease() as leased:
                self.assertIs(leased, client)
                raise RuntimeError("body failed")

        leased_again = await pool.acquire()
        self.assertIs(leased_again, client)
        await pool.release(leased_again)
        await pool.close()

    async def test_start_and_close_are_idempotent(self):
        created = []

        def factory():
            created.append(FakePoolClient())
            return created[-1]

        pool = AsyncClientPool(factory, size=2, acquire_timeout=0.1)
        await pool.start()
        await pool.start()
        await pool.close()
        await pool.close()

        self.assertEqual(len(created), 2)
        self.assertEqual([client.close_calls for client in created], [1, 1])

    async def test_closed_pool_cannot_be_started_or_acquired_again(self):
        pool = AsyncClientPool(FakePoolClient, size=1, acquire_timeout=0.1)
        await pool.start()
        await pool.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            await pool.start()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await pool.acquire()

    async def test_invalid_pool_parameters_fail_early(self):
        for size in (True, 0, -1, 1.5, "1", None):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    AsyncClientPool(
                        FakePoolClient,
                        size=size,
                        acquire_timeout=1,
                    )
        for timeout in (True, 0, -1, math.nan, math.inf, "1", None):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    AsyncClientPool(
                        FakePoolClient,
                        size=1,
                        acquire_timeout=timeout,
                    )

    async def test_acquire_timeout_is_a_queue_timeout(self):
        pool = AsyncClientPool(FakePoolClient, size=1, acquire_timeout=0.01)
        await pool.start()
        checked_out = await pool.acquire()

        with self.assertRaises(InferenceQueueTimeout):
            await pool.acquire()

        await pool.release(checked_out)
        await pool.close()

    async def test_close_rejects_waiters_and_waits_for_checked_out_client(self):
        client = FakePoolClient()
        pool = AsyncClientPool(lambda: client, size=1, acquire_timeout=1)
        first = await pool.acquire()
        waiter = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())

        close_task = asyncio.create_task(pool.close())
        with self.assertRaisesRegex(RuntimeError, "closing|closed"):
            await asyncio.wait_for(waiter, timeout=0.1)
        self.assertFalse(close_task.done())
        self.assertEqual(client.close_calls, 0)

        await pool.release(first)
        await asyncio.wait_for(close_task, timeout=0.1)
        self.assertEqual(client.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "released|closed"):
            await pool.release(first)

    async def test_cancelling_close_caller_does_not_abort_graceful_shutdown(self):
        client = FakePoolClient()
        pool = AsyncClientPool(lambda: client, size=1, acquire_timeout=0.1)
        checked_out = await pool.acquire()

        close_caller = asyncio.create_task(pool.close())
        await asyncio.sleep(0)
        close_caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await close_caller
        self.assertEqual(client.close_calls, 0)

        await pool.release(checked_out)
        await asyncio.wait_for(pool.close(), timeout=0.1)
        self.assertEqual(client.close_calls, 1)

    async def test_partial_factory_failure_is_cleaned_up_and_start_can_retry(self):
        created = []
        calls = 0
        fail_second = True

        def factory():
            nonlocal calls
            calls += 1
            if fail_second and calls == 2:
                raise RuntimeError("factory failed")
            client = FakePoolClient()
            created.append(client)
            return client

        pool = AsyncClientPool(factory, size=2, acquire_timeout=0.1)
        with self.assertRaisesRegex(RuntimeError, "factory failed"):
            await pool.start()
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].close_calls, 1)

        fail_second = False
        await pool.start()
        first = await pool.acquire()
        second = await pool.acquire()
        self.assertIsNot(first, created[0])
        self.assertIsNot(second, created[0])
        await pool.release(first)
        await pool.release(second)
        await pool.close()
        self.assertEqual([client.close_calls for client in created], [1, 1, 1])

    async def test_close_attempts_every_client_and_does_not_reclose_after_error(self):
        class FailingCloseClient(FakePoolClient):
            async def close(self):
                self.close_calls += 1
                raise RuntimeError("close failed")

        clients = [FailingCloseClient(), FakePoolClient()]
        pool = AsyncClientPool(
            lambda: clients.pop(0), size=2, acquire_timeout=0.1
        )
        await pool.start()
        owned_clients = list(pool._clients)

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            await pool.close()
        self.assertEqual(
            [client.close_calls for client in owned_clients], [1, 1]
        )

        await pool.close()
        self.assertEqual(
            [client.close_calls for client in owned_clients], [1, 1]
        )


class FakeInferInput:
    instances = []

    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None
        self.__class__.instances.append(self)

    def set_data_from_numpy(self, data):
        self.data = data


class FakeRequestedOutput:
    instances = []

    def __init__(self, name):
        self.name = name
        self.__class__.instances.append(self)


class FakeInferResult:
    def __init__(self, outputs):
        self.outputs = outputs

    def as_numpy(self, name):
        return self.outputs.get(name)


class FakeTritonClient:
    def __init__(self):
        self.calls = []
        self.close_calls = 0
        self.infer_effects = []
        self.infer_result = FakeInferResult({})

    async def close(self):
        self.close_calls += 1

    async def is_server_ready(self, **kwargs):
        self.calls.append(("server_ready", kwargs))
        return True

    async def is_model_ready(self, **kwargs):
        self.calls.append(("model_ready", kwargs))
        return True

    async def get_model_config(self, **kwargs):
        self.calls.append(("config", kwargs))
        return {"config": {}}

    async def get_model_metadata(self, **kwargs):
        self.calls.append(("metadata", kwargs))
        return {"name": kwargs["model_name"]}

    async def infer(self, **kwargs):
        self.calls.append(("infer", kwargs))
        if self.infer_effects:
            effect = self.infer_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return effect
        return self.infer_result


class CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def make_transport(client, **overrides):
    options = {
        "url": "triton.example:8001",
        "pool_size": 1,
        "acquire_timeout": 0.1,
        "infer_timeout": 3.0,
        "config_timeout": 2.0,
        "transient_retries": 0,
        "client_factory": lambda url: client,
    }
    options.update(overrides)
    return TritonTransport(**options)


class TritonTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeInferInput.instances.clear()
        FakeRequestedOutput.instances.clear()

    async def test_default_factory_constructs_aio_client_with_url(self):
        client = FakeTritonClient()
        with patch(
            "core.infer.transport.grpcclient_aio.InferenceServerClient",
            return_value=client,
        ) as constructor:
            transport = TritonTransport(
                url="triton.example:8001",
                pool_size=1,
                acquire_timeout=0.1,
                infer_timeout=3,
                config_timeout=2,
                transient_retries=0,
            )
            await transport.start()
            await transport.close()

        constructor.assert_called_once_with(url="triton.example:8001")

    async def test_readiness_config_and_metadata_use_pool_and_config_timeout(self):
        client = FakeTritonClient()
        transport = make_transport(client)

        self.assertTrue(await transport.is_server_ready())
        self.assertTrue(await transport.is_model_ready("model_a"))
        self.assertEqual(await transport.get_model_config("model_a"), {"config": {}})
        self.assertEqual(
            await transport.get_model_metadata("model_a"), {"name": "model_a"}
        )

        self.assertEqual(
            client.calls,
            [
                ("server_ready", {"client_timeout": 2.0}),
                (
                    "model_ready",
                    {"model_name": "model_a", "client_timeout": 2.0},
                ),
                (
                    "config",
                    {
                        "model_name": "model_a",
                        "as_json": True,
                        "client_timeout": 2.0,
                    },
                ),
                (
                    "metadata",
                    {
                        "model_name": "model_a",
                        "as_json": True,
                        "client_timeout": 2.0,
                    },
                ),
            ],
        )

    async def test_infer_builds_all_declared_inputs_and_requested_outputs(self):
        contract = MODEL_CONTRACTS["face_det"]
        client = FakeTritonClient()
        detections = np.zeros((2, 100, 6), dtype=np.float32)
        counts = np.ones((2, 1), dtype=np.int32)
        client.infer_result = FakeInferResult(
            {"DETECTIONS": detections, "NUM_DETECTIONS": counts}
        )
        transport = make_transport(client)
        input_array = np.zeros((2, 8, 12, 3), dtype=np.uint8)

        with patch("core.infer.transport.grpcclient.InferInput", FakeInferInput), patch(
            "core.infer.transport.grpcclient.InferRequestedOutput",
            FakeRequestedOutput,
        ):
            outputs = await transport.infer(
                contract, {"RAW_IMAGE": input_array}, request_id="req-7"
            )

        self.assertIs(outputs["DETECTIONS"], detections)
        self.assertIs(outputs["NUM_DETECTIONS"], counts)
        self.assertEqual(len(FakeInferInput.instances), 1)
        infer_input = FakeInferInput.instances[0]
        self.assertEqual(infer_input.name, "RAW_IMAGE")
        self.assertEqual(infer_input.shape, input_array.shape)
        self.assertEqual(infer_input.datatype, "UINT8")
        self.assertIs(infer_input.data, input_array)
        self.assertEqual(
            [output.name for output in FakeRequestedOutput.instances],
            ["DETECTIONS", "NUM_DETECTIONS"],
        )
        infer_call = client.calls[-1]
        self.assertEqual(infer_call[0], "infer")
        self.assertEqual(infer_call[1]["model_name"], contract.model_name)
        self.assertEqual(infer_call[1]["request_id"], "req-7")
        self.assertEqual(infer_call[1]["client_timeout"], 3.0)

    async def test_missing_and_extra_inputs_are_rejected_before_remote_call(self):
        contract = MODEL_CONTRACTS["face_rec"]
        client = FakeTritonClient()
        transport = make_transport(client)
        raw = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        landmark = np.zeros((1, 5, 2), dtype=np.float32)

        with self.assertRaisesRegex(InferenceInputError, "missing.*LANDMARK"):
            await transport.infer(contract, {"RAW_IMAGE": raw}, request_id="missing")
        with self.assertRaisesRegex(InferenceInputError, "extra.*UNKNOWN"):
            await transport.infer(
                contract,
                {"RAW_IMAGE": raw, "LANDMARK": landmark, "UNKNOWN": raw},
                request_id="extra",
            )

        self.assertFalse(client.calls)

    async def test_input_construction_errors_are_typed(self):
        contract = MODEL_CONTRACTS["face_det"]
        client = FakeTritonClient()
        transport = make_transport(client)

        class BrokenInferInput(FakeInferInput):
            def set_data_from_numpy(self, data):
                raise ValueError("numpy type mismatch")

        with patch("core.infer.transport.grpcclient.InferInput", BrokenInferInput):
            with self.assertRaisesRegex(
                InferenceInputError, "face_det.*RAW_IMAGE.*numpy type mismatch"
            ):
                await transport.infer(
                    contract,
                    {"RAW_IMAGE": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
                    request_id="bad-input",
                )

    async def test_missing_remote_output_is_a_response_error(self):
        contract = MODEL_CONTRACTS["face_det"]
        client = FakeTritonClient()
        client.infer_result = FakeInferResult(
            {"DETECTIONS": np.zeros((1, 100, 6), dtype=np.float32)}
        )
        transport = make_transport(client)

        with self.assertRaisesRegex(
            InferenceResponseError, "face_det.*NUM_DETECTIONS.*missing"
        ):
            await transport.infer(
                contract,
                {"RAW_IMAGE": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
                request_id="bad-output",
            )

    async def test_non_numpy_remote_output_is_a_response_error(self):
        contract = MODEL_CONTRACTS["face_det"]
        client = FakeTritonClient()
        client.infer_result = FakeInferResult(
            {
                "DETECTIONS": [[0.0]],
                "NUM_DETECTIONS": np.ones((1, 1), dtype=np.int32),
            }
        )
        transport = make_transport(client)

        with self.assertRaisesRegex(
            InferenceResponseError, "face_det.*DETECTIONS.*numpy array"
        ):
            await transport.infer(
                contract,
                {"RAW_IMAGE": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
                request_id="bad-output-type",
            )

    async def test_deadline_is_typed_and_is_not_retried(self):
        client = FakeTritonClient()
        client.infer_effects = [
            InferenceServerException(
                "deadline",
                status=str(grpc.StatusCode.DEADLINE_EXCEEDED),
            )
        ]
        transport = make_transport(client, transient_retries=1)

        with self.assertRaisesRegex(InferenceServerTimeout, "deadline"):
            await transport.infer(
                MODEL_CONTRACTS["face_det"],
                {"RAW_IMAGE": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
                request_id="timeout",
            )

        self.assertEqual([call[0] for call in client.calls], ["infer"])

    async def test_unavailable_is_retried_once_and_then_succeeds(self):
        contract = MODEL_CONTRACTS["face_det"]
        client = FakeTritonClient()
        expected = FakeInferResult(
            {
                "DETECTIONS": np.zeros((1, 100, 6), dtype=np.float32),
                "NUM_DETECTIONS": np.ones((1, 1), dtype=np.int32),
            }
        )
        client.infer_effects = [
            InferenceServerException(
                "temporarily unavailable",
                status=str(grpc.StatusCode.UNAVAILABLE),
            ),
            expected,
        ]
        transport = make_transport(client, transient_retries=1)

        outputs = await transport.infer(
            contract,
            {"RAW_IMAGE": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
            request_id="retry",
        )

        self.assertEqual(set(outputs), {"DETECTIONS", "NUM_DETECTIONS"})
        self.assertEqual([call[0] for call in client.calls], ["infer", "infer"])

    async def test_unavailable_after_retry_budget_is_typed(self):
        unavailable = InferenceServerException(
            "still unavailable", status=str(grpc.StatusCode.UNAVAILABLE)
        )
        client = FakeTritonClient()
        client.infer_effects = [unavailable, unavailable]
        transport = make_transport(client, transient_retries=1)

        with self.assertRaisesRegex(InferenceUnavailable, "still unavailable"):
            await transport.infer(
                MODEL_CONTRACTS["face_det"],
                {"RAW_IMAGE": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
                request_id="retry-exhausted",
            )

        self.assertEqual([call[0] for call in client.calls], ["infer", "infer"])

    async def test_other_remote_errors_use_base_inference_error(self):
        client = FakeTritonClient()
        client.infer_effects = [
            InferenceServerException(
                "internal", status=str(grpc.StatusCode.INTERNAL)
            )
        ]
        transport = make_transport(client)

        with self.assertRaises(TritonInferenceError) as caught:
            await transport.infer(
                MODEL_CONTRACTS["face_det"],
                {"RAW_IMAGE": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
                request_id="internal",
            )

        self.assertIs(type(caught.exception), TritonInferenceError)

    async def test_config_deadline_uses_same_remote_error_mapping(self):
        class DeadlineConfigClient(FakeTritonClient):
            async def get_model_config(self, **kwargs):
                raise InferenceServerException(
                    "config deadline",
                    status=str(grpc.StatusCode.DEADLINE_EXCEEDED),
                )

        with self.assertRaisesRegex(InferenceServerTimeout, "config deadline"):
            await make_transport(DeadlineConfigClient()).get_model_config("model_a")

    async def test_transport_preserves_pool_queue_timeout(self):
        client = FakeTritonClient()
        transport = make_transport(client, acquire_timeout=0.01)
        await transport.start()
        checked_out = await transport._pool.acquire()
        try:
            with self.assertRaises(InferenceQueueTimeout):
                await transport.infer(
                    MODEL_CONTRACTS["face_det"],
                    {"RAW_IMAGE": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
                    request_id="queue-timeout",
                )
        finally:
            await transport._pool.release(checked_out)

    async def test_transport_event_has_wait_context_without_tensor_payload(self):
        contract = MODEL_CONTRACTS["face_det"]
        client = FakeTritonClient()
        client.infer_result = FakeInferResult(
            {
                "DETECTIONS": np.zeros((1, 100, 6), dtype=np.float32),
                "NUM_DETECTIONS": np.ones((1, 1), dtype=np.int32),
            }
        )
        logger = logging.getLogger("tests.triton.transport")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = CollectingHandler()
        logger.handlers = [handler]
        transport = make_transport(client, logger=logger)
        sensitive = np.frombuffer(b"secret-image-bytes", dtype=np.uint8).reshape(
            1, 1, -1, 1
        )

        await transport.infer(
            contract,
            {"RAW_IMAGE": sensitive},
            request_id="safe-request-id",
        )

        self.assertTrue(handler.records)
        record = handler.records[-1]
        self.assertEqual(record.event, "triton_transport")
        self.assertEqual(record.model, contract.model_name)
        self.assertEqual(record.request_id, "safe-request-id")
        self.assertIsInstance(record.client_wait_ms, float)
        self.assertNotIn("secret-image-bytes", repr(record.__dict__))


if __name__ == "__main__":
    unittest.main()
