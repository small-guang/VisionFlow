from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from core.infer.capabilities import ModelCapability
from core.infer.contracts import MODEL_CONTRACTS
from core.infer.errors import InferenceUnavailable, TritonContractError
from core.infer.results import FeatureResult
from core.infer.triton_client import UnifiedTritonClient


def facade_settings(**overrides):
    values = {
        "TRITON_SERVER_URL": "triton.example:8001",
        "TRITON_BATCH_TARGET_DEFAULT": 1,
        "TRITON_BATCH_WAIT_MS_DEFAULT": 0.0,
        "TRITON_BATCH_TARGETS": {},
        "TRITON_BATCH_WAIT_MS": {},
        "TRITON_QUEUE_TIMEOUT_SECONDS": 60.0,
        "TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS": 5.0,
        "TRITON_INFER_TIMEOUT_SECONDS": 30.0,
        "TRITON_CONFIG_TIMEOUT_SECONDS": 4.0,
        "TRITON_CLIENT_POOL_SIZE": 7,
        "TRITON_EXECUTOR_CONCURRENCY": 4,
        "TRITON_EXECUTOR_QUEUE_CAPACITY": 16,
        "TRITON_TRANSIENT_RETRIES": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def target_one_capabilities():
    return {
        key: ModelCapability(
            key=key,
            model_name=contract.model_name,
            api_contract_version=contract.api_contract_version,
            server_max_batch=8,
            target_batch=1,
        )
        for key, contract in MODEL_CONTRACTS.items()
    }


class FakeTransport:
    def __init__(self, *, ready: bool = True):
        self.ready = ready
        self.started = False
        self.closed = False
        self.start_calls = 0
        self.close_calls = 0

    async def start(self):
        self.start_calls += 1
        self.started = True

    async def is_server_ready(self):
        return self.ready

    async def close(self):
        self.close_calls += 1
        self.closed = True

    async def infer(self, contract, inputs, *, request_id):
        del inputs, request_id
        if contract.key == "sbs_image":
            return {"embedding": np.ones((1, 2048), dtype=np.float32)}
        if contract.key == "person_pose":
            keypoints = np.zeros((1, 100, 17, 3), dtype=np.float32)
            keypoints[0, 0, :, 2] = 0.9
            detections = np.zeros((1, 100, 6), dtype=np.float32)
            detections[0, 0] = [1, 1, 9, 9, 0.95, 0]
            return {
                "DETECTIONS": detections,
                "KEYPOINTS": keypoints,
                "NUM_DETECTIONS": np.array([[1]], dtype=np.int32),
            }
        raise AssertionError(f"unexpected inference for {contract.key}")


class FakeExecutor:
    def __init__(
        self,
        *,
        order=None,
        start_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ):
        self.order = order
        self.start_error = start_error
        self.close_error = close_error
        self.started = False
        self.closed = False
        self.start_calls = 0
        self.close_calls = 0

    async def start(self):
        self.start_calls += 1
        if self.order is not None:
            self.order.append("executor.start")
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    async def submit(self, _model, operation, *, deadline):
        del deadline
        return await operation()

    async def close(self):
        self.close_calls += 1
        self.closed = True
        if self.order is not None:
            self.order.append("executor.close")
        if self.close_error is not None:
            raise self.close_error


class OrderedTransport(FakeTransport):
    def __init__(self, order, *, ready: bool = True):
        super().__init__(ready=ready)
        self.order = order

    async def close(self):
        self.order.append("transport.close")
        await super().close()


class FailingStartTransport(OrderedTransport):
    def __init__(self, order, error):
        super().__init__(order)
        self.error = error

    async def start(self):
        self.start_calls += 1
        raise self.error


class BlockingStartupTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.readiness_entered = asyncio.Event()
        self.close_entered = asyncio.Event()
        self.allow_close = asyncio.Event()

    async def is_server_ready(self):
        self.readiness_entered.set()
        await asyncio.Event().wait()

    async def close(self):
        self.close_calls += 1
        self.close_entered.set()
        await self.allow_close.wait()
        self.closed = True


class BlockingCloseTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.close_entered = asyncio.Event()
        self.allow_close = asyncio.Event()

    async def close(self):
        self.close_calls += 1
        self.close_entered.set()
        await self.allow_close.wait()
        self.closed = True


class FailingBlockingCloseTransport(BlockingCloseTransport):
    async def close(self):
        self.close_calls += 1
        self.close_entered.set()
        await self.allow_close.wait()
        raise RuntimeError("transport close failed")


class CapturingLogger:
    def __init__(self):
        self.bindings = []
        self.errors = []

    def bind(self, **values):
        self.bindings.append(values)
        return self

    def info(self, *_args, **_kwargs):
        pass

    def error(self, message, *args, **_kwargs):
        self.errors.append((message, args))


class ClosingProbe:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        wait_ms: float = 0.0,
    ):
        self.error = error
        self.wait_ms = wait_ms
        self.closed = False

    async def close(self):
        self.closed = True
        if self.error is not None:
            raise self.error


class OrderedScheduler(ClosingProbe):
    def __init__(self, order, label, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.label = label

    async def close(self):
        self.order.append(f"scheduler.close:{self.label}")
        await super().close()


class TritonPoseClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_start_failure_closes_executor_then_transport(self):
        order = []
        failure = RuntimeError("transport start failed")
        transport = FailingStartTransport(order, failure)
        executor = FakeExecutor(order=order)
        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=transport,
            executor=executor,
        )

        with self.assertRaises(RuntimeError) as raised:
            await client.initialize()

        self.assertIs(raised.exception, failure)
        self.assertEqual(order, ["executor.close", "transport.close"])
        self.assertEqual(executor.start_calls, 0)
        self.assertEqual(executor.close_calls, 1)
        self.assertEqual(transport.close_calls, 1)
        self.assertEqual(client.schedulers, {})

    async def test_adapter_build_failure_closes_executor_then_transport(self):
        order = []
        failure = RuntimeError("adapter build failed")
        transport = OrderedTransport(order)
        executor = FakeExecutor(order=order)
        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=transport,
            executor=executor,
        )

        with (
            patch(
                "core.infer.triton_client.discover_all_capabilities",
                new=AsyncMock(return_value=target_one_capabilities()),
            ),
            patch(
                "core.infer.triton_client.build_adapters",
                side_effect=failure,
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                await client.initialize()

        self.assertIs(raised.exception, failure)
        self.assertEqual(order, ["executor.close", "transport.close"])
        self.assertEqual(executor.start_calls, 0)
        self.assertEqual(executor.close_calls, 1)
        self.assertEqual(transport.close_calls, 1)
        self.assertEqual(client.schedulers, {})

    async def test_cancelled_close_waiter_can_observe_later_close_failure(self):
        transport = FailingBlockingCloseTransport()
        active_logger = CapturingLogger()
        client = UnifiedTritonClient(
            settings=facade_settings(), transport=transport
        )
        with (
            patch(
                "core.infer.triton_client.discover_all_capabilities",
                new=AsyncMock(return_value=target_one_capabilities()),
            ),
            patch("core.infer.triton_client.logger", active_logger),
        ):
            await client.initialize()
            close_waiter = asyncio.create_task(client.close())
            await transport.close_entered.wait()
            close_waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await close_waiter

            transport.allow_close.set()
            for _ in range(20):
                if client._close_task.done():
                    break
                await asyncio.sleep(0)
            self.assertTrue(client._close_task.done())

            with self.assertRaisesRegex(
                RuntimeError, "transport close failed"
            ):
                await client.close()

        self.assertEqual(transport.close_calls, 1)
        self.assertIsInstance(client._close_error, RuntimeError)
        self.assertTrue(active_logger.errors)
        self.assertTrue(
            any(
                binding.get("error_classes") == ["RuntimeError"]
                for binding in active_logger.bindings
            )
        )

    async def test_repeated_startup_cancellation_cannot_cancel_cleanup(self):
        transport = BlockingStartupTransport()
        client = UnifiedTritonClient(
            settings=facade_settings(), transport=transport
        )
        initialize_task = asyncio.create_task(client.initialize())
        await transport.readiness_entered.wait()

        initialize_task.cancel()
        await transport.close_entered.wait()
        initialize_task.cancel()
        await asyncio.sleep(0)

        self.assertFalse(transport.closed)
        self.assertFalse(initialize_task.done())
        transport.allow_close.set()
        with self.assertRaises(asyncio.CancelledError):
            await initialize_task

        self.assertTrue(transport.closed)
        self.assertEqual(client._state, client._CLOSED)
        await client.close()
        self.assertEqual(transport.close_calls, 1)
        self.assertFalse(
            any(
                task.get_name() == "triton-facade-startup-cleanup"
                and not task.done()
                for task in asyncio.all_tasks()
            )
        )

    async def test_cleanup_cancellation_preserves_original_startup_error(self):
        transport = BlockingCloseTransport()
        probes = []

        def build_scheduler(**_kwargs):
            if len(probes) == 2:
                raise RuntimeError("original scheduler failure")
            probe = ClosingProbe()
            probes.append(probe)
            return probe

        client = UnifiedTritonClient(
            settings=facade_settings(), transport=transport
        )
        with (
            patch(
                "core.infer.triton_client.discover_all_capabilities",
                new=AsyncMock(return_value=target_one_capabilities()),
            ),
            patch(
                "core.infer.triton_client.ModelBatchScheduler",
                side_effect=build_scheduler,
            ),
        ):
            initialize_task = asyncio.create_task(client.initialize())
            await transport.close_entered.wait()
            initialize_task.cancel()
            initialize_task.cancel()
            transport.allow_close.set()
            with self.assertRaisesRegex(
                RuntimeError, "original scheduler failure"
            ):
                await initialize_task

        self.assertTrue(all(probe.closed for probe in probes))
        self.assertTrue(transport.closed)
        self.assertEqual(client._state, client._CLOSED)
        await client.close()
        self.assertEqual(transport.close_calls, 1)

    async def test_default_transport_receives_all_transport_settings(self):
        settings = facade_settings()
        transport = FakeTransport()
        executor = FakeExecutor()
        with (
            patch(
                "core.infer.triton_client.TritonTransport",
                return_value=transport,
            ) as transport_factory,
            patch(
                "core.infer.triton_client.FairInferenceExecutor",
                return_value=executor,
            ) as executor_factory,
        ):
            client = UnifiedTritonClient(settings=settings)

        transport_factory.assert_called_once_with(
            url="triton.example:8001",
            pool_size=7,
            acquire_timeout=5.0,
            infer_timeout=30.0,
            config_timeout=4.0,
            transient_retries=1,
            shm_slots=0,
            shm_slot_bytes=33554432,
        )
        executor_factory.assert_called_once_with(
            concurrency=4,
            queue_capacity=16,
        )
        self.assertIs(client.transport, transport)
        self.assertIs(client.executor, executor)
        await client.close()

    async def test_default_transport_receives_shared_memory_settings(self):
        settings = facade_settings(
            TRITON_SYSTEM_SHM_SLOTS=8,
            TRITON_SYSTEM_SHM_SLOT_BYTES=16777216,
        )
        with patch(
            "core.infer.triton_client.TritonTransport", return_value=FakeTransport()
        ) as factory:
            client = UnifiedTritonClient(settings=settings)
        self.assertEqual(factory.call_args.kwargs["shm_slots"], 8)
        self.assertEqual(factory.call_args.kwargs["shm_slot_bytes"], 16777216)
        await client.close()

    async def test_injected_executor_is_reused_without_constructing_another(self):
        executor = FakeExecutor()
        with patch(
            "core.infer.triton_client.FairInferenceExecutor"
        ) as executor_factory:
            client = UnifiedTritonClient(
                settings=facade_settings(),
                transport=FakeTransport(),
                executor=executor,
            )

        executor_factory.assert_not_called()
        self.assertIs(client.executor, executor)
        await client.close()

    async def test_initialize_publishes_all_capabilities_and_schedulers(self):
        transport = FakeTransport()
        executor = FakeExecutor()
        capabilities = target_one_capabilities()
        discovery = AsyncMock(return_value=capabilities)

        with patch(
            "core.infer.triton_client.discover_all_capabilities", discovery
        ):
            client = UnifiedTritonClient(
                settings=facade_settings(),
                transport=transport,
                executor=executor,
            )
            await client.initialize()

        self.assertTrue(transport.started)
        self.assertTrue(executor.started)
        self.assertEqual(executor.start_calls, 1)
        self.assertEqual(set(client.capabilities), set(MODEL_CONTRACTS))
        self.assertEqual(set(client.schedulers), set(MODEL_CONTRACTS))
        self.assertTrue(
            all(
                scheduler.executor is executor
                for scheduler in client.schedulers.values()
            )
        )
        self.assertTrue(
            all(item.target_batch == 1 for item in client.capabilities.values())
        )
        discovery.assert_awaited_once()

        await client.initialize()
        await client.close()
        await client.close()
        self.assertTrue(transport.closed)
        self.assertEqual(transport.start_calls, 1)
        self.assertEqual(transport.close_calls, 1)
        self.assertEqual(executor.close_calls, 1)

    async def test_executor_starts_before_any_scheduler_is_constructed(self):
        order = []
        executor = FakeExecutor(order=order)

        def build_scheduler(**kwargs):
            order.append("scheduler.construct")
            probe = ClosingProbe(wait_ms=kwargs["wait_ms"])
            probe.executor = kwargs["executor"]
            return probe

        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=FakeTransport(),
            executor=executor,
        )
        with (
            patch(
                "core.infer.triton_client.discover_all_capabilities",
                new=AsyncMock(return_value=target_one_capabilities()),
            ),
            patch(
                "core.infer.triton_client.ModelBatchScheduler",
                side_effect=build_scheduler,
            ),
        ):
            await client.initialize()

        self.assertEqual(order[0], "executor.start")
        self.assertEqual(
            order[1:], ["scheduler.construct"] * len(MODEL_CONTRACTS)
        )
        self.assertTrue(
            all(
                scheduler.executor is executor
                for scheduler in client.schedulers.values()
            )
        )
        await client.close()

    async def test_per_model_target_and_wait_are_resolved_by_stable_key(self):
        settings = facade_settings(
            TRITON_BATCH_TARGETS={"face_det": 3},
            TRITON_BATCH_WAIT_MS={"face_det": 4.25},
        )
        capabilities = target_one_capabilities()
        face_det = capabilities["face_det"]
        capabilities["face_det"] = ModelCapability(
            key=face_det.key,
            model_name=face_det.model_name,
            api_contract_version=face_det.api_contract_version,
            server_max_batch=face_det.server_max_batch,
            target_batch=3,
        )
        discovery = AsyncMock(return_value=capabilities)
        client = UnifiedTritonClient(
            settings=settings, transport=FakeTransport()
        )

        with patch(
            "core.infer.triton_client.discover_all_capabilities", discovery
        ):
            await client.initialize()

        targets = discovery.await_args.args[2]
        self.assertEqual(set(targets), set(MODEL_CONTRACTS))
        self.assertEqual(targets["face_det"], 3)
        self.assertTrue(
            all(
                target == 1
                for key, target in targets.items()
                if key != "face_det"
            )
        )
        self.assertEqual(client.schedulers["face_det"].wait_ms, 4.25)
        self.assertTrue(
            all(
                scheduler.wait_ms == 0.0
                for key, scheduler in client.schedulers.items()
                if key != "face_det"
            )
        )
        await client.close()

    async def test_server_unready_closes_transport_without_publishing_schedulers(self):
        order = []
        transport = OrderedTransport(order, ready=False)
        executor = FakeExecutor(order=order)
        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=transport,
            executor=executor,
        )

        with self.assertRaises(InferenceUnavailable):
            await client.initialize()

        self.assertTrue(transport.closed)
        self.assertTrue(executor.closed)
        self.assertEqual(order, ["executor.close", "transport.close"])
        self.assertEqual(client.capabilities, {})
        self.assertEqual(client.schedulers, {})

    async def test_contract_failure_closes_transport_without_publishing_schedulers(self):
        order = []
        transport = OrderedTransport(order)
        cleanup_failure = RuntimeError("executor cleanup failed")
        executor = FakeExecutor(
            order=order,
            close_error=cleanup_failure,
        )
        failure = TritonContractError("person_pose: incompatible")
        discovery = AsyncMock(side_effect=failure)
        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=transport,
            executor=executor,
        )

        with patch(
            "core.infer.triton_client.discover_all_capabilities", discovery
        ):
            with self.assertRaises(TritonContractError) as raised:
                await client.initialize()

        self.assertIs(raised.exception, failure)
        self.assertTrue(transport.closed)
        self.assertTrue(executor.closed)
        self.assertEqual(order, ["executor.close", "transport.close"])
        self.assertEqual(client.capabilities, {})
        self.assertEqual(client.schedulers, {})

    async def test_executor_start_failure_cleans_executor_then_transport(self):
        order = []
        transport = OrderedTransport(order)
        failure = RuntimeError("executor start failed")
        executor = FakeExecutor(order=order, start_error=failure)
        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=transport,
            executor=executor,
        )

        with patch(
            "core.infer.triton_client.discover_all_capabilities",
            new=AsyncMock(return_value=target_one_capabilities()),
        ):
            with self.assertRaises(RuntimeError) as raised:
                await client.initialize()

        self.assertIs(raised.exception, failure)
        self.assertEqual(
            order,
            ["executor.start", "executor.close", "transport.close"],
        )
        self.assertEqual(client.capabilities, {})
        self.assertEqual(client.schedulers, {})

    async def test_discovery_registry_requires_exact_contract_keys(self):
        base = target_one_capabilities()
        missing = dict(base)
        missing.pop("person_pose")
        extra = dict(base)
        sample = next(iter(base.values()))
        extra["unexpected"] = ModelCapability(
            key="unexpected",
            model_name=sample.model_name,
            api_contract_version=sample.api_contract_version,
            server_max_batch=sample.server_max_batch,
            target_batch=sample.target_batch,
        )

        for label, capabilities in (("missing", missing), ("extra", extra)):
            with self.subTest(label=label):
                transport = FakeTransport()
                client = UnifiedTritonClient(
                    settings=facade_settings(), transport=transport
                )
                with (
                    patch(
                        "core.infer.triton_client.discover_all_capabilities",
                        new=AsyncMock(return_value=capabilities),
                    ),
                    patch(
                        "core.infer.triton_client.ModelBatchScheduler",
                        side_effect=lambda **kwargs: ClosingProbe(
                            wait_ms=kwargs["wait_ms"]
                        ),
                    ) as scheduler_factory,
                ):
                    try:
                        with self.assertRaisesRegex(
                            TritonContractError, label
                        ):
                            await client.initialize()
                    finally:
                        if client._state == client._OPEN:
                            await client.close()

                scheduler_factory.assert_not_called()
                self.assertTrue(transport.closed)
                self.assertEqual(client.capabilities, {})
                self.assertEqual(client.schedulers, {})

    async def test_scheduler_construction_failure_closes_partial_resources(self):
        order = []
        transport = OrderedTransport(order)
        executor = FakeExecutor(order=order)
        probes = []

        def build_scheduler(**kwargs):
            if len(probes) == 4:
                raise RuntimeError("scheduler construction failed")
            probe = OrderedScheduler(
                order,
                str(len(probes)),
                wait_ms=kwargs["wait_ms"],
            )
            probes.append(probe)
            return probe

        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=transport,
            executor=executor,
        )
        with (
            patch(
                "core.infer.triton_client.discover_all_capabilities",
                new=AsyncMock(return_value=target_one_capabilities()),
            ),
            patch(
                "core.infer.triton_client.ModelBatchScheduler",
                side_effect=build_scheduler,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "scheduler construction failed"
            ):
                await client.initialize()

        self.assertEqual(len(probes), 4)
        self.assertTrue(all(probe.closed for probe in probes))
        self.assertTrue(executor.closed)
        self.assertTrue(transport.closed)
        self.assertEqual(
            order,
            [
                "executor.start",
                "scheduler.close:0",
                "scheduler.close:1",
                "scheduler.close:2",
                "scheduler.close:3",
                "executor.close",
                "transport.close",
            ],
        )
        self.assertEqual(client.capabilities, {})
        self.assertEqual(client.schedulers, {})

    async def test_close_attempts_every_scheduler_before_transport(self):
        order = []
        transport = OrderedTransport(order)
        executor = FakeExecutor(order=order)
        probes = []
        first_failure = RuntimeError("first close failed")

        def build_scheduler(**kwargs):
            error = first_failure if not probes else None
            probe = OrderedScheduler(
                order,
                str(len(probes)),
                error=error,
                wait_ms=kwargs["wait_ms"],
            )
            probes.append(probe)
            return probe

        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=transport,
            executor=executor,
        )
        with (
            patch(
                "core.infer.triton_client.discover_all_capabilities",
                new=AsyncMock(return_value=target_one_capabilities()),
            ),
            patch(
                "core.infer.triton_client.ModelBatchScheduler",
                side_effect=build_scheduler,
            ),
        ):
            await client.initialize()

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError, "first close failed"
        ) as raised:
            await client.close()

        self.assertEqual(len(probes), len(MODEL_CONTRACTS))
        self.assertTrue(all(probe.closed for probe in probes))
        self.assertTrue(executor.closed)
        self.assertTrue(transport.closed)
        self.assertIs(raised.exception.__cause__, first_failure)
        self.assertEqual(
            order,
            [
                *(f"scheduler.close:{index}" for index in range(len(probes))),
                "executor.close",
                "transport.close",
            ],
        )

    async def test_executor_close_failure_still_closes_transport(self):
        order = []
        transport = OrderedTransport(order)
        failure = RuntimeError("executor close failed")
        executor = FakeExecutor(order=order, close_error=failure)
        probes = []

        def build_scheduler(**kwargs):
            probe = OrderedScheduler(
                order,
                str(len(probes)),
                wait_ms=kwargs["wait_ms"],
            )
            probes.append(probe)
            return probe

        client = UnifiedTritonClient(
            settings=facade_settings(),
            transport=transport,
            executor=executor,
        )
        with (
            patch(
                "core.infer.triton_client.discover_all_capabilities",
                new=AsyncMock(return_value=target_one_capabilities()),
            ),
            patch(
                "core.infer.triton_client.ModelBatchScheduler",
                side_effect=build_scheduler,
            ),
        ):
            await client.initialize()

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError, "executor close failed"
        ) as raised:
            await client.close()

        self.assertIs(raised.exception.__cause__, failure)
        self.assertTrue(all(probe.closed for probe in probes))
        self.assertTrue(transport.closed)
        self.assertEqual(
            order,
            [
                *(f"scheduler.close:{index}" for index in range(len(probes))),
                "executor.close",
                "transport.close",
            ],
        )

    async def test_public_registries_cannot_mutate_owned_resources(self):
        transport = FakeTransport()
        probes = []

        def build_scheduler(**kwargs):
            probe = ClosingProbe(wait_ms=kwargs["wait_ms"])
            probes.append(probe)
            return probe

        client = UnifiedTritonClient(
            settings=facade_settings(), transport=transport
        )
        with (
            patch(
                "core.infer.triton_client.discover_all_capabilities",
                new=AsyncMock(return_value=target_one_capabilities()),
            ),
            patch(
                "core.infer.triton_client.ModelBatchScheduler",
                side_effect=build_scheduler,
            ),
        ):
            await client.initialize()

        with self.assertRaises(AttributeError):
            client.schedulers.clear()
        with self.assertRaises(AttributeError):
            client.capabilities.clear()
        with self.assertRaises(TypeError):
            client.schedulers["intruder"] = ClosingProbe()
        with self.assertRaises(AttributeError):
            client.schedulers = {}
        with self.assertRaises(AttributeError):
            client.capabilities = {}

        await client.close()

        self.assertEqual(len(probes), len(MODEL_CONTRACTS))
        self.assertTrue(all(probe.closed for probe in probes))
        self.assertTrue(transport.closed)

    async def test_sbs_and_pose_methods_return_business_types(self):
        transport = FakeTransport()
        client = UnifiedTritonClient(
            settings=facade_settings(), transport=transport
        )
        with patch(
            "core.infer.triton_client.discover_all_capabilities",
            new=AsyncMock(return_value=target_one_capabilities()),
        ):
            await client.initialize()

        feature = await client.extract_sbs_features(
            np.arange(10, dtype=np.uint8)
        )
        poses = await client.extract_person_poses(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )

        self.assertIsInstance(feature, FeatureResult)
        self.assertEqual(feature.vector.shape, (2048,))
        self.assertAlmostEqual(
            float(np.linalg.norm(feature.vector)), 1.0, places=6
        )
        self.assertIsInstance(poses, list)
        self.assertEqual(poses[0]["keypoints"].shape, (17, 3))
        await client.close()

    async def test_friendly_method_before_initialize_has_typed_failure(self):
        client = UnifiedTritonClient(
            settings=facade_settings(), transport=FakeTransport()
        )

        with self.assertRaises(InferenceUnavailable):
            await client.extract_person_poses(
                np.zeros((10, 10, 3), dtype=np.uint8)
            )


if __name__ == "__main__":
    unittest.main()
