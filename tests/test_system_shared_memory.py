from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import grpc
import numpy as np
from tritonclient.utils import InferenceServerException

from core.infer.contracts import MODEL_CONTRACTS
from core.infer.errors import InferenceInputError, InferenceServerTimeout, TritonInferenceError
from core.infer.metrics import InferenceMetrics, active_inference_metrics
from core.infer.system_shared_memory import SystemSharedMemoryPool
from core.infer.transport import TritonTransport
from core.performance import RuntimePerformance


MIB = 1024 * 1024


class FakeResult:
    def as_numpy(self, name):
        return np.zeros((1,), dtype=np.int32 if name == "NUM_DETECTIONS" else np.float32)


class FakeClient:
    def __init__(self, state):
        self.state = state

    async def register_system_shared_memory(self, **kwargs):
        self.state.registered.append(kwargs)
        if self.state.register_operation is not None:
            await self.state.register_operation(kwargs)
        self.state.registry[kwargs["name"]] = kwargs

    async def get_system_shared_memory_status(self, **kwargs):
        self.state.status_queries.append(kwargs)
        if self.state.status_operation is not None:
            return await self.state.status_operation(kwargs)
        return {"regions": dict(self.state.registry)}

    async def unregister_system_shared_memory(self, **kwargs):
        self.state.unregistered.append(kwargs)
        if self.state.unregister_operation is not None:
            await self.state.unregister_operation(kwargs)
        self.state.registry.pop(kwargs["name"], None)

    async def infer(self, **kwargs):
        self.state.inferred.append(kwargs)
        if self.state.infer_operation is not None:
            return await self.state.infer_operation(kwargs)
        return FakeResult()

    async def close(self):
        self.state.closed_clients.append(self)


class FakeState:
    def __init__(self):
        self.created = []
        self.destroyed = []
        self.copied = []
        self.clients = []
        self.closed_clients = []
        self.registered = []
        self.registry = {}
        self.status_queries = []
        self.status_operation = None
        self.unregistered = []
        self.inferred = []
        self.register_operation = None
        self.unregister_operation = None
        self.infer_operation = None
        self.copy_operation = None
        self.events = []

    def client_factory(self, *args):
        client = FakeClient(self)
        self.clients.append(client)
        return client

    def create(self, name, key, byte_size, *, create_only):
        if not create_only:
            raise AssertionError("must never reopen or truncate an existing SHM object")
        handle = SimpleNamespace(name=name, key=key, data=bytearray(byte_size))
        self.created.append(handle)
        self.events.append(("create", name))
        return handle

    def copy(self, handle, values, offset=0):
        if self.copy_operation is not None:
            self.copy_operation(handle, values)
        array = values[0]
        handle.data[offset:offset + array.nbytes] = array.tobytes()
        self.copied.append((handle, array.shape, array.dtype, threading.get_ident()))
        self.events.append(("copy", handle.name))

    def destroy(self, handle):
        if handle in self.destroyed:
            raise AssertionError("destroyed twice")
        self.destroyed.append(handle)
        self.events.append(("destroy", handle.name))


def region(input_tensor):
    # 检查 SDK 最终 wire metadata；生产代码只调用公开 API。
    parameter = input_tensor._input.parameters.get("shared_memory_region")
    return parameter.string_param if parameter is not None else None


def large_image(value=7):
    return np.full((1, 640, 640, 3), value, dtype=np.uint8)


def missing_region_error(name):
    return InferenceServerException(
        f"Unable to find shared memory region: '{name}'",
        status=str(grpc.StatusCode.NOT_FOUND),
    )


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


class ShmTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = FakeState()
        self.records = []
        recorder = SimpleNamespace(
            record=lambda name, duration, **kwargs: self.records.append(
                (name, duration, kwargs, threading.get_ident())
            )
        )
        for target, replacement in (
            ("core.infer.system_shared_memory.shm.create_shared_memory_region", self.state.create),
            ("core.infer.system_shared_memory.shm.set_shared_memory_region", self.state.copy),
            ("core.infer.system_shared_memory.shm.destroy_shared_memory_region", self.state.destroy),
            ("core.infer.system_shared_memory.os.statvfs", lambda path: SimpleNamespace(f_bavail=1024, f_frsize=MIB)),
            ("core.infer.system_shared_memory.runtime_performance", recorder),
            ("core.infer.transport.runtime_performance", recorder),
        ):
            patcher = patch(target, replacement, create=target.endswith(".os.statvfs"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_pool(self, *, cleanup=True, **overrides):
        options = dict(
            slots=2, slot_bytes=2 * MIB,
            client_factory=self.state.client_factory, config_timeout=2,
        )
        options.update(overrides)
        pool = SystemSharedMemoryPool(**options)
        if cleanup:
            self.addAsyncCleanup(pool.close)
        return pool

    def make_transport(self, **overrides):
        options = dict(
            url="triton.example:8001", pool_size=2, acquire_timeout=1,
            infer_timeout=3, config_timeout=2, transient_retries=0,
            shm_slots=1, shm_slot_bytes=2 * MIB,
            client_factory=self.state.client_factory,
        )
        options.update(overrides)
        transport = TritonTransport(**options)
        self.addAsyncCleanup(transport.close)
        return transport

    async def infer(self, transport, array=None, *, model="face_det", request_id="request"):
        return await transport.infer(
            MODEL_CONTRACTS[model],
            {"RAW_IMAGE": large_image() if array is None else array},
            request_id=request_id,
        )


class SharedMemoryPoolTests(ShmTestCase):
    async def test_fixed_unique_exclusive_slots_poisoned_slot_is_never_replaced(self):
        pool = self.make_pool()
        self.assertIsNone(pool.borrow())
        await pool.start()
        await pool.start()
        first, second = pool.borrow(), pool.borrow()
        self.assertIsNot(first, second)
        self.assertIsNone(pool.borrow())
        pool.poison(second)
        pool.poison(second)
        pool.release(second)
        self.assertIsNone(pool.borrow())
        pool.release(first)
        self.assertIs(pool.borrow(), first)
        pool.release(first)
        await pool.close()
        self.assertEqual(len(self.state.created), 2)
        self.assertEqual(len({h.name for h in self.state.created}), 2)
        self.assertTrue(all(h.key == "/" + h.name for h in self.state.created))
        self.assertEqual(self.state.created, self.state.destroyed)
        self.assertEqual([x["name"] for x in self.state.registered], [x["name"] for x in self.state.unregistered])
        self.assertTrue(all(x["client_timeout"] == 2 and x["name"] for x in self.state.unregistered))
        self.assertEqual(sum(r[0] == "triton_shm_poisoned" for r in self.records), 1)

    async def test_copy_preserves_strided_array_values_and_metrics_return_to_loop(self):
        pool = self.make_pool(slots=1)
        await pool.start()
        slot = pool.borrow()
        array = np.arange(640 * 640 * 3, dtype=np.uint8).reshape(1, 640, 640, 3)[:, :, ::-1, :]
        try:
            await pool.copy(slot, array)
            self.assertEqual(slot.handle.data[:array.nbytes], array.tobytes())
            self.assertNotEqual(self.state.copied[0][3], threading.get_ident())
            self.assertEqual({r[0] for r in self.records}, {"triton_shm_copy_run", "triton_shm_copy_queue"})
            self.assertTrue(all(r[3] == threading.get_ident() for r in self.records))
        finally:
            pool.release(slot)

    async def test_repeated_cancel_during_copy_drains_before_release_and_destroy(self):
        pool = self.make_pool(slots=1)
        await pool.start()
        entered, release = threading.Event(), threading.Event()

        def block(handle, values):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release copy")

        self.state.copy_operation = block
        slot = pool.borrow()

        async def use():
            try:
                await pool.copy(slot, large_image())
            finally:
                pool.release(slot)

        task = asyncio.create_task(use())
        closer = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertIsNone(pool.borrow())
            closer = asyncio.create_task(pool.close())
            await asyncio.sleep(0)
            self.assertFalse(closer.done())
            self.assertEqual(self.state.destroyed, [])
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closer, 1)
            self.assertEqual([event[0] for event in self.state.events], ["create", "copy", "destroy"])
        finally:
            release.set()
            await asyncio.gather(*(t for t in (task, closer) if t), return_exceptions=True)

    async def test_cancelled_close_keeps_shared_cleanup_and_waits_for_leases(self):
        pool = self.make_pool(slots=1)
        await pool.start()
        slot = pool.borrow()
        closer = asyncio.create_task(pool.close())
        await wait_until(lambda: pool._close_task is not None)
        closer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closer
        self.assertEqual(self.state.destroyed, [])
        self.assertIsNone(pool.borrow())
        pool.release(slot)
        await asyncio.wait_for(pool.close(), 1)
        await pool.close()
        self.assertEqual(len(self.state.destroyed), 1)

    async def test_partial_registration_failure_cleans_each_attempted_own_name(self):
        async def fail_second(kwargs):
            if len(self.state.registered) == 2:
                raise RuntimeError("registration failed")

        self.state.register_operation = fail_second
        pool = self.make_pool(slots=3)
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            await pool.start()
        self.assertIsNone(pool.borrow())
        await pool.close()
        self.assertEqual(len(self.state.created), 2)
        self.assertEqual(self.state.created, self.state.destroyed)
        self.assertEqual([x["name"] for x in self.state.registered], [x["name"] for x in self.state.unregistered])
        self.assertEqual(self.state.clients, self.state.closed_clients)

    async def test_creation_collision_never_reopens_or_destroys_foreign_object(self):
        create = self.state.create

        def fail_second(*args, **kwargs):
            if self.state.created:
                raise FileExistsError("foreign object")
            return create(*args, **kwargs)

        pool = self.make_pool()
        with patch("core.infer.system_shared_memory.shm.create_shared_memory_region", side_effect=fail_second):
            with self.assertRaises(FileExistsError):
                await pool.start()
        await pool.close()
        self.assertEqual(len(self.state.created), 1)
        self.assertEqual(self.state.created, self.state.destroyed)
        self.assertEqual(len(self.state.unregistered), 1)

    async def test_unregister_failure_still_destroys_own_regions_and_closes_control(self):
        async def fail_unregister(kwargs):
            raise RuntimeError("unregister failed")

        pool = self.make_pool(cleanup=False)
        await pool.start()
        self.state.unregister_operation = fail_unregister
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "failed to clean up"):
                await pool.close()
        self.assertEqual(len(self.state.unregistered), 2)
        self.assertEqual(self.state.created, self.state.destroyed)
        self.assertEqual(self.state.clients, self.state.closed_clients)

    async def test_cancelled_initializing_caller_does_not_abandon_registration(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def block_register(kwargs):
            entered.set()
            await release.wait()

        self.state.register_operation = block_register
        pool = self.make_pool(slots=1)
        starter = asyncio.create_task(pool.start())
        await entered.wait()
        starter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await starter
        closer = asyncio.create_task(pool.close())
        await asyncio.sleep(0)
        self.assertEqual(self.state.destroyed, [])
        release.set()
        await asyncio.wait_for(closer, 1)
        self.assertEqual(self.state.created, self.state.destroyed)

    async def test_not_started_close_allocates_nothing_and_insufficient_space_fails_early(self):
        pool = self.make_pool()
        await pool.close()
        self.assertEqual(self.state.created, [])
        self.assertEqual(self.state.clients, [])
        another = self.make_pool()
        with patch("core.infer.system_shared_memory.os.statvfs", return_value=SimpleNamespace(f_bavail=0, f_frsize=MIB)):
            with self.assertRaisesRegex(RuntimeError, "insufficient free space"):
                await another.start()
        self.assertEqual(self.state.created, [])
        self.assertEqual(self.state.clients, [])

    async def test_copy_executor_has_at_most_four_real_workers(self):
        pool = self.make_pool(slots=5)
        await pool.start()
        guard = threading.Lock()
        entered, release = threading.Event(), threading.Event()
        active = peak = 0

        def block(handle, values):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(active, peak)
                if active == 4:
                    entered.set()
            try:
                if not release.wait(2):
                    raise TimeoutError("test did not release copies")
            finally:
                with guard:
                    active -= 1

        self.state.copy_operation = block
        slots = [pool.borrow() for _ in range(5)]

        async def use(slot):
            try:
                await pool.copy(slot, large_image())
            finally:
                pool.release(slot)

        tasks = [asyncio.create_task(use(slot)) for slot in slots]
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            await asyncio.sleep(0)
            self.assertEqual(peak, 4)
            release.set()
            await asyncio.gather(*tasks)
            self.assertEqual(len(self.state.copied), 5)
            self.assertEqual(peak, 4)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


class SharedMemoryTransportTests(ShmTestCase):
    async def test_repair_outcomes_are_aggregated_by_real_runtime_metrics(self):
        transport = self.make_transport()
        await transport.start()
        self.state.registry.clear()

        async def check_registration(kwargs):
            name = region(kwargs["inputs"][0])
            if name not in self.state.registry:
                raise missing_region_error(name)
            return FakeResult()

        async def invalid_status(kwargs):
            return {"regions": []}

        self.state.infer_operation = check_registration
        metrics = RuntimePerformance()
        with patch("core.infer.system_shared_memory.runtime_performance", metrics), patch(
            "core.infer.transport.runtime_performance", metrics
        ):
            await self.infer(transport)
            self.state.registry.clear()
            self.state.status_operation = invalid_status
            with self.assertRaises(TritonInferenceError):
                await self.infer(transport)
        operations = metrics.snapshot()["operations"]
        self.assertEqual(operations["triton_shm_reregister_attempt"]["count"], 2)
        self.assertEqual(operations["triton_shm_reregistered"]["count"], 1)
        self.assertEqual(operations["triton_shm_reregister_failed"]["count"], 1)
        self.assertEqual(operations["triton_shm_poisoned"]["count"], 1)

    async def test_server_registry_loss_repairs_once_without_recopy_and_preserves_foreign_region(self):
        transport = self.make_transport()
        await self.infer(transport, large_image(10))
        slot = transport._shm_pool._slots[0]
        self.state.registry.clear()  # 模拟已成功请求后 Triton 重启，丢失注册表。
        self.state.registry["foreign"] = {"name": "foreign", "key": "/foreign"}
        expected = large_image(20).tobytes()

        async def check_registration(kwargs):
            name = region(kwargs["inputs"][0])
            self.assertEqual(slot.handle.data[:len(expected)], expected)
            if name not in self.state.registry:
                raise missing_region_error(name)
            return FakeResult()

        self.state.infer_operation = check_registration
        metrics = InferenceMetrics()
        token = active_inference_metrics.set(metrics)
        try:
            with self.assertLogs("core.infer.transport", level="WARNING") as logs:
                await self.infer(transport, large_image(20), request_id="repaired")
        finally:
            active_inference_metrics.reset(token)
        self.assertIn("re-registered", logs.output[0])
        self.assertEqual(len(self.state.created), 1)
        self.assertEqual(len(self.state.copied), 2)
        self.assertEqual(len(self.state.registered), 2)
        self.assertEqual(self.state.registered[0], self.state.registered[1])
        self.assertEqual(self.state.status_queries, [{"as_json": True, "client_timeout": 2}])
        self.assertIs(self.state.inferred[1]["inputs"], self.state.inferred[2]["inputs"])
        self.assertEqual(self.state.inferred[1]["request_id"], "repaired")
        self.assertEqual(metrics.rpc_attempts, 2)
        self.assertFalse(slot.poisoned)
        counts = [r[0] for r in self.records]
        self.assertEqual(counts.count("triton_shm_reregister_attempt"), 1)
        self.assertEqual(counts.count("triton_shm_reregistered"), 1)
        self.assertNotIn("triton_shm_reregister_failed", counts)
        self.assertNotIn("triton_shm_fallback", counts)
        self.state.infer_operation = None
        await self.infer(transport, large_image(30))
        self.assertEqual(region(self.state.inferred[-1]["inputs"][0]), slot.name)
        await transport.close()
        self.assertEqual(set(self.state.registry), {"foreign"})
        self.assertEqual([x["name"] for x in self.state.unregistered], [slot.name])

    async def test_empty_sdk_status_dict_confirms_missing_region(self):
        transport = self.make_transport()
        await transport.start()

        async def empty_status(kwargs):
            return {}  # protobuf 的空 regions map 在 as_json=True 时省略该字段。

        async def fail_first(kwargs):
            if len(self.state.inferred) == 1:
                raise missing_region_error(region(kwargs["inputs"][0]))
            return FakeResult()

        self.state.status_operation = empty_status
        self.state.infer_operation = fail_first
        await self.infer(transport)
        self.assertEqual(len(self.state.inferred), 2)
        self.assertFalse(transport._shm_pool._slots[0].poisoned)

    async def test_nonmatching_errors_never_query_or_repair_registration(self):
        cases = (
            (grpc.StatusCode.INVALID_ARGUMENT, "Unable to find shared memory region: '{name}'"),
            (grpc.StatusCode.NOT_FOUND, "Unable to find shared memory region: 'foreign'"),
            (grpc.StatusCode.NOT_FOUND, "model not found"),
            (grpc.StatusCode.NOT_FOUND, "Unable to find shared memory region: '{name}' trailing"),
        )
        for status, message in cases:
            with self.subTest(status=status, message=message):
                transport = self.make_transport()

                async def fail(kwargs):
                    name = region(kwargs["inputs"][0])
                    raise InferenceServerException(message.format(name=name), status=str(status))

                self.state.infer_operation = fail
                before = len(self.state.registered)
                with self.assertRaises(TritonInferenceError):
                    await self.infer(transport)
                self.assertEqual(len(self.state.registered), before + 1)
                self.assertTrue(transport._shm_pool._slots[0].poisoned)
                self.assertEqual(self.state.status_queries, [])

    async def test_present_or_malformed_status_declines_repair_and_poisons_slot(self):
        for status in ("present", [], {"regions": []}, {"unexpected": "value"}):
            with self.subTest(status=status):
                transport = self.make_transport()

                async def status_result(kwargs):
                    return {"regions": dict(self.state.registry)} if status == "present" else status

                async def fail(kwargs):
                    raise missing_region_error(region(kwargs["inputs"][0]))

                self.state.status_operation = status_result
                self.state.infer_operation = fail
                registrations = len(self.state.registered)
                inferences = len(self.state.inferred)
                with self.assertRaises(TritonInferenceError):
                    await self.infer(transport)
                self.assertEqual(len(self.state.registered), registrations + 1)
                self.assertEqual(len(self.state.inferred), inferences + 1)
                self.assertTrue(transport._shm_pool._slots[0].poisoned)

    async def test_repair_status_or_registration_failure_preserves_pixels_and_falls_back_next_time(self):
        for failed_stage in ("status", "register", "status_timeout", "register_timeout"):
            with self.subTest(stage=failed_stage):
                transport = self.make_transport()
                self.state.register_operation = self.state.status_operation = None
                await transport.start()
                self.state.registry.clear()

                async def fail_control(kwargs):
                    if failed_stage.endswith("_timeout"):
                        raise InferenceServerException(
                            "repair control failed", status=str(grpc.StatusCode.DEADLINE_EXCEEDED)
                        )
                    raise RuntimeError("repair control failed")

                async def fail_infer(kwargs):
                    raise missing_region_error(region(kwargs["inputs"][0]))

                if failed_stage.startswith("status"):
                    self.state.status_operation = fail_control
                else:
                    self.state.register_operation = fail_control
                self.state.infer_operation = fail_infer
                calls = len(self.state.inferred)
                poisoned = sum(r[0] == "triton_shm_poisoned" for r in self.records)
                expected_error = InferenceServerTimeout if failed_stage.endswith("_timeout") else TritonInferenceError
                with self.assertRaisesRegex(expected_error, "repair control failed"):
                    await self.infer(transport, large_image(10))
                slot = transport._shm_pool._slots[0]
                self.assertTrue(slot.poisoned)
                self.assertEqual(len(self.state.inferred), calls + 1)
                self.assertEqual(sum(r[0] == "triton_shm_poisoned" for r in self.records), poisoned + 1)
                self.state.infer_operation = None
                await self.infer(transport, large_image(20))
                self.assertIsNone(region(self.state.inferred[-1]["inputs"][0]))
                self.assertEqual(slot.handle.data[:large_image().nbytes], large_image(10).tobytes())

    async def test_missing_region_after_one_repair_is_not_repaired_again(self):
        transport = self.make_transport(transient_retries=3)
        await transport.start()
        self.state.registry.clear()

        async def fail(kwargs):
            raise missing_region_error(region(kwargs["inputs"][0]))

        self.state.infer_operation = fail
        with self.assertRaises(TritonInferenceError):
            await self.infer(transport)
        self.assertEqual(len(self.state.registered), 2)
        self.assertEqual(len(self.state.inferred), 2)
        self.assertEqual(len(self.state.status_queries), 1)
        self.assertTrue(transport._shm_pool._slots[0].poisoned)

    async def test_repaired_rpc_unknown_failure_keeps_slot_poisoned_for_raw_transient_retry(self):
        transport = self.make_transport(transient_retries=1)
        await transport.start()
        self.state.registry.clear()

        async def fail_twice(kwargs):
            if len(self.state.inferred) == 1:
                raise missing_region_error(region(kwargs["inputs"][0]))
            if len(self.state.inferred) == 2:
                raise InferenceServerException("unavailable", status=str(grpc.StatusCode.UNAVAILABLE))
            return FakeResult()

        self.state.infer_operation = fail_twice
        metrics = InferenceMetrics()
        token = active_inference_metrics.set(metrics)
        try:
            await self.infer(transport, large_image(10))
        finally:
            active_inference_metrics.reset(token)
        self.assertEqual(metrics.rpc_attempts, 3)
        self.assertEqual(len(self.state.registered), 2)
        self.assertEqual(len(self.state.copied), 1)
        self.assertIsNone(region(self.state.inferred[-1]["inputs"][0]))
        self.assertTrue(transport._shm_pool._slots[0].poisoned)

    async def test_repair_retains_exclusive_slot_while_another_request_falls_back(self):
        transport = self.make_transport()
        await transport.start()
        self.state.registry.clear()
        entered, release = asyncio.Event(), asyncio.Event()

        async def block_register(kwargs):
            entered.set()
            await release.wait()

        async def check_region(kwargs):
            name = region(kwargs["inputs"][0])
            if name is not None and name not in self.state.registry:
                raise missing_region_error(name)
            return FakeResult()

        self.state.register_operation = block_register
        self.state.infer_operation = check_region
        first = asyncio.create_task(self.infer(transport, large_image(10)))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await self.infer(transport, large_image(20))
            self.assertIsNone(region(self.state.inferred[-1]["inputs"][0]))
            self.assertEqual(len(self.state.copied), 1)
            self.assertEqual(self.state.created[0].data[:large_image().nbytes], large_image(10).tobytes())
            release.set()
            await first
            self.assertFalse(transport._shm_pool._slots[0].poisoned)
            self.assertEqual(len(self.state.registered), 2)
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)

    async def test_close_during_repair_query_or_registration_never_resends_inference(self):
        for stage in ("status", "register"):
            with self.subTest(stage=stage):
                self.state.register_operation = self.state.status_operation = None
                transport = self.make_transport()
                await transport.start()
                self.state.registry.clear()
                entered, release = asyncio.Event(), asyncio.Event()

                async def block_control(kwargs):
                    entered.set()
                    await release.wait()
                    return {}

                async def fail(kwargs):
                    raise missing_region_error(region(kwargs["inputs"][0]))

                self.state.infer_operation = fail
                if stage == "status":
                    self.state.status_operation = block_control
                else:
                    self.state.register_operation = block_control
                calls = len(self.state.inferred)
                task = asyncio.create_task(self.infer(transport))
                closer = None
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    closer = asyncio.create_task(transport.close())
                    await wait_until(lambda: transport._shm_pool._closing)
                    self.assertFalse(closer.done())
                    self.assertNotIn(self.state.created[-1], self.state.destroyed)
                    release.set()
                    with self.assertRaises(TritonInferenceError):
                        await task
                    await asyncio.wait_for(closer, 1)
                    self.assertEqual(len(self.state.inferred), calls + 1)
                    self.assertNotIn(self.state.created[-1].name, self.state.registry)
                finally:
                    release.set()
                    await asyncio.gather(*(t for t in (task, closer) if t), return_exceptions=True)

    async def test_repeated_cancel_during_registration_drains_before_close_cleanup(self):
        transport = self.make_transport()
        await transport.start()
        self.state.registry.clear()
        entered, release = asyncio.Event(), asyncio.Event()

        async def block_register(kwargs):
            entered.set()
            await release.wait()

        async def fail(kwargs):
            raise missing_region_error(region(kwargs["inputs"][0]))

        self.state.register_operation = block_register
        self.state.infer_operation = fail
        task = asyncio.create_task(self.infer(transport, large_image(10)))
        closer = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
            closer = asyncio.create_task(transport.close())
            await wait_until(lambda: transport._shm_pool._closing)
            self.assertEqual(self.state.unregistered, [])
            self.assertEqual(self.state.destroyed, [])
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closer, 1)
            self.assertEqual(len(self.state.inferred), 1)
            self.assertEqual(len(self.state.copied), 1)
            self.assertEqual(self.state.registry, {})
            self.assertEqual(self.state.created, self.state.destroyed)
            self.assertEqual(sum(r[0] == "triton_shm_poisoned" for r in self.records), 1)
        finally:
            release.set()
            await asyncio.gather(*(t for t in (task, closer) if t), return_exceptions=True)

    async def test_cancel_during_repair_status_or_retried_rpc_poisons_without_recopy(self):
        for stage in ("status", "retry"):
            with self.subTest(stage=stage):
                self.state.status_operation = None
                transport = self.make_transport()
                await transport.start()
                self.state.registry.clear()
                entered = asyncio.Event()
                calls = len(self.state.inferred)

                async def block_status(kwargs):
                    entered.set()
                    await asyncio.Event().wait()

                async def fail_then_block(kwargs):
                    if len(self.state.inferred) == calls + 1:
                        raise missing_region_error(region(kwargs["inputs"][0]))
                    entered.set()
                    await asyncio.Event().wait()

                if stage == "status":
                    self.state.status_operation = block_status
                self.state.infer_operation = fail_then_block
                copied = len(self.state.copied)
                task = asyncio.create_task(self.infer(transport, large_image(10)))
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                slot = transport._shm_pool._slots[0]
                self.assertTrue(slot.poisoned)
                self.state.infer_operation = None
                await self.infer(transport, large_image(20))
                self.assertEqual(len(self.state.copied), copied + 1)
                self.assertIsNone(region(self.state.inferred[-1]["inputs"][0]))
                self.assertEqual(slot.handle.data[:large_image().nbytes], large_image(10).tobytes())

    async def test_disabled_path_creates_no_shared_memory_or_control_client(self):
        transport = self.make_transport(shm_slots=0, pool_size=1)
        await transport.start()
        await self.infer(transport)
        await transport.close()
        self.assertEqual(len(self.state.clients), 1)
        self.assertEqual(self.state.registered, [])
        self.assertEqual(self.state.created, [])
        self.assertIsNone(transport._shm_pool)
        self.assertIsNone(region(self.state.inferred[0]["inputs"][0]))

    async def test_only_three_large_raw_image_contracts_use_exact_shm_bytes_and_shape(self):
        transport = self.make_transport()
        shapes = ((1, 640, 640, 3), (1, 600, 720, 3), (2, 400, 500, 3))
        for index, model in enumerate(("face_det", "person_seg", "person_pose")):
            array = np.full(shapes[index], index, np.uint8)
            array.setflags(write=False)
            outputs = await self.infer(transport, array, model=model, request_id=model)
            call = self.state.inferred[-1]
            tensor = call["inputs"][0]
            handle = self.state.created[0]
            self.assertEqual(region(tensor), handle.name)
            self.assertEqual(tensor._input.parameters["shared_memory_byte_size"].int64_param, array.nbytes)
            self.assertEqual(tuple(tensor.shape()), array.shape)
            self.assertIsNone(tensor._raw_content)
            self.assertEqual(handle.data[:array.nbytes], array.tobytes())
            self.assertEqual(set(outputs), {t.name for t in MODEL_CONTRACTS[model].outputs})
            self.assertEqual(call["model_name"], MODEL_CONTRACTS[model].model_name)
            self.assertEqual(call["request_id"], model)
            self.assertEqual(call["client_timeout"], 3)
        self.assertEqual(len(self.state.created), 1)
        self.assertEqual(len(self.state.copied), 3)

    async def test_small_oversize_bytes_and_other_contracts_fall_back_without_shm_init(self):
        transport = self.make_transport()
        for array in (np.zeros((1, 8, 8, 3), np.uint8), np.zeros((1, 1024, 1024, 3), np.uint8)):
            await self.infer(transport, array)
        await transport.infer(
            MODEL_CONTRACTS["clip_image"], {"IMAGE_BYTES": np.asarray([[b"image bytes"]], dtype=object)}, request_id="bytes"
        )
        await transport.infer(
            MODEL_CONTRACTS["face_rec"],
            {"RAW_IMAGE": large_image(), "LANDMARK": np.zeros((1, 5, 2), np.float32)}, request_id="other"
        )
        self.assertEqual(self.state.created, [])
        self.assertTrue(all(region(call["inputs"][0]) is None for call in self.state.inferred))
        self.assertEqual(sum(r[0] == "triton_shm_fallback" for r in self.records), 4)

    async def test_wrong_dtype_missing_extra_and_nonarray_preserve_input_error(self):
        transport = self.make_transport()
        for inputs in (
            {}, {"RAW_IMAGE": large_image(), "EXTRA": np.zeros(1)},
            {"RAW_IMAGE": large_image().astype(np.float32)},
            {"RAW_IMAGE": np.full((1, 640, 640, 3), "x", dtype=object)},
            {"RAW_IMAGE": [1, 2, 3]},
        ):
            with self.subTest(keys=list(inputs)):
                with self.assertRaises(InferenceInputError):
                    await transport.infer(MODEL_CONTRACTS["face_det"], inputs, request_id="invalid")
        self.assertEqual(self.state.created, [])
        self.assertEqual(self.state.inferred, [])

    async def test_full_slot_falls_back_immediately_without_overwriting_first_request(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def block_first(kwargs):
            if kwargs["request_id"] == "first":
                entered.set()
                await release.wait()
            return FakeResult()

        self.state.infer_operation = block_first
        transport = self.make_transport()
        first_array = large_image(10)
        first = asyncio.create_task(self.infer(transport, first_array, request_id="first"))
        try:
            await entered.wait()
            await self.infer(transport, large_image(20), request_id="second")
            self.assertIsNotNone(region(self.state.inferred[0]["inputs"][0]))
            self.assertIsNone(region(self.state.inferred[1]["inputs"][0]))
            self.assertEqual(self.state.created[0].data[:first_array.nbytes], first_array.tobytes())
            self.assertEqual(len(self.state.copied), 1)
        finally:
            release.set()
            await first

    async def test_deadline_poisons_slot_and_next_request_uses_raw(self):
        async def fail_first(kwargs):
            if len(self.state.inferred) == 1:
                raise InferenceServerException("deadline", status=str(grpc.StatusCode.DEADLINE_EXCEEDED))
            return FakeResult()

        self.state.infer_operation = fail_first
        transport = self.make_transport()
        with self.assertRaises(InferenceServerTimeout):
            await self.infer(transport, large_image(10))
        await self.infer(transport, large_image(20))
        self.assertIsNone(region(self.state.inferred[1]["inputs"][0]))
        self.assertEqual(len(self.state.created), 1)
        self.assertEqual(len(self.state.copied), 1)
        self.assertEqual(self.state.created[0].data[:large_image().nbytes], large_image(10).tobytes())

    async def test_unavailable_then_success_retries_raw_and_keeps_original_slot_poisoned(self):
        async def fail_first(kwargs):
            if len(self.state.inferred) == 1:
                raise InferenceServerException("retry", status=str(grpc.StatusCode.UNAVAILABLE))
            return FakeResult()

        self.state.infer_operation = fail_first
        transport = self.make_transport(transient_retries=1)
        await self.infer(transport, large_image(10))
        await self.infer(transport, large_image(20))
        self.assertIsNotNone(region(self.state.inferred[0]["inputs"][0]))
        self.assertTrue(all(region(call["inputs"][0]) is None for call in self.state.inferred[1:]))
        self.assertEqual(len(self.state.copied), 1)
        self.assertEqual(len(self.state.created), 1)
        self.assertEqual(sum(r[0] == "triton_shm_poisoned" for r in self.records), 1)

    async def test_rpc_cancel_poisons_slot_and_cleanup_never_rewrites_region(self):
        entered = asyncio.Event()

        async def block_first(kwargs):
            if len(self.state.inferred) == 1:
                entered.set()
                await asyncio.Event().wait()
            return FakeResult()

        self.state.infer_operation = block_first
        transport = self.make_transport()
        first = asyncio.create_task(self.infer(transport, large_image(10)))
        await entered.wait()
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        await self.infer(transport, large_image(20))
        self.assertEqual(len(self.state.copied), 1)
        self.assertIsNone(region(self.state.inferred[-1]["inputs"][0]))
        self.assertEqual(self.state.created[0].data[:large_image().nbytes], large_image(10).tobytes())
        await transport.close()
        self.assertEqual(self.state.created, self.state.destroyed)

    async def test_cancel_before_client_acquire_does_not_poison_or_send_second_rpc(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def block_first(kwargs):
            if kwargs["request_id"] == "small":
                entered.set()
                await release.wait()
            return FakeResult()

        self.state.infer_operation = block_first
        transport = self.make_transport(pool_size=1)
        first = asyncio.create_task(self.infer(transport, np.zeros((1, 8, 8, 3), np.uint8), request_id="small"))
        second = None
        try:
            await entered.wait()
            second = asyncio.create_task(self.infer(transport))
            await wait_until(lambda: len(self.state.copied) == 1)
            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second
            self.assertEqual(len(self.state.inferred), 1)
            self.assertFalse(transport._shm_pool._slots[0].poisoned)
            release.set()
            await first
            await self.infer(transport)
            self.assertIsNotNone(region(self.state.inferred[-1]["inputs"][0]))
            self.assertEqual(len(self.state.copied), 2)
        finally:
            release.set()
            await asyncio.gather(*(t for t in (first, second) if t), return_exceptions=True)

    async def test_output_request_failure_releases_unsent_slot_and_copy_failure_sends_no_rpc(self):
        transport = self.make_transport()
        with patch("core.infer.transport.grpcclient.InferRequestedOutput", side_effect=ValueError("output")):
            with self.assertRaisesRegex(InferenceInputError, "output request construction"):
                await self.infer(transport)
        self.assertEqual(self.state.inferred, [])
        self.assertFalse(transport._shm_pool._slots[0].poisoned)

        def fail_copy(handle, values):
            raise ValueError("copy failed")

        self.state.copy_operation = fail_copy
        with self.assertRaisesRegex(InferenceInputError, "could not be copied"):
            await self.infer(transport)
        self.assertEqual(self.state.inferred, [])
        self.state.copy_operation = None
        await self.infer(transport)
        self.assertIsNotNone(region(self.state.inferred[-1]["inputs"][0]))

    async def test_close_during_unsent_copy_drains_copy_before_destroy_and_sends_no_rpc(self):
        entered, release = threading.Event(), threading.Event()

        def block(handle, values):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release copy")

        self.state.copy_operation = block
        transport = self.make_transport()
        task = asyncio.create_task(self.infer(transport))
        closer = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            closer = asyncio.create_task(transport.close())
            await wait_until(lambda: transport._shm_pool._closing)
            self.assertEqual(self.state.destroyed, [])
            release.set()
            with self.assertRaisesRegex(RuntimeError, "closing|closed"):
                await task
            await asyncio.wait_for(closer, 1)
            self.assertEqual(self.state.inferred, [])
            self.assertEqual(self.state.created, self.state.destroyed)
        finally:
            release.set()
            await asyncio.gather(*(t for t in (task, closer) if t), return_exceptions=True)

    async def test_parameter_validation_rejects_invalid_counts_and_memory(self):
        for kwargs in (
            {"shm_slots": True}, {"shm_slots": -1}, {"shm_slots": 65},
            {"shm_slots": 1.5}, {"shm_slot_bytes": True}, {"shm_slot_bytes": MIB - 1},
            {"shm_slots": 17, "shm_slot_bytes": 32 * MIB},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.make_transport(**kwargs)
        self.assertEqual(self.state.clients, [])


if __name__ == "__main__":
    unittest.main()
