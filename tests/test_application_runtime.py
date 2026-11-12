"""用纯 fake 工厂验证应用资源的所有权与关闭顺序，不连接外部服务。"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from core.application import ApplicationRuntime, OutputNamespacedStorage, RuntimeFactories


def runtime_config(**overrides):
    values = dict(
        ALOG_MAX_INFLIGHT_MESSAGES=0, ALOG_WORKER_COUNT=4,
        ALOG_MAX_INFLIGHT_FRAMES=0, FRAME_MEMORY_BUDGET_BYTES=0,
        FRAME_RESERVATION_BYTES=64, STORAGE_OUTPUT_PREFIX="",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeAssembly:
    def __init__(self, *, failure=None):
        self.events = []
        self.failure = failure
        self.initialize_entered = asyncio.Event()
        self.initialize_gate = None
        self.creation_entered = asyncio.Event()
        self.creation_gate = None
        self.resources_close_entered = asyncio.Event()
        self.resources_close_gate = None
        self.output_entered = asyncio.Event()
        self.output_gate = None
        self.resources = SimpleNamespace(
            vector_write=SimpleNamespace(run=self.run_creation),
            close=self.close_resources, queue_snapshot=lambda: {},
        )
        self.triton = SimpleNamespace(initialize=self.initialize, close=self.close_triton, queue_snapshot=lambda: {})
        self.databases = SimpleNamespace(close=self.close_databases)
        self.storage = SimpleNamespace(get_object=AsyncMock(return_value=b"image"), upload_object=AsyncMock(return_value="stored"))
        self.output = SimpleNamespace(start=self.start_output, stop=self.stop_output,
                                      wait_for_capacity=AsyncMock(), queue_snapshot=lambda: {})
        self.processor = SimpleNamespace(process_message=AsyncMock())
        self.factories = RuntimeFactories(
            resources=lambda config: self.create("resources", self.resources),
            triton=lambda config, resources: self.create("triton", self.triton),
            databases=lambda config, resources: self.create("databases", self.databases),
            storage=lambda config, resources: self.create("storage", self.storage),
            output=lambda config, databases: self.create("output", self.output),
            processor=lambda runtime: self.create("processor", self.processor),
        )

    def create(self, name, value):
        self.events.append(name + ".create")
        if self.failure == name + ".create":
            raise RuntimeError(self.failure)
        return value

    async def initialize(self):
        self.events.append("triton.initialize")
        self.initialize_entered.set()
        if self.initialize_gate is not None:
            await self.initialize_gate.wait()
        if self.failure == "triton.initialize":
            raise RuntimeError(self.failure)

    async def run_creation(self, factory, *args):
        self.creation_entered.set()
        if self.creation_gate is not None:
            await self.creation_gate.wait()
        return factory(*args)

    async def start_output(self):
        self.events.append("output.start")
        self.output_entered.set()
        if self.output_gate is not None:
            await self.output_gate.wait()
        if self.failure == "output.start":
            raise RuntimeError(self.failure)

    async def stop_output(self):
        self.events.append("output.stop")
        if self.failure == "output.stop":
            raise RuntimeError(self.failure)

    async def close_triton(self):
        self.events.append("triton.close")

    async def close_resources(self):
        self.events.append("resources.close.begin")
        self.resources_close_entered.set()
        if self.resources_close_gate is not None:
            await self.resources_close_gate.wait()
        self.events.append("resources.close.end")

    def close_databases(self):
        self.events.append("databases.close")
        if "resources.close.end" not in self.events:
            raise AssertionError("database closed before worker threads drained")


class ApplicationRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_assembly_processes_message_and_closes_in_dependency_order(self):
        fake = FakeAssembly()
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        self.assertEqual(fake.events, [])
        self.assertEqual(runtime.frames.queue_snapshot()["capacity_frames"], 4)
        self.assertIs(await runtime.start(), runtime)
        self.assertIs(await runtime.start(), runtime)
        await runtime.process_message("message")
        fake.processor.process_message.assert_awaited_once_with("message", fake.storage)
        self.assertEqual(set(runtime.metric_providers()), {"resources", "frames", "triton", "output"})
        await runtime.close()
        await runtime.close()
        self.assertEqual(fake.events[-5:], ["output.stop", "triton.close", "resources.close.begin", "resources.close.end", "databases.close"])
        self.assertEqual(fake.events.count("resources.create"), 1)
        self.assertEqual(fake.events.count("databases.close"), 1)
        with self.assertRaisesRegex(RuntimeError, "not started"):
            await runtime.process_message("late")
        with self.assertRaises(RuntimeError):
            await runtime.start()

    async def test_concurrent_start_calls_share_one_resource_assembly(self):
        fake = FakeAssembly()
        fake.initialize_gate = asyncio.Event()
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        first = asyncio.create_task(runtime.start())
        await fake.initialize_entered.wait()
        second = asyncio.create_task(runtime.start())
        await asyncio.sleep(0)
        fake.initialize_gate.set()
        self.assertEqual(await asyncio.gather(first, second), [runtime, runtime])
        self.assertEqual(fake.events.count("resources.create"), 1)
        self.assertEqual(fake.events.count("output.start"), 1)
        await runtime.close()

    async def test_context_preserves_business_exception_after_cleanup_error(self):
        fake = FakeAssembly(failure="output.stop")
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        with self.assertRaisesRegex(ValueError, "business failed"):
            async with runtime:
                raise ValueError("business failed")
        self.assertEqual(fake.events[-1], "databases.close")

    async def test_start_failure_closes_every_resource_already_owned(self):
        for failure in ("triton.initialize", "databases.create", "storage.create", "output.start", "processor.create"):
            with self.subTest(failure=failure):
                fake = FakeAssembly(failure=failure)
                runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
                with self.assertRaisesRegex(RuntimeError, failure):
                    await runtime.start()
                self.assertIn("triton.close", fake.events)
                self.assertIn("resources.close.end", fake.events)
                if failure not in ("triton.initialize", "databases.create"):
                    self.assertIn("databases.close", fake.events)
                if failure in ("output.start", "processor.create"):
                    self.assertIn("output.stop", fake.events)

    async def test_cancelled_initialization_closes_resources(self):
        fake = FakeAssembly()
        fake.initialize_gate = asyncio.Event()
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        task = asyncio.create_task(runtime.start())
        await fake.initialize_entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(fake.events[-3:], ["triton.close", "resources.close.begin", "resources.close.end"])
        self.assertNotIn("databases.create", fake.events)

    async def test_cancelled_database_construction_registers_and_closes_late_result(self):
        fake = FakeAssembly()
        fake.creation_gate = asyncio.Event()
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        task = asyncio.create_task(runtime.start())
        await fake.creation_entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertNotIn("resources.close.begin", fake.events)
        task.cancel()
        fake.creation_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNotNone(runtime._close_task)
        await asyncio.wait_for(asyncio.shield(runtime._close_task), 1)
        self.assertIn("databases.create", fake.events)
        self.assertEqual(fake.events[-1], "databases.close")
        self.assertNotIn("output.start", fake.events)

    async def test_cancelled_output_start_still_stops_output(self):
        fake = FakeAssembly()
        fake.output_gate = asyncio.Event()
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        task = asyncio.create_task(runtime.start())
        await fake.output_entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIn("output.stop", fake.events)
        self.assertEqual(fake.events[-1], "databases.close")

    async def test_cancelled_close_wait_does_not_close_database_before_threads_finish(self):
        fake = FakeAssembly()
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        await runtime.start()
        fake.resources_close_gate = asyncio.Event()
        first = asyncio.create_task(runtime.close())
        await fake.resources_close_entered.wait()
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertNotIn("databases.close", fake.events)
        second = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        fake.resources_close_gate.set()
        await asyncio.wait_for(second, 1)
        self.assertEqual(fake.events.count("databases.close"), 1)

    async def test_output_cleanup_error_does_not_skip_other_resources(self):
        fake = FakeAssembly(failure="output.stop")
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        await runtime.start()
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            await runtime.close()
        self.assertEqual(fake.events[-1], "databases.close")
        self.assertIn("triton.close", fake.events)

    async def test_injected_storage_skips_factory_and_namespaces_only_upload(self):
        fake = FakeAssembly()
        runtime = ApplicationRuntime(runtime_config(STORAGE_OUTPUT_PREFIX="/replay/run/"), factories=fake.factories, storage=fake.storage)
        await runtime.start()
        try:
            self.assertNotIn("storage.create", fake.events)
            self.assertEqual(await runtime.storage.get_object("camera/original.jpg", "thumb"), b"image")
            self.assertEqual(await runtime.storage.upload_object(b"crop", "/camera/crop.jpg", {"type": "image"}), "stored")
            fake.storage.get_object.assert_awaited_once_with("camera/original.jpg", "thumb")
            fake.storage.upload_object.assert_awaited_once_with(b"crop", "replay/run/camera/crop.jpg", {"type": "image"})
        finally:
            await runtime.close()

    async def test_close_during_start_prevents_late_resource_ownership(self):
        fake = FakeAssembly()
        fake.initialize_gate = asyncio.Event()
        runtime = ApplicationRuntime(runtime_config(), factories=fake.factories)
        starting = asyncio.create_task(runtime.start())
        await fake.initialize_entered.wait()
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        fake.initialize_gate.set()
        results = await asyncio.wait_for(asyncio.gather(starting, closing, return_exceptions=True), 1)
        self.assertFalse(runtime._started, results)
        if "databases.create" in fake.events:
            self.assertEqual(fake.events.count("databases.close"), 1)
        if "output.start" in fake.events:
            self.assertEqual(fake.events.count("output.stop"), 1)

    async def test_empty_namespace_preserves_upload_key(self):
        storage = SimpleNamespace(upload_object=AsyncMock(return_value="/key"))
        wrapper = OutputNamespacedStorage(storage, "")
        self.assertEqual(await wrapper.upload_object(b"data", "/key"), "/key")
        storage.upload_object.assert_awaited_once_with(b"data", "/key", None)

    def test_import_is_free_of_external_clients_and_runtime_factories(self):
        script = '''
import importlib.abc
import sys
class DenyExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"aiokafka", "pymilvus", "tritonclient", "oss2", "minio", "db"}:
            raise AssertionError("external dependency imported: " + fullname)
sys.meta_path.insert(0, DenyExternal())
import core.application
import core.worker_service
import core.admission
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
