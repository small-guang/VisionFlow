"""视觉运行时不建立身份/输出资源，并将业务失败交还调度 owner。"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from core.multiprocess.visual import VisualRuntime, VisualRuntimeFactories


class FakeVisualAssembly:
    def __init__(self, failure=None):
        self.failure = failure
        self.events = []
        self.gate = None
        self.entered = asyncio.Event()
        self.resources = SimpleNamespace(close=self.close_resources, queue_snapshot=lambda: {"image_cpu": {}})
        self.triton = SimpleNamespace(initialize=self.initialize, close=self.close_triton, queue_snapshot=lambda: {})
        self.storage = SimpleNamespace(get_object=AsyncMock(), upload_object=AsyncMock(return_value="crop"))
        self.processor = SimpleNamespace(process_payload=AsyncMock())
        self.factories = VisualRuntimeFactories(
            configure_process=lambda config: self.events.append("configure"),
            resources=lambda config: self.create("resources", self.resources),
            triton=lambda config, resources: self.create("triton", self.triton),
            storage=lambda config, resources: self.create("storage", self.storage),
            processor=lambda runtime: self.create("processor", self.processor),
        )

    def create(self, name, value):
        self.events.append(name)
        if self.failure == name:
            raise RuntimeError(name)
        return value

    async def initialize(self):
        self.events.append("initialize")
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.failure == "initialize":
            raise RuntimeError("initialize")

    async def close_triton(self):
        self.events.append("triton.close")

    async def close_resources(self):
        self.events.append("resources.close")

    def runtime(self, prefix=""):
        self.peer = SimpleNamespace(call=AsyncMock())
        return VisualRuntime(SimpleNamespace(STORAGE_OUTPUT_PREFIX=prefix), self.peer,
                             factories=self.factories, observability=False)


class VisualRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_child_metrics_use_separate_files_and_close_background_tasks(self):
        from core.performance import RuntimePerformance
        from core.queue_metrics import PipelineMetrics

        fake = FakeVisualAssembly()
        performance, pipeline = RuntimePerformance(), PipelineMetrics()
        with TemporaryDirectory() as directory, \
             patch("core.performance.runtime_performance", performance), \
             patch("core.queue_metrics.pipeline_metrics", pipeline):
            config = SimpleNamespace(
                STORAGE_OUTPUT_PREFIX="", MP_WORKER_INDEX=1,
                PIPELINE_STATS_PATH=str(Path(directory) / "pipeline_stats.jsonl"),
                PIPELINE_STATS_ENABLED=True, PIPELINE_STATS_INTERVAL_SECONDS=30,
            )
            runtime = VisualRuntime(config, SimpleNamespace(call=AsyncMock()), factories=fake.factories)
            await runtime.start()
            await runtime.process_message({"value": "{}", "topic": "frames"})
            await runtime.close()
            self.assertEqual(performance._tasks, [])
            self.assertIsNone(pipeline._task)
            self.assertEqual(pipeline._providers, {})
            self.assertTrue((Path(directory) / "visual-2.log").exists())
            self.assertTrue((Path(directory) / "pipeline_stats.visual-2.jsonl").exists())
            self.assertFalse((Path(directory) / "pipeline_stats.jsonl").exists())

    async def test_default_processor_assembly_injects_remote_ports_and_strict_errors(self):
        from core.multiprocess.assembly import make_visual_config
        from core.multiprocess.visual import _processor
        from tests.test_multiprocess_resources import settings

        config = make_visual_config(settings(VISUAL_WORKER_PROCESSES=2), 0)
        runtime = SimpleNamespace(
            config=config, peer=SimpleNamespace(call=AsyncMock()),
            triton=SimpleNamespace(), resources=SimpleNamespace(image_cpu=SimpleNamespace()),
            frames=SimpleNamespace(), image_completed=AsyncMock(),
        )
        with patch("core.keyframe_processor.FaceIdentityService", side_effect=AssertionError("local identity")), \
             patch("core.keyframe_processor.PersonReidSearchService", side_effect=AssertionError("local gallery")):
            processor = _processor(runtime)
        self.assertTrue(processor.strict_errors)
        self.assertIs(processor.image_completed_callback, runtime.image_completed)
        self.assertIs(processor.frame_budget, runtime.frames)
        self.assertIsNone(processor.face_embedding_db)
        self.assertIsNone(processor.person_sbs_embedding_db)

    async def test_spawn_configuration_assigns_distinct_ids_at_the_same_millisecond(self):
        from core.multiprocess.visual import _configure_process
        from utils.identifiers.id_generator import SnowflakeGenerator

        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                ids = []
                for index in range(-1, count):
                    generator = SnowflakeGenerator(datacenter_id=0, worker_id=0)
                    with patch.object(generator, "_current_millis", return_value=1700000000000):
                        if index >= 0:
                            with patch("utils.identifiers.id_generator.id_generator", generator), \
                                 patch("cv2.setNumThreads") as threads, patch("loguru.logger.remove"), patch("loguru.logger.add"):
                                _configure_process(SimpleNamespace(MP_WORKER_INDEX=index, MP_WORKER_COUNT=count))
                            threads.assert_called_once_with(1)
                        self.assertEqual(generator.worker_id, index + 1)
                        ids.append(generator.generate())
                self.assertEqual(len(set(ids)), count + 1)

    async def test_spawn_rejects_invalid_worker_identity_before_process_side_effects(self):
        from core.multiprocess.visual import _configure_process

        for index, count in ((0, 0), (0, 5), (-1, 4), (4, 4), (True, 2), (0, True)):
            with self.subTest(index=index, count=count), patch("cv2.setNumThreads") as threads, \
                 patch("loguru.logger.remove") as remove:
                with self.assertRaisesRegex(ValueError, "worker identity"):
                    _configure_process(SimpleNamespace(MP_WORKER_INDEX=index, MP_WORKER_COUNT=count))
                threads.assert_not_called()
                remove.assert_not_called()

    async def test_start_process_and_close_own_only_visual_resources(self):
        fake = FakeVisualAssembly()
        runtime = fake.runtime()
        self.assertIs(await runtime.start(), runtime)
        await runtime.start()
        result = await runtime.process_message({"value": b'{"uId":"one"}', "topic": "frames", "offset": 7})
        self.assertEqual(result, {"topic": "frames", "message_completed": 1})
        fake.processor.process_payload.assert_awaited_once_with(
            {"uId": "one"}, oss_client=fake.storage, topic="frames", offset=7,
        )
        self.assertEqual(set(runtime.metrics()), {"resources", "frames", "triton"})
        self.assertFalse(hasattr(runtime, "databases"))
        self.assertFalse(hasattr(runtime, "output"))
        await runtime.close()
        await runtime.close()
        self.assertEqual(fake.events[-2:], ["triton.close", "resources.close"])
        self.assertEqual(fake.events.count("resources"), 1)
        with self.assertRaises(RuntimeError):
            await runtime.process_message({"value": "{}"})
        with self.assertRaises(RuntimeError):
            await runtime.start()

    async def test_business_and_invalid_payload_errors_propagate(self):
        fake = FakeVisualAssembly()
        runtime = fake.runtime()
        await runtime.start()
        failure = ValueError("identity unavailable")
        fake.processor.process_payload.side_effect = failure
        try:
            with self.assertRaises(ValueError) as raised:
                await runtime.process_message(SimpleNamespace(value="{}"))
            self.assertIs(raised.exception, failure)
            for value in (b"invalid json", b"[]"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    await runtime.process_message({"value": value})
            self.assertEqual(runtime._active_tasks, set())
        finally:
            await runtime.close()

    async def test_start_failure_or_cancel_closes_registered_resources(self):
        for failure in ("initialize", "storage", "processor"):
            with self.subTest(failure=failure):
                fake = FakeVisualAssembly(failure)
                with self.assertRaisesRegex(RuntimeError, failure):
                    await fake.runtime().start()
                self.assertEqual(fake.events[-2:], ["triton.close", "resources.close"])
        fake = FakeVisualAssembly()
        fake.gate = asyncio.Event()
        runtime = fake.runtime()
        task = asyncio.create_task(runtime.start())
        await fake.entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(fake.events[-2:], ["triton.close", "resources.close"])

    async def test_close_waits_for_inflight_frame_before_closing_resources(self):
        fake = FakeVisualAssembly()
        entered, release = asyncio.Event(), asyncio.Event()

        async def processing(*args, **kwargs):
            entered.set()
            await release.wait()

        fake.processor.process_payload.side_effect = processing
        runtime = fake.runtime()
        await runtime.start()
        task = asyncio.create_task(runtime.process_message({"value": "{}"}))
        await entered.wait()
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertNotIn("triton.close", fake.events)
        release.set()
        await task
        await closing
        self.assertEqual(fake.events[-2:], ["triton.close", "resources.close"])

    async def test_completion_rpc_and_namespaced_upload_are_explicit(self):
        fake = FakeVisualAssembly()
        runtime = fake.runtime("replay/run")
        await runtime.start()
        try:
            await runtime.image_completed("topic", "uid", True)
            fake.peer.call.assert_awaited_once_with(
                "frame.complete", {"topic": "topic", "uid": "uid", "is_pano": True},
            )
            await runtime.storage.upload_object(b"crop", "face/one.jpg")
            fake.storage.upload_object.assert_awaited_once_with(b"crop", "replay/run/face/one.jpg", None)
        finally:
            await runtime.close()


if __name__ == "__main__":
    unittest.main()
