from __future__ import annotations

import importlib
import sys
import types
import unittest


_MISSING = object()


class _MainImport:
    module_names = (
        "main",
        "uvloop",
        "aiokafka",
        "aiokafka.structs",
        "loguru",
        "sqlalchemy",
        "sqlalchemy.future",
        "core.config",
        "core.infer",
        "core.keyframe_processor",
        "db.pgsql.cam_device",
        "db.zilliz",
        "utils.buffers.buffer",
        "utils.keyframe.detection_helpers",
        "utils.storage.oss_util",
    )

    def __enter__(self):
        self.originals = {
            name: sys.modules.get(name, _MISSING) for name in self.module_names
        }
        for name in self.module_names:
            sys.modules.pop(name, None)

        uvloop = types.ModuleType("uvloop")
        uvloop.install = lambda: None
        sys.modules["uvloop"] = uvloop

        aiokafka = types.ModuleType("aiokafka")
        aiokafka.AIOKafkaConsumer = object
        structs = types.ModuleType("aiokafka.structs")
        structs.ConsumerRecord = object
        sys.modules["aiokafka"] = aiokafka
        sys.modules["aiokafka.structs"] = structs

        class Logger:
            def __getattr__(self, _name):
                return lambda *_args, **_kwargs: None

        loguru = types.ModuleType("loguru")
        loguru.logger = Logger()
        sys.modules["loguru"] = loguru

        sqlalchemy = types.ModuleType("sqlalchemy")
        future = types.ModuleType("sqlalchemy.future")
        future.select = lambda *_args: object()
        sys.modules["sqlalchemy"] = sqlalchemy
        sys.modules["sqlalchemy.future"] = future

        settings = types.SimpleNamespace(
            LOG_LEVEL="INFO",
            KAFKA_GROUP_ID="group",
            KAFKA_AUTO_OFFSET_RESET="earliest",
            kafka_client_config={},
        )
        config = types.ModuleType("core.config")
        config.settings = settings
        sys.modules["core.config"] = config

        infer = types.ModuleType("core.infer")
        infer.UnifiedTritonClient = lambda: object()
        sys.modules["core.infer"] = infer

        keyframe = types.ModuleType("core.keyframe_processor")
        keyframe.KeyframeProcessor = lambda **_kwargs: object()
        sys.modules["core.keyframe_processor"] = keyframe

        cam_device = types.ModuleType("db.pgsql.cam_device")
        cam_device.SessionLocal = object
        sys.modules["db.pgsql.cam_device"] = cam_device

        zilliz = types.ModuleType("db.zilliz")
        zilliz.face_embedding_db = object()
        zilliz.person_sbs_embedding_db = object()
        sys.modules["db.zilliz"] = zilliz

        buffer = types.ModuleType("utils.buffers.buffer")
        buffer.feature_buffer = object()
        sys.modules["utils.buffers.buffer"] = buffer

        helpers = types.ModuleType("utils.keyframe.detection_helpers")
        helpers.parse_keyframe_context = lambda _data: None
        sys.modules["utils.keyframe.detection_helpers"] = helpers

        oss = types.ModuleType("utils.storage.oss_util")
        oss.AsyncOSSClient = object
        sys.modules["utils.storage.oss_util"] = oss
        return importlib.import_module("main")

    def __exit__(self, *_exc):
        for name, module in self.originals.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class RunWithTritonTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_failure_is_propagated_after_client_close(self):
        events: list[str] = []
        failure = RuntimeError("service failed")

        class Client:
            async def initialize(self):
                events.append("initialize")

            async def close(self):
                events.append("close")

        async def runner():
            events.append("runner")
            raise failure

        with _MainImport() as main:
            with self.assertRaises(RuntimeError) as raised:
                await main.run_with_triton(Client(), runner)

        self.assertIs(raised.exception, failure)
        self.assertEqual(events, ["initialize", "runner", "close"])

    async def test_runner_error_is_not_masked_by_close_error(self):
        runner_failure = RuntimeError("runner failed")

        class Client:
            async def initialize(self):
                pass

            async def close(self):
                raise RuntimeError("close failed")

        async def runner():
            raise runner_failure

        with _MainImport() as main:
            with self.assertRaises(RuntimeError) as raised:
                await main.run_with_triton(Client(), runner)

        self.assertIs(raised.exception, runner_failure)

    async def test_success_value_is_returned_after_client_close(self):
        events: list[str] = []

        class Client:
            async def initialize(self):
                events.append("initialize")

            async def close(self):
                events.append("close")

        async def runner():
            events.append("runner")
            return "finished"

        with _MainImport() as main:
            result = await main.run_with_triton(Client(), runner)

        self.assertEqual(result, "finished")
        self.assertEqual(events, ["initialize", "runner", "close"])


if __name__ == "__main__":
    unittest.main()
