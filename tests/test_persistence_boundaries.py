"""持久化模块边界、身份回执与独立 pending 视图的行为验证。"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from utils.buffers.pending import IdentityRegistry
from utils.buffers.persistence_ticket import PersistenceTicket
from utils.buffers.kafka_publisher import KafkaPublisher
from tests import test_milvus_records as fixtures


class PersistenceTicketTests(unittest.IsolatedAsyncioTestCase):
    async def test_record_waits_for_durable_receipt_and_releases_once(self):
        records = []

        async def accept(record):
            records.append(record)

        ticket = PersistenceTicket()
        await ticket.attach({"id": "birth"}, accept)
        self.assertEqual(records, [])
        await asyncio.gather(ticket.mark_persisted(), ticket.mark_persisted())
        await ticket.attach({"id": "replacement"}, accept)
        self.assertEqual(records, [{"id": "birth"}])
        self.assertTrue(ticket.released)

    async def test_receipt_can_precede_record_attachment(self):
        records = []

        async def accept(record):
            records.append(record)

        ticket = PersistenceTicket()
        await ticket.mark_persisted()
        self.assertFalse(ticket.released)
        await ticket.attach({"id": "late"}, accept)
        self.assertEqual(records, [{"id": "late"}])

    async def test_failed_acceptance_keeps_receipt_for_retry(self):
        attempts = 0
        accepted = []

        async def accept(record):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("acceptance failed")
            accepted.append(record)

        ticket = PersistenceTicket(record={"id": "retry"}, on_ready=accept)
        with self.assertRaisesRegex(RuntimeError, "acceptance failed"):
            await ticket.mark_persisted()
        self.assertTrue(ticket.persisted)
        self.assertFalse(ticket.released)
        await ticket.mark_persisted()
        await ticket.mark_persisted()
        self.assertEqual(attempts, 2)
        self.assertEqual(accepted, [{"id": "retry"}])

    async def test_cancelled_acceptance_does_not_mark_unaccepted_record_released(self):
        entered = asyncio.Event()
        allow = asyncio.Event()
        accepted = []

        async def accept(record):
            entered.set()
            await allow.wait()
            accepted.append(record)

        ticket = PersistenceTicket(record={"id": "cancel"}, on_ready=accept)
        task = asyncio.create_task(ticket.mark_persisted())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(ticket.persisted)
        self.assertFalse(ticket.released)
        self.assertEqual(accepted, [])
        allow.set()
        await ticket.mark_persisted()
        self.assertEqual(accepted, [{"id": "cancel"}])


class PendingIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_runtime_has_independent_face_and_sbs_versions(self):
        first, second = IdentityRegistry(), IdentityRegistry()
        face = {"faceId": "one", "sampleId": "sample", "embedding": [1.0, 0.0]}
        prototype = {"reId": "one", "embedding": [1.0, 0.0]}
        await first.faces.add(face)
        await first.sbs.add(prototype)
        self.assertEqual(await second.faces.snapshot(), [])
        self.assertEqual(await second.sbs.snapshot(), [])
        await first.faces.remove([face])
        self.assertEqual(first.faces.removal_revision(), 1)
        self.assertEqual(first.sbs.removal_revision(), 0)
        self.assertEqual(second.faces.removal_revision(), 0)
        self.assertEqual(await first.sbs.snapshot(), [prototype])

    async def test_pending_face_search_does_not_need_persistence_or_transport(self):
        registry = IdentityRegistry()
        await registry.faces.add({
            "faceId": "known", "sampleId": "sample", "embedding": [1.0, 0.0],
            "deviceId": "camera", "faceCropPath": "crop.jpg",
        })
        samples = await registry.faces.asearch_pending_face_samples([1.0, 0.0], ["camera"])
        self.assertEqual([(s.face_id, s.score) for s in samples], [("known", 1.0)])
        self.assertEqual(await registry.faces.asearch_pending_face_samples([1.0, 0.0], ["other"]), [])


class OutputStartupCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = fixtures.FeatureBufferRoutingTests()
        self.fixture.setUp()
        self.buffers = []

    async def asyncTearDown(self):
        for buffer in self.buffers:
            await buffer.stop()

    def tearDown(self):
        self.fixture.tearDown()

    def make_buffer(self, producer):
        buffer = self.fixture.buffer_module.FeatureBuffer(
            face_db=self.fixture.face_db, picture_db=self.fixture.picture_db,
            sbs_db=self.fixture.sbs_db, tbps_db=self.fixture.person_db,
            producer_factory=lambda: producer, flush_interval=3600,
        )
        self.buffers.append(buffer)
        return buffer

    def assert_stopped(self, buffer, producer):
        producer.stop.assert_awaited_once()
        self.assertIsNone(buffer.records.producer)
        self.assertFalse(buffer._running)
        self.assertFalse(buffer._background_enabled)
        self.assertIsNone(buffer.records.request_flush)
        self.assertTrue(all(writer._task is None for writer in buffer._writers.values()))
        self.assertTrue(all(owner.request_flush is None for owner in buffer.output.vector_writers.values()))

    async def test_publisher_registers_partially_started_producer_for_owner_cleanup(self):
        producer = SimpleNamespace(start=AsyncMock(side_effect=RuntimeError("partial connection")), stop=AsyncMock())
        publisher = KafkaPublisher(topics={"output": "test-output"}, producer_factory=lambda: producer, max_batch_size=1)
        with self.assertRaisesRegex(RuntimeError, "partial connection"):
            await publisher.start()
        self.assertIs(publisher.producer, producer)
        await publisher.stop()
        await publisher.stop()
        producer.stop.assert_awaited_once()

    async def test_facade_start_failure_closes_producer_without_explicit_stop(self):
        producer = SimpleNamespace(start=AsyncMock(side_effect=RuntimeError("partial connection")), stop=AsyncMock())
        buffer = self.make_buffer(producer)
        with self.assertRaisesRegex(RuntimeError, "partial connection"):
            await buffer.start()
        self.assert_stopped(buffer, producer)

    async def test_cancelled_start_closes_partially_connected_producer(self):
        entered = asyncio.Event()

        async def start():
            entered.set()
            await asyncio.Event().wait()

        producer = SimpleNamespace(start=start, stop=AsyncMock())
        buffer = self.make_buffer(producer)
        startup = asyncio.create_task(buffer.start())
        await entered.wait()
        startup.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(startup, 1)
        self.assert_stopped(buffer, producer)

    async def test_partial_background_start_failure_reclaims_started_tasks(self):
        producer = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        buffer = self.make_buffer(producer)
        writers = list(buffer._writers.values())
        with patch.object(writers[1], "start", side_effect=RuntimeError("writer startup")):
            with self.assertRaisesRegex(RuntimeError, "writer startup"):
                await buffer.start()
        self.assert_stopped(buffer, producer)

    async def test_repeated_start_cancellation_does_not_cancel_owned_cleanup(self):
        started = asyncio.Event()
        stopping = asyncio.Event()
        allow_stop = asyncio.Event()

        async def start():
            started.set()
            await asyncio.Event().wait()

        async def stop():
            stopping.set()
            await allow_stop.wait()

        producer = SimpleNamespace(start=start, stop=AsyncMock(side_effect=stop))
        buffer = self.make_buffer(producer)
        startup = asyncio.create_task(buffer.start())
        await started.wait()
        startup.cancel()
        await stopping.wait()
        startup.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await startup
        self.assertFalse(buffer._stop_task.done())
        allow_stop.set()
        await asyncio.wait_for(asyncio.shield(buffer._stop_task), 1)
        self.assert_stopped(buffer, producer)


class ExplicitPersistenceAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FeatureBufferRoutingTests()
        self.fixture.setUp()
        self.module = self.fixture.buffer_module

    def tearDown(self):
        self.fixture.tearDown()

    def test_explicit_dependencies_do_not_resolve_legacy_database_globals(self):
        registry = IdentityRegistry()
        with patch.object(self.module.importlib, "import_module", side_effect=AssertionError("implicit DB")):
            buffer = self.module.FeatureBuffer(
                face_db=self.fixture.face_db, picture_db=self.fixture.picture_db,
                sbs_db=self.fixture.sbs_db, tbps_db=self.fixture.person_db,
                identity_registry=registry,
            )
        self.assertIs(buffer.face_pending_view, registry.faces)
        self.assertIs(buffer.sbs_pending_view, registry.sbs)
        self.assertIs(buffer.face_features.pending, registry.faces)
        self.assertIs(buffer.sbs.pending_index, registry.sbs)
        self.assertIs(buffer.output.publisher, buffer.records)
        self.assertIs(buffer.face_features.vector_db, self.fixture.face_db)
        self.assertIs(buffer.picture_features.vector_db, self.fixture.picture_db)
        self.assertFalse(buffer.queue_snapshot()["background_running"])

    def test_module_import_never_loads_database_package(self):
        script = '''
import importlib.abc
import sys
import types
class RejectDatabase(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "db" or fullname.startswith("db."):
            raise AssertionError("database import during persistence module import")
sys.meta_path.insert(0, RejectDatabase())
config = types.ModuleType("core.config")
config.settings = types.SimpleNamespace()
sys.modules["core.config"] = config
from utils.buffers.buffer import feature_buffer
assert feature_buffer._instance is None
assert not any(name == "db" or name.startswith("db.") for name in sys.modules)
'''
        result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_global_binding_preserves_narrow_port_identity(self):
        buffer = self.module.FeatureBuffer()
        self.module.bind_default_feature_buffer(buffer)
        self.assertIs(self.module.feature_buffer.records, buffer.records)
        self.assertIs(self.module.feature_buffer.face_pending_view, buffer.face_pending_view)
        self.module.bind_default_feature_buffer(None)
        self.assertIsNone(self.module.feature_buffer._instance)
