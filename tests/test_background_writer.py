import asyncio
import json
import threading
import unittest
from unittest.mock import AsyncMock

from tests import test_milvus_records as fixtures


class RecordingProducer:
    def __init__(self):
        self.sent = []
        self.attempted = []
        self.entered = asyncio.Event()
        self.gate = None
        self.fail_id = None
        self.fail_always = False
        self.stop = AsyncMock()

    async def send_and_wait(self, topic, payload):
        record = json.loads(payload)
        self.attempted.append((topic, record))
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_always or (self.fail_id is not None and record.get("id") == self.fail_id):
            raise RuntimeError("test Kafka unavailable")
        self.sent.append((topic, record))


class BackgroundWriterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = fixtures.FeatureBufferRoutingTests()
        self.fixture.setUp()
        self.module = self.fixture.buffer_module
        self.buffers = []
        self.release_events = []
        self.thread_release_events = []

    async def asyncTearDown(self):
        for event in self.release_events:
            event.set()
        for event in self.thread_release_events:
            event.set()
        for buffer in self.buffers:
            await buffer.stop()

    def tearDown(self):
        self.fixture.tearDown()

    async def make_buffer(self, **kwargs):
        buffer = self.module.FeatureBuffer(flush_interval=3600, **kwargs)
        producer = RecordingProducer()
        buffer._kafka_producer = producer
        await buffer.start()
        self.buffers.append(buffer)
        return buffer, producer

    def gate(self):
        event = asyncio.Event()
        self.release_events.append(event)
        return event

    async def eventually(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0)
        await asyncio.wait_for(wait(), 1)

    async def add_face(self, buffer, **overrides):
        kwargs = dict(
            face_id="face-1", embedding=[0.1, 0.2], device_id="CAM001",
            keyframe_id="frame-1", picture_vector_id=42,
            keyframe_url="CAM001/picture/k.jpg",
            face_rect={"x": 1, "y": 2, "w": 30, "h": 60},
            face_key_points=[], face_crop_path="CAM001/face_crop/f.jpg",
            timestamp=171, ai_id="ai-1", quality_score=8.25,
        )
        kwargs.update(overrides)
        await buffer.add_face_feature(**kwargs)

    async def add_picture(self, buffer, picture_id=42):
        await buffer.add_picture_feature(
            keyframe_id="frame-1", picture_vector_id=picture_id,
            embedding=[0.1, 0.2], device_id="CAM001",
            image_path="CAM001/picture/k.jpg", timestamp=171,
        )

    async def test_face_admission_does_not_wait_for_milvus_or_kafka(self):
        buffer, producer = await self.make_buffer(max_batch_size=1)
        db_entered = asyncio.Event()
        db_gate = self.gate()
        producer.gate = self.gate()
        original_insert = self.fixture.face_db.ainsert_embeddings

        async def blocked_insert(**kwargs):
            db_entered.set()
            await db_gate.wait()
            return await original_insert(**kwargs)

        self.fixture.face_db.ainsert_embeddings = blocked_insert
        await asyncio.wait_for(self.add_face(
            buffer, new_face_recognition_record={"id": "new-face"},
        ), 0.5)
        self.assertEqual(len(buffer.pending_faces), 1)
        await asyncio.wait_for(db_entered.wait(), 1)
        self.assertEqual(buffer.queue_snapshot()["queues"]["face"]["inflight"], 1)
        self.assertEqual(producer.attempted, [])

        db_gate.set()
        await asyncio.wait_for(buffer.face_buffer.join(), 1)
        await asyncio.wait_for(producer.entered.wait(), 1)
        self.assertEqual(producer.sent, [])
        self.assertEqual(buffer.pending_faces, {})
        self.assertEqual(buffer.queue_snapshot()["queues"]["face"]["completed_total"], 1)
        producer.gate.set()
        await buffer.flush_all()
        self.assertIn(("face-recognition", {"id": "new-face"}), producer.sent)

    async def test_sbs_retry_keeps_pending_uuid_and_new_identity_release(self):
        buffer, producer = await self.make_buffer(max_batch_size=1)
        calls = []
        fail = True
        original_apply = self.fixture.sbs_db.aapply_prototype_updates

        async def apply(batch):
            calls.append(batch[0]["prototypeId"])
            if fail:
                raise RuntimeError("test Milvus unavailable")
            return await original_apply(batch)

        self.fixture.sbs_db.aapply_prototype_updates = apply
        outcome = await buffer.sbs.offer_feature(
            re_id="reid-1", identity_state="ACTIVE", body_completeness=fixtures.BODY_FULL,
            embedding=[0.1, 0.2], device_id="CAM001", keyframe_id="frame-1",
            picture_vector_id=42, keyframe_url="CAM001/picture/k.jpg",
            person_rect={"x": 1, "y": 2, "w": 30, "h": 60}, timestamp=171,
            ai_id="ai-1", quality_score=8.0, crop_url="CAM001/person_crop/p.jpg",
            observed_face_id="face-1", defer_new_reid_recognition=True,
        )
        await buffer.sbs.attach_new_reid_recognition(
            outcome.recognition_release, {"id": "new-reid"},
            buffer.add_person_recognition_record,
        )
        await self.eventually(lambda: buffer.sbs.queue.retried_total == 1)
        await asyncio.sleep(0.02)
        self.assertEqual(len(calls), 1, "failed flush must not spin")
        self.assertEqual(len(buffer.sbs.pending), 1)
        self.assertFalse(outcome.recognition_release.persisted)
        self.assertFalse(outcome.recognition_release.released)
        self.assertEqual(producer.sent, [])
        fail = False
        buffer._writers["sbs"].request()
        await asyncio.wait_for(buffer.sbs.queue.join(), 1)
        await buffer.flush_all()
        self.assertEqual(calls, [calls[0], calls[0]])
        self.assertTrue(outcome.recognition_release.released)
        self.assertEqual(buffer.sbs.pending, {})
        records = [record for topic, record in producer.sent if topic == "person-recognition"]
        self.assertEqual(records, [{"id": "new-reid"}])

    async def test_explicit_flush_and_background_publisher_have_one_owner(self):
        buffer, producer = await self.make_buffer(max_batch_size=2)
        producer.gate = self.gate()
        await buffer.add_face_detection_record({"id": 1})
        await buffer.add_face_detection_record({"id": 2})
        await asyncio.wait_for(producer.entered.wait(), 1)
        explicit = asyncio.create_task(buffer._flush_face_detection_records())
        await buffer.add_face_detection_record({"id": 3})
        await buffer.add_face_detection_record({"id": 4})
        await asyncio.sleep(0)
        self.assertEqual(len(producer.attempted), 1)
        self.assertFalse(explicit.done())
        producer.gate.set()
        await asyncio.wait_for(explicit, 1)
        await asyncio.wait_for(buffer.face_detection_record_buffer.join(), 1)
        self.assertEqual([record["id"] for _, record in producer.sent], [1, 2, 3, 4])
        self.assertEqual(buffer.queue_snapshot()["queues"]["face_detection"]["acked_total"], 4)

    async def test_partial_kafka_failure_requeues_only_unacknowledged_records(self):
        buffer, producer = await self.make_buffer(max_batch_size=3)
        producer.fail_id = 2
        for index in range(1, 4):
            await buffer.add_face_detection_record({"id": index})
        queue = buffer.face_detection_record_buffer
        await self.eventually(lambda: queue.retried_total == 2)
        await asyncio.sleep(0.02)
        self.assertEqual(len(producer.attempted), 2)
        self.assertEqual(queue.snapshot()["acked_total"], 1)
        self.assertEqual(queue.snapshot()["retry_queued"], 2)
        self.assertEqual(queue.snapshot()["inflight"], 0)
        producer.fail_id = None
        await buffer.flush_all()
        self.assertEqual([record["id"] for _, record in producer.sent], [1, 2, 3])
        self.assertEqual(queue.snapshot()["acked_total"], 3)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 0)

    async def test_timer_flushes_a_batch_below_threshold(self):
        buffer = self.module.FeatureBuffer(max_batch_size=10, flush_interval=0.01)
        producer = RecordingProducer()
        buffer._kafka_producer = producer
        await buffer.start()
        self.buffers.append(buffer)
        await buffer.add_face_detection_record({"id": 1})
        await asyncio.wait_for(buffer.face_detection_record_buffer.join(), 1)
        self.assertEqual(producer.sent, [("face-detection", {"id": 1})])

    async def test_stop_drains_all_batches_and_generated_vector_records(self):
        buffer, producer = await self.make_buffer(max_batch_size=2)
        # These admissions do not yield to I/O. Stop has to drain more than one
        # batch itself, including the Kafka records generated by vector writes.
        for index in range(7):
            await self.add_picture(buffer, index)
            await buffer.add_face_detection_record({"id": index})
        await asyncio.wait_for(buffer.stop(), 1)
        self.buffers.remove(buffer)
        self.assertEqual(sum(len(call["keyframeId"]) for call in self.fixture.picture_db.calls), 7)
        self.assertEqual(len(producer.sent), 14)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 0)
        self.assertEqual(buffer.queue_snapshot()["queues"]["vector_record"]["acked_total"], 7)
        producer.stop.assert_awaited_once()

    async def test_backpressure_counts_inflight_and_uses_low_watermark(self):
        buffer, producer = await self.make_buffer(
            max_batch_size=1, backlog_high_watermark=3, backlog_low_watermark=1,
        )
        producer.gate = self.gate()
        for index in range(3):
            await buffer.add_face_detection_record({"id": index})
        await producer.entered.wait()
        waiter = asyncio.create_task(buffer.wait_for_capacity())
        await asyncio.sleep(0)
        snapshot = buffer.queue_snapshot()
        self.assertEqual(snapshot["total_outstanding"], 3)
        self.assertEqual(snapshot["queues"]["face_detection"]["inflight"], 1)
        self.assertFalse(waiter.done())
        # Complete one record, and make the next write wait again.
        first_gate = producer.gate
        second_gate = self.gate()
        producer.gate = second_gate
        first_gate.set()
        await self.eventually(lambda: len(producer.sent) == 1)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 2)
        self.assertFalse(waiter.done(), "hysteresis must wait until low watermark")
        third_gate = self.gate()
        producer.gate = third_gate
        second_gate.set()
        await asyncio.wait_for(waiter, 1)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 1)
        self.assertEqual(buffer.queue_snapshot()["capacity_waiters"], 0)
        third_gate.set()

    async def test_cancelled_admission_wait_releases_waiter_accounting(self):
        buffer, producer = await self.make_buffer(
            max_batch_size=1, backlog_high_watermark=1, backlog_low_watermark=0,
        )
        producer.gate = self.gate()
        await buffer.add_face_detection_record({"id": 1})
        waiter = asyncio.create_task(buffer.wait_for_capacity())
        await asyncio.sleep(0)
        self.assertEqual(buffer.queue_snapshot()["capacity_waiters"], 1)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(buffer.queue_snapshot()["capacity_waiters"], 0)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 1)

    async def test_stop_retains_failed_items_without_retry_spin(self):
        buffer, producer = await self.make_buffer(max_batch_size=1)
        producer.fail_always = True
        for index in range(3):
            await buffer.add_face_detection_record({"id": index})
        await asyncio.wait_for(buffer.stop(), 1)
        self.buffers.remove(buffer)
        snapshot = buffer.queue_snapshot()
        self.assertEqual(snapshot["total_outstanding"], 3)
        self.assertEqual(snapshot["queues"]["face_detection"]["inflight"], 0)
        self.assertEqual(snapshot["queues"]["face_detection"]["acked_total"], 0)
        # At most one pre-stop background attempt plus one final drain attempt.
        self.assertLessEqual(len(producer.attempted), 2)

    async def test_explicit_flush_all_joins_an_inflight_background_batch(self):
        buffer, producer = await self.make_buffer(max_batch_size=1)
        producer.gate = self.gate()
        await buffer.add_face_detection_record({"id": 1})
        await producer.entered.wait()
        self.assertTrue(buffer.face_detection_record_buffer.empty())
        explicit = asyncio.create_task(buffer.flush_all())
        await asyncio.sleep(0)
        self.assertFalse(explicit.done())
        producer.gate.set()
        await asyncio.wait_for(explicit, 1)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 0)

    async def test_picture_cancellation_restores_batch_and_retry_accounting(self):
        buffer = self.module.FeatureBuffer(max_batch_size=10)
        await self.add_picture(buffer)
        started = asyncio.Event()
        gate = self.gate()
        original_insert = self.fixture.picture_db.ainsert_embeddings

        async def insert(**kwargs):
            started.set()
            await gate.wait()
            return await original_insert(**kwargs)

        self.fixture.picture_db.ainsert_embeddings = insert
        task = asyncio.create_task(buffer._flush_pictures())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        snapshot = buffer.queue_snapshot()["queues"]["picture"]
        self.assertEqual((snapshot["queued"], snapshot["inflight"], snapshot["retry_queued"]), (1, 0, 1))
        gate.set()
        await buffer._flush_pictures()
        self.assertEqual(buffer.queue_snapshot()["queues"]["picture"]["completed_total"], 1)

    async def test_tbps_cancellation_restores_batch(self):
        buffer = self.module.FeatureBuffer(max_batch_size=10)
        await buffer.tbps.add_feature(
            face_id="face-1", re_id="reid-1", embedding=[0.1, 0.2],
            device_id="CAM001", keyframe_id="frame-1", picture_vector_id=42,
            keyframe_url="CAM001/picture/k.jpg", person_rect={}, timestamp=171, ai_id="ai-1",
        )
        started = asyncio.Event()
        gate = self.gate()
        original_insert = self.fixture.person_db.ainsert_embeddings

        async def insert(**kwargs):
            started.set()
            await gate.wait()
            return await original_insert(**kwargs)

        self.fixture.person_db.ainsert_embeddings = insert
        task = asyncio.create_task(buffer.tbps.flush())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(buffer.tbps.queue.snapshot()["inflight"], 0)
        self.assertEqual(buffer.tbps.queue.snapshot()["retry_queued"], 1)
        gate.set()
        await buffer.tbps.flush()
        self.assertEqual(buffer.tbps.queue.snapshot()["completed_total"], 1)

    async def test_unstarted_buffer_keeps_synchronous_threshold_flush(self):
        buffer = self.module.FeatureBuffer(max_batch_size=1)
        producer = RecordingProducer()
        buffer._kafka_producer = producer
        await buffer.add_face_detection_record({"id": 1})
        self.assertEqual(producer.sent, [("face-detection", {"id": 1})])
        self.assertFalse(buffer.queue_snapshot()["background_running"])

    async def test_cancelled_kafka_flush_retains_only_unacknowledged_records(self):
        buffer = self.module.FeatureBuffer(max_batch_size=10)
        producer = RecordingProducer()
        buffer._kafka_producer = producer
        blocked = asyncio.Event()
        gate = self.gate()
        original_send = producer.send_and_wait

        async def send(topic, payload):
            if json.loads(payload)["id"] == 2:
                blocked.set()
                await gate.wait()
            await original_send(topic, payload)

        producer.send_and_wait = send
        for index in range(1, 4):
            await buffer.add_face_detection_record({"id": index})
        task = asyncio.create_task(buffer._flush_face_detection_records())
        await blocked.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        queue = buffer.face_detection_record_buffer
        self.assertEqual(queue.snapshot()["acked_total"], 1)
        self.assertEqual((queue.qsize(), queue.inflight, queue.retry_queued), (2, 0, 2))
        gate.set()
        await buffer.flush_all()
        self.assertEqual([record["id"] for _, record in producer.sent], [1, 2, 3])

    async def test_stop_waits_for_an_active_vector_write_before_final_drain(self):
        buffer, _producer = await self.make_buffer(max_batch_size=1)
        started = asyncio.Event()
        gate = self.gate()
        original_insert = self.fixture.picture_db.ainsert_embeddings
        cancelled = False

        async def insert(**kwargs):
            nonlocal cancelled
            started.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled = True
                raise
            return await original_insert(**kwargs)

        self.fixture.picture_db.ainsert_embeddings = insert
        await self.add_picture(buffer)
        await started.wait()
        stopping = asyncio.create_task(buffer.stop())
        await asyncio.sleep(0)
        self.assertFalse(stopping.done())
        self.assertFalse(cancelled)
        gate.set()
        await asyncio.wait_for(stopping, 1)
        self.buffers.remove(buffer)
        self.assertFalse(cancelled)
        self.assertEqual(len(self.fixture.picture_db.calls), 1)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 0)

    async def test_cancelled_writer_stop_does_not_cancel_the_write(self):
        from utils.buffers.background_writer import BackgroundWriter, WriteQueue

        queue = WriteQueue(unit="vector_item")
        entered = asyncio.Event()
        gate = self.gate()
        cancelled = False
        calls = 0

        async def flush():
            nonlocal cancelled, calls
            item = queue.get_nowait()
            calls += 1
            entered.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled = True
                queue.requeue([item])
                raise
            queue.complete()

        writer = BackgroundWriter("test", queue, flush, 1)
        self.addAsyncCleanup(writer.stop)
        queue.put_nowait({"id": 1})
        writer.start()
        await entered.wait()
        stopping = asyncio.create_task(writer.stop())
        await asyncio.sleep(0)
        stopping.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await stopping
        self.assertFalse(cancelled)
        self.assertEqual((queue.inflight, queue.retried_total), (1, 0))
        gate.set()
        await asyncio.wait_for(writer.stop(), 1)
        self.assertEqual(calls, 1)
        self.assertEqual(queue.completed_total, 1)

    async def test_cancelled_buffer_stop_continues_cleanup_without_retrying_running_thread(self):
        buffer, producer = await self.make_buffer(max_batch_size=1)
        entered = asyncio.Event()
        thread_gate = threading.Event()
        self.thread_release_events.append(thread_gate)
        loop = asyncio.get_running_loop()
        writes = 0
        cancelled = False

        def blocking_insert():
            nonlocal writes
            writes += 1
            loop.call_soon_threadsafe(entered.set)
            if not thread_gate.wait(2):
                raise RuntimeError("test thread release timed out")
            return {"primary_keys": []}

        async def insert(**kwargs):
            nonlocal cancelled
            try:
                return await asyncio.to_thread(blocking_insert)
            except asyncio.CancelledError:
                cancelled = True
                raise

        self.fixture.picture_db.ainsert_embeddings = insert
        await self.add_picture(buffer)
        await asyncio.wait_for(entered.wait(), 1)
        stopping = asyncio.create_task(buffer.stop())
        await self.eventually(lambda: buffer._stop_task is not None and not buffer._running)
        cleanup = buffer._stop_task
        stopping.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await stopping
        self.assertFalse(cleanup.cancelled())
        self.assertFalse(cancelled)
        self.assertEqual(writes, 1)
        self.assertEqual((buffer.picture_buffer.inflight, buffer.picture_buffer.retried_total), (1, 0))
        producer.stop.assert_not_awaited()

        # A later caller joins the same cleanup instead of flushing a second copy.
        resumed = asyncio.create_task(buffer.stop())
        await asyncio.sleep(0)
        self.assertIs(buffer._stop_task, cleanup)
        self.assertFalse(resumed.done())
        thread_gate.set()
        await asyncio.wait_for(resumed, 1)
        self.buffers.remove(buffer)
        self.assertFalse(cancelled)
        self.assertEqual(writes, 1)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 0)
        producer.stop.assert_awaited_once()

    async def test_restart_restores_backpressure_event_between_watermarks(self):
        buffer, producer = await self.make_buffer(
            max_batch_size=1, backlog_high_watermark=4, backlog_low_watermark=1,
        )
        producer.fail_id = 2
        for index in range(4):
            await buffer.add_face_detection_record({"id": index})
        queue = buffer.face_detection_record_buffer
        await self.eventually(lambda: queue.retried_total == 1)
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 2)
        self.assertTrue(buffer.queue_snapshot()["backpressured"])
        # Keep the residual backlog strictly between low and high during stop.
        producer.fail_always = True
        await buffer.stop()
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 2)
        buffer._kafka_producer = producer
        await buffer.start()
        self.assertTrue(buffer.queue_snapshot()["backpressured"])
        # Assert before creating a waiter: the old bug would starve the event
        # loop, so an asyncio timeout alone could not detect it safely.
        self.assertFalse(buffer._capacity_available.is_set())
        waiter = asyncio.create_task(buffer.wait_for_capacity())
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        producer.fail_always = False
        producer.fail_id = None
        buffer._writers["face_detection"].request()
        await asyncio.wait_for(waiter, 1)
        await buffer.flush_all()
        self.assertEqual(buffer.queue_snapshot()["total_outstanding"], 0)


if __name__ == "__main__":
    unittest.main()
