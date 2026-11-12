import asyncio
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from core.infer.batching import (
    ModelBatchScheduler,
    NormalizedRequest,
    PendingRequest,
)
from core.infer.capabilities import ModelCapability
from core.infer.contracts import MODEL_CONTRACTS
from core.infer.errors import (
    InferenceQueueTimeout,
    InferenceResponseError,
    InferenceUnavailable,
)
from core.infer.executor import FairInferenceExecutor


class SyntheticBaseFailure(BaseException):
    pass


class FakeAdapter:
    contract = MODEL_CONTRACTS["face_det"]

    def __init__(self):
        self.validation_error = None
        self.pack_error = None
        self.unpack_error_for = None
        self.unpack_errors = {}
        self.pack_calls = []
        self.validation_calls = []

    def pack(self, requests):
        self.pack_calls.append(list(requests))
        if self.pack_error is not None:
            raise self.pack_error
        return {
            "VALUE": np.asarray(
                [[request.context] for request in requests], dtype=object
            )
        }

    def validate_outputs(self, outputs, batch_size):
        self.validation_calls.append(batch_size)
        if self.validation_error is not None:
            raise self.validation_error
        if outputs["VALUE"].shape[0] != batch_size:
            raise InferenceResponseError("VALUE batch size mismatch")

    def unpack_one(self, outputs, index, request):
        if request.context in self.unpack_errors:
            raise self.unpack_errors[request.context]
        if request.context == self.unpack_error_for:
            raise ValueError(f"cannot unpack {request.context}")
        return f"result:{request.context}"


class FakeTransport:
    def __init__(self):
        self.batch_sizes = []
        self.calls = []
        self.error = None
        self.outputs = None
        self.started = asyncio.Event()
        self.release = None

    async def infer(self, contract, inputs, *, request_id):
        self.batch_sizes.append(inputs["VALUE"].shape[0])
        self.calls.append((contract, inputs, request_id))
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        if self.outputs is not None:
            return self.outputs
        return {"VALUE": inputs["VALUE"]}


class ImmediateExecutor:
    def __init__(self):
        self.calls = []

    async def submit(self, model, operation, *, deadline):
        self.calls.append((model, deadline))
        return await operation()


class GatedExecutor(ImmediateExecutor):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, model, operation, *, deadline):
        self.calls.append((model, deadline))
        self.started.set()
        await self.release.wait()
        return await operation()


class RejectingExecutor:
    def __init__(self, error):
        self.error = error
        self.calls = []

    async def submit(self, model, operation, *, deadline):
        self.calls.append((model, deadline))
        raise self.error


class RecordingLogger:
    def __init__(self, records=None, context=None):
        self.records = records if records is not None else []
        self.context = context if context is not None else {}

    def bind(self, **context):
        return RecordingLogger(self.records, {**self.context, **context})

    def info(self, message, *args, **kwargs):
        self.records.append((self.context, message, args, kwargs))


class FailingInfoLogger:
    def bind(self, **context):
        return self

    def info(self, message, *args, **kwargs):
        raise RuntimeError("logging failed")


class BaseFailingInfoLogger:
    def __init__(self, error):
        self.error = error

    def bind(self, **context):
        return self

    def info(self, message, *args, **kwargs):
        raise self.error


async def wait_until(predicate, *, timeout=0.5):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=timeout)


def sample(value, key=(10, 12, 3)):
    return NormalizedRequest(
        inputs={"VALUE": np.asarray([value], dtype=object)},
        compatibility_key=key,
        context=value,
    )


def pending_samples(*values):
    loop = asyncio.get_running_loop()
    return [
        PendingRequest(
            f"request-{index}",
            sample(value),
            loop.create_future(),
            0.0,
        )
        for index, value in enumerate(values)
    ]


def make_scheduler(
    *,
    target_batch,
    wait_ms,
    queue_timeout=0.2,
    adapter=None,
    transport=None,
    executor=None,
    logger=None,
):
    adapter = adapter or FakeAdapter()
    transport = transport or FakeTransport()
    executor = executor or ImmediateExecutor()
    capability = ModelCapability(
        key="face_det",
        model_name="yolov8n_face_ensemble",
        api_contract_version=1,
        server_max_batch=4,
        target_batch=target_batch,
    )
    client = ModelBatchScheduler(
        adapter=adapter,
        capability=capability,
        transport=transport,
        executor=executor,
        wait_ms=wait_ms,
        queue_timeout=queue_timeout,
        logger=logger,
    )
    return client, transport, adapter, executor


class StableInterfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_dataclasses_expose_the_planned_fields(self):
        normalized = sample("a")
        future = asyncio.Future()
        pending = PendingRequest("req-1", normalized, future, 1.5)

        self.assertEqual(normalized.context, "a")
        self.assertEqual(normalized.compatibility_key, (10, 12, 3))
        self.assertEqual(pending.request_id, "req-1")
        self.assertIs(pending.normalized, normalized)
        self.assertIs(pending.future, future)
        self.assertEqual(pending.queued_at, 1.5)


class SchedulerBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_capability_values_are_strictly_bounded_integers(self):
        invalid_values = (
            (True, 1, "server_max_batch"),
            (0, 1, "server_max_batch"),
            (1.0, 1, "server_max_batch"),
            (1, True, "target_batch"),
            (1, 0, "target_batch"),
            (1, 1.0, "target_batch"),
            (1, 2, "target_batch"),
        )
        for server_max_batch, target_batch, expected in invalid_values:
            with self.subTest(
                server_max_batch=server_max_batch,
                target_batch=target_batch,
            ):
                capability = ModelCapability(
                    key="face_det",
                    model_name="yolov8n_face_ensemble",
                    api_contract_version=1,
                    server_max_batch=server_max_batch,
                    target_batch=target_batch,
                )
                with self.assertRaisesRegex(ValueError, expected):
                    ModelBatchScheduler(
                        adapter=FakeAdapter(),
                        capability=capability,
                        transport=FakeTransport(),
                        executor=ImmediateExecutor(),
                        wait_ms=0,
                        queue_timeout=0.2,
                    )

    async def test_target_one_dispatches_without_batch_wait(self):
        client, transport, _, _ = make_scheduler(target_batch=1, wait_ms=0)
        self.addAsyncCleanup(client.close)

        result = await client.submit(sample("a"))

        self.assertEqual(result, "result:a")
        self.assertEqual(transport.batch_sizes, [1])

    async def test_formed_batch_waits_for_executor_before_transport(self):
        executor = GatedExecutor()
        client, transport, adapter, _ = make_scheduler(
            target_batch=1,
            wait_ms=0,
            queue_timeout=0.2,
            executor=executor,
        )
        self.addAsyncCleanup(client.close)
        submitted = asyncio.create_task(client.submit(sample("held")))

        try:
            await asyncio.wait_for(executor.started.wait(), timeout=0.1)
            self.assertEqual(adapter.pack_calls, [])
            self.assertEqual(transport.calls, [])
            executor.release.set()
            self.assertEqual(
                await asyncio.wait_for(submitted, timeout=0.1),
                "result:held",
            )
            self.assertEqual(len(transport.calls), 1)
        finally:
            executor.release.set()

    async def test_executor_deadline_uses_original_queue_timestamp(self):
        executor = ImmediateExecutor()
        client, _, _, _ = make_scheduler(
            target_batch=2,
            wait_ms=10_000,
            queue_timeout=0.2,
            executor=executor,
        )
        self.addAsyncCleanup(client.close)

        clock_values = iter((10.0, 11.0, 50.0))
        with patch(
            "core.infer.batching.time.perf_counter",
            side_effect=lambda: next(clock_values, 51.0),
        ):
            first = asyncio.create_task(client.submit(sample("first")))
            await asyncio.sleep(0)
            results = await asyncio.gather(
                first,
                client.submit(sample("second")),
            )

        model, deadline = executor.calls[0]
        self.assertEqual(results, ["result:first", "result:second"])
        self.assertEqual(model, "face_det")
        self.assertEqual(deadline, 10.2)

    async def test_executor_rejection_fails_every_pending_before_transport(self):
        error = RuntimeError("executor rejected before operation")
        executor = RejectingExecutor(error)
        logger = RecordingLogger()
        client, transport, adapter, _ = make_scheduler(
            target_batch=2,
            wait_ms=20,
            executor=executor,
            logger=logger,
        )
        self.addAsyncCleanup(client.close)

        results = await asyncio.gather(
            client.submit(sample("first")),
            client.submit(sample("second")),
            return_exceptions=True,
        )

        self.assertIs(results[0], error)
        self.assertIs(results[1], error)
        self.assertEqual(adapter.pack_calls, [])
        self.assertEqual(transport.calls, [])
        self.assertEqual(len(logger.records), 1)
        _, _, args, _ = logger.records[0]
        self.assertEqual(args[1], 0.0)

    async def assert_pipeline_base_failure_is_completed(self, source):
        logger = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=4,
            logger=logger,
        )
        await executor.start()
        adapter = FakeAdapter()
        transport = FakeTransport()
        error = SyntheticBaseFailure(f"{source} failed")
        if source == "pack":
            adapter.pack_error = error
        elif source == "transport":
            transport.error = error
        else:
            adapter.validation_error = error
        client, _, _, _ = make_scheduler(
            target_batch=2,
            wait_ms=0,
            queue_timeout=1.0,
            adapter=adapter,
            transport=transport,
            executor=executor,
            logger=logger,
        )

        try:
            failures = await asyncio.wait_for(
                asyncio.gather(
                    client.submit(sample("failed-1")),
                    client.submit(sample("failed-2")),
                    return_exceptions=True,
                ),
                timeout=0.2,
            )
            self.assertIs(failures[0], error)
            self.assertIs(failures[1], error)

            adapter.pack_error = None
            adapter.validation_error = None
            transport.error = None
            self.assertEqual(
                await asyncio.wait_for(
                    asyncio.gather(
                        client.submit(sample("recovered-1")),
                        client.submit(sample("recovered-2")),
                    ),
                    timeout=0.2,
                ),
                ["result:recovered-1", "result:recovered-2"],
            )
            await wait_until(lambda: not client._active_dispatches)
            self.assertEqual(client._dispatch_pending, {})
            self.assertEqual(client._request_dispatches, {})
            self.assertEqual(executor._active_count, 0)
        finally:
            await client.close()
            await executor.close()

    async def test_pack_base_failure_completes_batch_and_lane_recovers(self):
        await self.assert_pipeline_base_failure_is_completed("pack")

    async def test_transport_base_failure_completes_batch_and_lane_recovers(self):
        await self.assert_pipeline_base_failure_is_completed("transport")

    async def test_validate_base_failure_completes_batch_and_lane_recovers(self):
        await self.assert_pipeline_base_failure_is_completed("validate")

    async def test_fatal_pipeline_failure_completes_batch_before_reraise(self):
        for error in (SystemExit("stop"), KeyboardInterrupt("stop")):
            with self.subTest(error=type(error).__name__):
                adapter = FakeAdapter()
                adapter.pack_error = error
                client, _, _, _ = make_scheduler(
                    target_batch=2,
                    wait_ms=0,
                    adapter=adapter,
                    logger=RecordingLogger(),
                )
                pending = pending_samples("first", "second")

                try:
                    with self.assertRaises(type(error)) as raised:
                        await client._dispatch(pending)
                    self.assertIs(raised.exception, error)
                    self.assertTrue(
                        all(item.future.done() for item in pending)
                    )
                    self.assertTrue(
                        all(item.future.exception() is error for item in pending)
                    )
                finally:
                    for item in pending:
                        if not item.future.done():
                            item.future.cancel()
                    await client.close()

    async def test_real_executor_transfers_fatal_after_batch_completion(self):
        script_template = textwrap.dedent(
            """
            import asyncio
            import time

            import numpy as np

            from core.infer.batching import (
                ModelBatchScheduler,
                NormalizedRequest,
                PendingRequest,
            )
            from core.infer.capabilities import ModelCapability
            from core.infer.contracts import MODEL_CONTRACTS
            from core.infer.executor import FairInferenceExecutor

            SOURCE = {source!r}
            fatal = SystemExit(23)

            class Adapter:
                contract = MODEL_CONTRACTS["face_det"]

                def pack(self, requests):
                    if SOURCE == "operation":
                        raise fatal
                    return {{
                        "VALUE": np.asarray(
                            [[request.context] for request in requests],
                            dtype=object,
                        )
                    }}

                def validate_outputs(self, outputs, batch_size):
                    pass

                def unpack_one(self, outputs, index, request):
                    return f"result:{{request.context}}"

            class Transport:
                async def infer(self, contract, inputs, *, request_id):
                    return inputs

            class SilentLogger:
                def bind(self, **context):
                    return self

                def info(self, message, *args, **kwargs):
                    pass

            class FatalLogger(SilentLogger):
                def info(self, message, *args, **kwargs):
                    raise fatal

            class RecordingLogger:
                def __init__(self, records=None):
                    self.records = records if records is not None else []

                def bind(self, **context):
                    return RecordingLogger(self.records)

                def info(self, message, *args, **kwargs):
                    self.records.append((message, args, kwargs))

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            executor = FairInferenceExecutor(
                concurrency=1,
                queue_capacity=2,
                logger=FatalLogger() if SOURCE == "logger" else SilentLogger(),
            )
            batch_logger = RecordingLogger()
            scheduler = ModelBatchScheduler(
                adapter=Adapter(),
                capability=ModelCapability(
                    key="face_det",
                    model_name="yolov8n_face_ensemble",
                    api_contract_version=1,
                    server_max_batch=4,
                    target_batch=2,
                ),
                transport=Transport(),
                executor=executor,
                wait_ms=0,
                queue_timeout=1.0,
                logger=batch_logger,
            )
            pending = [
                PendingRequest(
                    f"request-{{index}}",
                    NormalizedRequest(
                        inputs={{
                            "VALUE": np.asarray([value], dtype=object)
                        }},
                        compatibility_key=(1,),
                        context=value,
                    ),
                    loop.create_future(),
                    time.perf_counter(),
                )
                for index, value in enumerate(("first", "second"))
            ]
            state = {{}}

            async def bootstrap():
                await executor.start()
                state["dispatch"] = asyncio.create_task(
                    scheduler._dispatch(pending)
                )

            loop.run_until_complete(bootstrap())
            dispatch = state["dispatch"]
            try:
                loop.run_until_complete(dispatch)
            except SystemExit as caught:
                assert caught is fatal
            else:
                raise AssertionError("dispatch did not propagate SystemExit")

            assert dispatch.done(), "scheduler dispatch is still pending"
            assert all(item.future.done() for item in pending), pending
            assert all(
                item.future.exception() is fatal for item in pending
            )
            assert len(batch_logger.records) == 1
            assert batch_logger.records[0][1][0] == "SystemExit"
            assert executor._active_count == 0
            assert all(lane.done() for lane in executor._lanes)
            lane_failure = executor._lanes[0].exception()
            assert isinstance(lane_failure, Exception)
            assert lane_failure.__cause__ is fatal

            try:
                loop.run_until_complete(executor.close())
            except RuntimeError as close_error:
                assert close_error.__cause__ is fatal
            else:
                raise AssertionError("close did not report executor fatal")

            loop.run_until_complete(scheduler.close())
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()
            print(f"fatal-transferred:{{SOURCE}}")
            """
        )

        for source in ("operation", "logger"):
            with self.subTest(source=source):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        script_template.format(source=source),
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"fatal-transferred:{source}",
                    result.stdout,
                )
                self.assertNotIn(
                    "Task exception was never retrieved",
                    result.stderr,
                )

    async def test_unpack_base_failure_isolated_from_other_batch_items(self):
        logger = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=4,
            logger=logger,
        )
        await executor.start()
        adapter = FakeAdapter()
        error = SyntheticBaseFailure("cannot unpack")
        adapter.unpack_errors["bad"] = error
        client, _, _, _ = make_scheduler(
            target_batch=2,
            wait_ms=0,
            queue_timeout=1.0,
            adapter=adapter,
            executor=executor,
            logger=logger,
        )
        tasks = [
            asyncio.create_task(client.submit(sample("bad"))),
            asyncio.create_task(client.submit(sample("good"))),
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=0.2,
            )
            self.assertIs(results[0], error)
            self.assertEqual(results[1], "result:good")
            await wait_until(lambda: not client._active_dispatches)
            self.assertEqual(client._dispatch_pending, {})
            self.assertEqual(client._request_dispatches, {})
            self.assertEqual(executor._active_count, 0)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()
            await executor.close()

    async def test_unpack_cancelled_error_cancels_only_its_item(self):
        logger = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=4,
            logger=logger,
        )
        await executor.start()
        adapter = FakeAdapter()
        adapter.unpack_errors["cancelled"] = asyncio.CancelledError(
            "adapter cancelled item"
        )
        client, _, _, _ = make_scheduler(
            target_batch=2,
            wait_ms=0,
            queue_timeout=1.0,
            adapter=adapter,
            executor=executor,
            logger=logger,
        )
        tasks = [
            asyncio.create_task(client.submit(sample("cancelled"))),
            asyncio.create_task(client.submit(sample("good"))),
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=0.2,
            )
            self.assertIsInstance(results[0], asyncio.CancelledError)
            self.assertTrue(tasks[0].cancelled())
            self.assertEqual(results[1], "result:good")

            adapter.unpack_errors.clear()
            recovered = await asyncio.wait_for(
                asyncio.gather(
                    client.submit(sample("recovered-1")),
                    client.submit(sample("recovered-2")),
                ),
                timeout=0.2,
            )
            self.assertEqual(
                recovered,
                ["result:recovered-1", "result:recovered-2"],
            )
            await wait_until(lambda: not client._active_dispatches)
            self.assertEqual(client._dispatch_pending, {})
            self.assertEqual(client._request_dispatches, {})
            self.assertEqual(executor._active_count, 0)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()
            await executor.close()

    async def test_fatal_unpack_completes_remaining_items_before_reraise(self):
        for error in (SystemExit("stop"), KeyboardInterrupt("stop")):
            with self.subTest(error=type(error).__name__):
                adapter = FakeAdapter()
                adapter.unpack_errors["fatal"] = error
                client, _, _, _ = make_scheduler(
                    target_batch=3,
                    wait_ms=0,
                    adapter=adapter,
                    logger=RecordingLogger(),
                )
                pending = pending_samples("good", "fatal", "remaining")

                try:
                    with self.assertRaises(type(error)) as raised:
                        await client._dispatch(pending)
                    self.assertIs(raised.exception, error)
                    self.assertEqual(pending[0].future.result(), "result:good")
                    self.assertIs(pending[1].future.exception(), error)
                    self.assertIs(pending[2].future.exception(), error)
                finally:
                    for item in pending:
                        if not item.future.done():
                            item.future.cancel()
                    await client.close()

    async def test_log_batch_swallows_nonfatal_base_exceptions(self):
        for error in (
            SyntheticBaseFailure("logging failed"),
            asyncio.CancelledError(),
        ):
            with self.subTest(error=type(error).__name__):
                client, _, _, _ = make_scheduler(
                    target_batch=1,
                    wait_ms=0,
                    logger=BaseFailingInfoLogger(error),
                )
                pending = pending_samples("logged")

                try:
                    client._log_batch(
                        pending=pending,
                        queue_wait_ms=0.0,
                        packed_inputs={},
                        error_class=None,
                        infer_ms=0.0,
                        total_ms=0.0,
                    )
                finally:
                    pending[0].future.cancel()
                    await client.close()

    async def test_fatal_logger_rethrows_after_business_completion(self):
        for error in (SystemExit("stop"), KeyboardInterrupt("stop")):
            with self.subTest(error=type(error).__name__):
                client, _, _, _ = make_scheduler(
                    target_batch=1,
                    wait_ms=0,
                    logger=BaseFailingInfoLogger(error),
                )
                pending = pending_samples("completed")

                try:
                    with self.assertRaises(type(error)) as raised:
                        await client._dispatch(pending)
                    self.assertIs(raised.exception, error)
                    self.assertEqual(
                        pending[0].future.result(),
                        "result:completed",
                    )
                finally:
                    if not pending[0].future.done():
                        pending[0].future.cancel()
                    await client.close()

    async def test_compatible_requests_fill_one_batch_in_order(self):
        client, transport, _, _ = make_scheduler(target_batch=2, wait_ms=20)
        self.addAsyncCleanup(client.close)

        results = await asyncio.gather(
            client.submit(sample("a", key=(10, 12, 3))),
            client.submit(sample("b", key=(10, 12, 3))),
        )

        self.assertEqual(results, ["result:a", "result:b"])
        self.assertEqual(transport.batch_sizes, [2])
        np.testing.assert_array_equal(
            transport.calls[0][1]["VALUE"],
            np.asarray([["a"], ["b"]], dtype=object),
        )

    async def test_different_shapes_are_dispatched_in_separate_batches(self):
        client, transport, _, _ = make_scheduler(target_batch=2, wait_ms=1)
        self.addAsyncCleanup(client.close)

        await asyncio.gather(
            client.submit(sample("a", key=(10, 12, 3))),
            client.submit(sample("b", key=(20, 24, 3))),
        )

        self.assertEqual(sorted(transport.batch_sizes), [1, 1])

    async def test_cancelled_waiter_is_removed_before_dispatch(self):
        client, transport, _, _ = make_scheduler(target_batch=2, wait_ms=20)
        self.addAsyncCleanup(client.close)
        task = asyncio.create_task(client.submit(sample("cancelled")))
        await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.04)

        self.assertEqual(transport.batch_sizes, [])

    async def test_cancelling_all_callers_removes_queued_executor_batch(self):
        logger = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=8,
            logger=logger,
        )
        await executor.start()
        self.addAsyncCleanup(executor.close)
        transport = FakeTransport()
        transport.release = asyncio.Event()
        client, _, _, _ = make_scheduler(
            target_batch=1,
            wait_ms=0,
            queue_timeout=1.0,
            transport=transport,
            executor=executor,
            logger=logger,
        )
        self.addAsyncCleanup(client.close)
        first = asyncio.create_task(client.submit(sample("active")))
        second = None

        try:
            await asyncio.wait_for(transport.started.wait(), timeout=0.1)
            second = asyncio.create_task(client.submit(sample("cancelled")))
            await wait_until(lambda: executor._queued_count == 1)

            second.cancel()
            cancelled = await asyncio.gather(
                second,
                return_exceptions=True,
            )

            self.assertIsInstance(cancelled[0], asyncio.CancelledError)
            await wait_until(lambda: executor._queued_count == 0)
            self.assertEqual(len(transport.calls), 1)
        finally:
            transport.release.set()
            tasks = [first]
            if second is not None:
                tasks.append(second)
            await asyncio.gather(*tasks, return_exceptions=True)

        await wait_until(lambda: not client._active_dispatches)
        self.assertEqual(client._dispatch_pending, {})
        self.assertEqual(client._request_dispatches, {})
        self.assertEqual(len(transport.calls), 1)

    async def test_partial_batch_cancellation_keeps_executor_batch(self):
        logger = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=8,
            logger=logger,
        )
        await executor.start()
        self.addAsyncCleanup(executor.close)
        transport = FakeTransport()
        transport.release = asyncio.Event()
        client, _, _, _ = make_scheduler(
            target_batch=2,
            wait_ms=20,
            queue_timeout=1.0,
            transport=transport,
            executor=executor,
            logger=logger,
        )
        self.addAsyncCleanup(client.close)
        active = [
            asyncio.create_task(client.submit(sample("active-1"))),
            asyncio.create_task(client.submit(sample("active-2"))),
        ]
        cancelled = None
        live = None

        try:
            await asyncio.wait_for(transport.started.wait(), timeout=0.1)
            cancelled = asyncio.create_task(
                client.submit(sample("cancelled"))
            )
            live = asyncio.create_task(client.submit(sample("live")))
            await wait_until(lambda: executor._queued_count == 1)

            cancelled.cancel()
            cancelled_result = await asyncio.gather(
                cancelled,
                return_exceptions=True,
            )

            self.assertIsInstance(
                cancelled_result[0], asyncio.CancelledError
            )
            self.assertEqual(executor._queued_count, 1)
            transport.release.set()
            self.assertEqual(
                await asyncio.wait_for(live, timeout=0.1),
                "result:live",
            )
            self.assertEqual(
                await asyncio.gather(*active),
                ["result:active-1", "result:active-2"],
            )
        finally:
            transport.release.set()
            tasks = [*active]
            if cancelled is not None:
                tasks.append(cancelled)
            if live is not None:
                tasks.append(live)
            await asyncio.gather(*tasks, return_exceptions=True)

        await wait_until(lambda: not client._active_dispatches)
        self.assertEqual(client._dispatch_pending, {})
        self.assertEqual(client._request_dispatches, {})
        self.assertEqual(transport.batch_sizes, [2, 2])

    async def test_burst_only_packs_executor_started_batches(self):
        logger = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=1,
            logger=logger,
        )
        await executor.start()
        self.addAsyncCleanup(executor.close)
        transport = FakeTransport()
        transport.release = asyncio.Event()
        adapter = FakeAdapter()
        client, _, _, _ = make_scheduler(
            target_batch=1,
            wait_ms=0,
            queue_timeout=2.0,
            adapter=adapter,
            transport=transport,
            executor=executor,
            logger=logger,
        )
        self.addAsyncCleanup(client.close)
        first = asyncio.create_task(client.submit(sample("active")))
        rest = []
        results = []

        try:
            await asyncio.wait_for(transport.started.wait(), timeout=0.1)
            rest = [
                asyncio.create_task(client.submit(sample(f"burst-{index}")))
                for index in range(99)
            ]
            await wait_until(
                lambda: sum(task.done() for task in rest) >= 98
                and executor._queued_count == 1
                and len(client._active_dispatches) <= 2
            )

            overflow = [task.exception() for task in rest if task.done()]
            self.assertEqual(len(overflow), 98)
            self.assertTrue(
                all(isinstance(item, InferenceUnavailable) for item in overflow)
            )
            self.assertEqual(executor._active_count, 1)
            self.assertEqual(executor._queued_count, 1)
            self.assertLessEqual(len(client._active_dispatches), 2)
            self.assertEqual(len(adapter.pack_calls), 1)
        finally:
            transport.release.set()
            results = await asyncio.gather(
                first,
                *rest,
                return_exceptions=True,
            )

        await wait_until(lambda: not client._active_dispatches)
        successes = [item for item in results if isinstance(item, str)]
        unavailable = [
            item for item in results if isinstance(item, InferenceUnavailable)
        ]
        self.assertEqual(len(successes), 2)
        self.assertEqual(len(unavailable), 98)
        self.assertEqual(len(adapter.pack_calls), 2)
        self.assertEqual(client._dispatch_pending, {})
        self.assertEqual(client._request_dispatches, {})

    async def test_transport_failure_reaches_every_live_waiter(self):
        client, transport, _, _ = make_scheduler(target_batch=2, wait_ms=20)
        self.addAsyncCleanup(client.close)
        transport.error = InferenceUnavailable("offline")

        results = await asyncio.gather(
            client.submit(sample("a")),
            client.submit(sample("b")),
            return_exceptions=True,
        )

        self.assertTrue(
            all(isinstance(item, InferenceUnavailable) for item in results)
        )

    async def test_validate_outputs_failure_reaches_every_live_waiter(self):
        adapter = FakeAdapter()
        adapter.validation_error = InferenceResponseError("unaligned")
        client, _, _, _ = make_scheduler(
            target_batch=2, wait_ms=20, adapter=adapter
        )
        self.addAsyncCleanup(client.close)

        results = await asyncio.gather(
            client.submit(sample("a")),
            client.submit(sample("b")),
            return_exceptions=True,
        )

        self.assertEqual(adapter.validation_calls, [2])
        self.assertTrue(
            all(isinstance(item, InferenceResponseError) for item in results)
        )

    async def test_queue_timeout_only_expires_an_undispatched_request(self):
        client, transport, _, _ = make_scheduler(
            target_batch=2, wait_ms=100, queue_timeout=0.01
        )
        self.addAsyncCleanup(client.close)

        with self.assertRaises(InferenceQueueTimeout):
            await client.submit(sample("late"))

        self.assertEqual(transport.batch_sizes, [])

    async def test_detaching_a_batch_cancels_its_queue_timeout(self):
        transport = FakeTransport()
        transport.release = asyncio.Event()
        client, _, _, _ = make_scheduler(
            target_batch=1,
            wait_ms=0,
            queue_timeout=0.01,
            transport=transport,
        )
        self.addAsyncCleanup(client.close)
        task = asyncio.create_task(client.submit(sample("remote-slow")))
        await transport.started.wait()

        await asyncio.sleep(0.03)
        self.assertFalse(task.done())
        transport.release.set()

        self.assertEqual(await task, "result:remote-slow")

    async def test_wait_deadline_flushes_a_partial_batch(self):
        client, transport, _, _ = make_scheduler(target_batch=3, wait_ms=10)
        self.addAsyncCleanup(client.close)

        results = await asyncio.gather(
            client.submit(sample("a")), client.submit(sample("b"))
        )

        self.assertEqual(results, ["result:a", "result:b"])
        self.assertEqual(transport.batch_sizes, [2])

    async def test_stale_flush_does_not_detach_a_new_bucket_generation(self):
        key = (10, 12, 3)
        client, transport, _, _ = make_scheduler(
            target_batch=2, wait_ms=10_000, logger=RecordingLogger()
        )
        self.addAsyncCleanup(client.close)
        first = asyncio.create_task(client.submit(sample("a", key=key)))
        await asyncio.sleep(0)
        old_timer = client._batch_timers[key]
        old_callback = old_timer._callback
        old_args = old_timer._args

        second = asyncio.create_task(client.submit(sample("b", key=key)))
        self.assertEqual(
            await asyncio.gather(first, second), ["result:a", "result:b"]
        )

        async with client._lock:
            third = asyncio.create_task(client.submit(sample("c", key=key)))
            old_callback(*old_args)
        await asyncio.sleep(0.02)

        self.assertFalse(third.done())
        self.assertEqual(transport.batch_sizes, [2])
        fourth = asyncio.create_task(client.submit(sample("d", key=key)))
        self.assertEqual(
            await asyncio.gather(third, fourth), ["result:c", "result:d"]
        )
        self.assertEqual(transport.batch_sizes, [2, 2])

    async def test_output_batch_count_mismatch_fails_the_whole_batch(self):
        transport = FakeTransport()
        transport.outputs = {
            "VALUE": np.asarray([["only-one"]], dtype=object)
        }
        client, _, adapter, _ = make_scheduler(
            target_batch=2, wait_ms=20, transport=transport
        )
        self.addAsyncCleanup(client.close)

        results = await asyncio.gather(
            client.submit(sample("a")),
            client.submit(sample("b")),
            return_exceptions=True,
        )

        self.assertEqual(adapter.validation_calls, [2])
        self.assertTrue(
            all(isinstance(item, InferenceResponseError) for item in results)
        )

    async def test_unpack_error_only_fails_its_own_request(self):
        adapter = FakeAdapter()
        adapter.unpack_error_for = "bad"
        client, _, _, _ = make_scheduler(
            target_batch=2, wait_ms=20, adapter=adapter
        )
        self.addAsyncCleanup(client.close)

        results = await asyncio.gather(
            client.submit(sample("bad")),
            client.submit(sample("good")),
            return_exceptions=True,
        )

        self.assertIsInstance(results[0], ValueError)
        self.assertEqual(results[1], "result:good")

    async def test_unpack_error_is_reported_in_the_single_batch_log(self):
        logger = RecordingLogger()
        adapter = FakeAdapter()
        adapter.unpack_error_for = "secret-bad-item"
        client, _, _, _ = make_scheduler(
            target_batch=2,
            wait_ms=20,
            adapter=adapter,
            logger=logger,
        )
        self.addAsyncCleanup(client.close)

        await asyncio.gather(
            client.submit(sample("secret-bad-item")),
            client.submit(sample("good")),
            return_exceptions=True,
        )

        self.assertEqual(len(logger.records), 1)
        context, message, args, kwargs = logger.records[0]
        self.assertIn("status=error", message)
        self.assertIn("ValueError", repr((context, message, args, kwargs)))
        self.assertNotIn(
            "secret-bad-item", repr((context, message, args, kwargs))
        )

    async def test_close_cancels_timers_and_rejects_pending_requests(self):
        client, transport, _, _ = make_scheduler(
            target_batch=2, wait_ms=100, queue_timeout=0.05
        )
        task = asyncio.create_task(client.submit(sample("pending")))
        await asyncio.sleep(0)

        await client.close()

        with self.assertRaises(InferenceUnavailable):
            await task
        await asyncio.sleep(0.06)
        self.assertEqual(transport.batch_sizes, [])

    async def test_close_waits_for_an_active_dispatch(self):
        transport = FakeTransport()
        transport.release = asyncio.Event()
        client, _, _, _ = make_scheduler(
            target_batch=1, wait_ms=0, transport=transport
        )
        task = asyncio.create_task(client.submit(sample("active")))
        await transport.started.wait()

        close_task = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        self.assertFalse(close_task.done())
        transport.release.set()

        self.assertEqual(await task, "result:active")
        await close_task

    async def test_cancelling_one_close_keeps_shared_graceful_close_alive(self):
        transport = FakeTransport()
        transport.release = asyncio.Event()
        client, _, _, _ = make_scheduler(
            target_batch=1, wait_ms=0, transport=transport
        )
        submit_task = asyncio.create_task(client.submit(sample("active")))
        await transport.started.wait()

        first_close = asyncio.create_task(client.close())
        second_close = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        first_close.cancel()
        first_result = await asyncio.gather(
            first_close, return_exceptions=True
        )
        await asyncio.sleep(0)

        self.assertIsInstance(first_result[0], asyncio.CancelledError)
        self.assertFalse(submit_task.done())
        self.assertFalse(second_close.done())
        transport.release.set()
        self.assertEqual(await submit_task, "result:active")
        await second_close
        await client.close()

    async def test_submit_after_close_is_rejected(self):
        client, _, _, _ = make_scheduler(target_batch=1, wait_ms=0)
        await client.close()

        with self.assertRaises(InferenceUnavailable):
            await client.submit(sample("too-late"))

    async def test_success_log_contains_batch_metadata_but_no_input_values(self):
        logger = RecordingLogger()
        client, _, _, _ = make_scheduler(
            target_batch=1, wait_ms=0, logger=logger
        )
        self.addAsyncCleanup(client.close)

        await client.submit(sample("secret-image-content"))

        self.assertEqual(len(logger.records), 1)
        context, message, args, kwargs = logger.records[0]
        self.assertEqual(context["model"], "face_det")
        self.assertEqual(context["api_contract_version"], 1)
        self.assertEqual(context["batch_size"], 1)
        self.assertEqual(context["server_max_batch"], 4)
        self.assertEqual(context["target_batch"], 1)
        self.assertEqual(context["input_shapes"], {"VALUE": (1, 1)})
        self.assertEqual(len(context["request_ids"]), 1)
        self.assertIn("status=ok", message)
        self.assertIn("infer_ms=", message.format(*args))
        self.assertIn("total_ms=", message.format(*args))
        self.assertEqual(kwargs, {})
        self.assertNotIn(
            "secret-image-content", repr((context, message, args, kwargs))
        )

    async def test_failure_log_contains_error_class_and_timings(self):
        logger = RecordingLogger()
        transport = FakeTransport()
        transport.error = InferenceUnavailable("secret-remote-message")
        client, _, _, _ = make_scheduler(
            target_batch=1,
            wait_ms=0,
            transport=transport,
            logger=logger,
        )
        self.addAsyncCleanup(client.close)

        with self.assertRaises(InferenceUnavailable):
            await client.submit(sample("secret-input"))

        self.assertEqual(len(logger.records), 1)
        context, message, args, kwargs = logger.records[0]
        self.assertIn("status=error", message)
        self.assertIn("InferenceUnavailable", repr((context, message, args, kwargs)))
        self.assertGreaterEqual(len(args), 2)
        self.assertNotIn("secret-input", repr((context, message, args, kwargs)))
        self.assertNotIn(
            "secret-remote-message", repr((context, message, args, kwargs))
        )

    async def test_logger_bind_failure_does_not_change_a_success_result(self):
        client, _, _, _ = make_scheduler(
            target_batch=1, wait_ms=0, logger=object()
        )
        self.addAsyncCleanup(client.close)

        result = await asyncio.wait_for(
            client.submit(sample("success")), timeout=0.1
        )

        self.assertEqual(result, "result:success")

    async def test_logger_info_failure_does_not_hide_a_batch_failure(self):
        transport = FakeTransport()
        transport.error = InferenceUnavailable("offline")
        client, _, _, _ = make_scheduler(
            target_batch=1,
            wait_ms=0,
            transport=transport,
            logger=FailingInfoLogger(),
        )
        self.addAsyncCleanup(client.close)

        with self.assertRaises(InferenceUnavailable):
            await asyncio.wait_for(
                client.submit(sample("failed")), timeout=0.1
            )

    async def test_logger_info_failure_keeps_unpack_results_isolated(self):
        adapter = FakeAdapter()
        adapter.unpack_error_for = "bad"
        client, _, _, _ = make_scheduler(
            target_batch=2,
            wait_ms=20,
            adapter=adapter,
            logger=FailingInfoLogger(),
        )
        self.addAsyncCleanup(client.close)

        results = await asyncio.wait_for(
            asyncio.gather(
                client.submit(sample("bad")),
                client.submit(sample("good")),
                return_exceptions=True,
            ),
            timeout=0.1,
        )

        self.assertIsInstance(results[0], ValueError)
        self.assertEqual(results[1], "result:good")


if __name__ == "__main__":
    unittest.main()
