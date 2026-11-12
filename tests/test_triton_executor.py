import asyncio
import gc
import math
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import unittest
import weakref
from unittest.mock import patch

from core.infer.errors import InferenceQueueTimeout, InferenceUnavailable
from core.infer.executor import FairInferenceExecutor


class RecordingLogger:
    def __init__(self, records=None, context=None):
        self.records = records if records is not None else []
        self.context = context if context is not None else {}

    def bind(self, **context):
        return RecordingLogger(self.records, {**self.context, **context})

    def info(self, message, *args, **kwargs):
        self.records.append((self.context, message, args, kwargs))


class FailingLogger:
    def __init__(self):
        self.contexts = []

    def bind(self, **context):
        self.contexts.append(context)
        return self

    def info(self, message, *args, **kwargs):
        raise RuntimeError("logging is unavailable")


class SyntheticBaseFailure(BaseException):
    pass


class CancelledBindLogger:
    def bind(self, **context):
        raise asyncio.CancelledError


class SyntheticBaseInfoLogger:
    def bind(self, **context):
        return self

    def info(self, message, *args, **kwargs):
        raise SyntheticBaseFailure("logging failed")


class PausingBeforeOperationExecutor(FairInferenceExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.item_dequeued = asyncio.Event()
        self.allow_operation = asyncio.Event()

    async def _execute(self, item):
        self.item_dequeued.set()
        await self.allow_operation.wait()
        await super()._execute(item)


class PausingBeforeTakeExecutor(FairInferenceExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lane_waiting = asyncio.Event()
        self.allow_take = asyncio.Event()

    async def _take_next(self):
        self.lane_waiting.set()
        await self.allow_take.wait()
        return await super()._take_next()


async def wait_until(predicate, *, attempts=100):
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


class FairInferenceExecutorTests(unittest.IsolatedAsyncioTestCase):
    def assert_completion_log(self, record, *, model, status, error_class):
        context, message, args, kwargs = record
        self.assertEqual(context["model"], model)
        self.assertEqual(context["status"], status)
        self.assertEqual(context["error_class"], error_class)
        self.assertTrue(
            {
                "model",
                "executor_queue_wait_ms",
                "executor_queue_depth",
                "executor_active_slots",
                "executor_concurrency",
                "deadline_remaining_ms",
                "status",
                "error_class",
            }.issubset(context)
        )
        self.assertEqual(
            message,
            "Triton executor completed: status={}, error_class={}",
        )
        self.assertEqual(args, (status, error_class))
        self.assertEqual(kwargs, {})

    async def test_concurrency_is_bounded_without_limiting_one_model_to_one_lane(self):
        executor = FairInferenceExecutor(concurrency=2, queue_capacity=8)
        await executor.start()
        release = asyncio.Event()
        started = []
        active = 0
        peak = 0

        def make_operation(index):
            async def operation():
                nonlocal active, peak
                started.append(index)
                active += 1
                peak = max(peak, active)
                await release.wait()
                active -= 1
                return index

            return operation

        deadline = time.perf_counter() + 2
        submissions = [
            asyncio.create_task(
                executor.submit("clip_image", make_operation(index), deadline=deadline)
            )
            for index in range(4)
        ]

        await wait_until(lambda: len(started) == 2)
        self.assertEqual(peak, 2)
        self.assertEqual(started, [0, 1])
        release.set()
        self.assertEqual(await asyncio.gather(*submissions), [0, 1, 2, 3])
        await executor.close()

    async def test_fifty_slow_requests_never_exceed_executor_capacity(self):
        executor = FairInferenceExecutor(concurrency=8, queue_capacity=64)
        await executor.start()
        self.addAsyncCleanup(executor.close)
        active = 0
        peak = 0

        async def slow_operation(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.005)
                return value
            finally:
                active -= 1

        results = await asyncio.gather(
            *(
                executor.submit(
                    "clip_image",
                    lambda value=value: slow_operation(value),
                    deadline=time.perf_counter() + 1,
                )
                for value in range(50)
            )
        )
        self.assertEqual(results, list(range(50)))
        self.assertEqual(peak, 8)

    async def test_single_lane_uses_round_robin_between_model_queues(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=8)
        await executor.start()
        release_a0 = asyncio.Event()
        order = []

        async def a0():
            order.append("a0")
            await release_a0.wait()

        def record(name):
            async def operation():
                order.append(name)

            return operation

        deadline = time.perf_counter() + 2
        first = asyncio.create_task(executor.submit("a", a0, deadline=deadline))
        await wait_until(lambda: order == ["a0"])
        queued = [
            asyncio.create_task(executor.submit("a", record("a1"), deadline=deadline)),
            asyncio.create_task(executor.submit("a", record("a2"), deadline=deadline)),
            asyncio.create_task(executor.submit("b", record("b1"), deadline=deadline)),
        ]
        await wait_until(lambda: executor._queued_count == 3)

        release_a0.set()
        await asyncio.gather(first, *queued)

        self.assertEqual(order, ["a0", "a1", "b1", "a2"])
        await executor.close()

    async def test_expired_queued_request_does_not_call_operation(self):
        log = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=2,
            logger=log,
        )
        await executor.start()
        release = asyncio.Event()
        blocker_started = asyncio.Event()
        expired_called = False

        async def blocker():
            blocker_started.set()
            await release.wait()

        async def expired_operation():
            nonlocal expired_called
            expired_called = True

        blocker_task = asyncio.create_task(
            executor.submit("busy", blocker, deadline=time.perf_counter() + 2)
        )
        await blocker_started.wait()

        with self.assertRaisesRegex(
            InferenceQueueTimeout,
            "timed_out: request expired in global inference queue",
        ):
            await executor.submit(
                "timed_out",
                expired_operation,
                deadline=time.perf_counter() + 0.01,
            )

        self.assertFalse(expired_called)
        release.set()
        await blocker_task
        await executor.close()
        timeout_records = [
            record for record in log.records if record[0]["model"] == "timed_out"
        ]
        self.assertEqual(len(timeout_records), 1)
        self.assert_completion_log(
            timeout_records[0],
            model="timed_out",
            status="error",
            error_class="InferenceQueueTimeout",
        )

    async def test_deadline_expiring_immediately_after_enqueue_uses_queue_error(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=1)
        await executor.start()
        operation_called = False

        async def operation():
            nonlocal operation_called
            operation_called = True

        try:
            with patch(
                "core.infer.executor.time.perf_counter",
                side_effect=[1.0, 2.0, 4.0, 4.0],
            ):
                with self.assertRaises(InferenceQueueTimeout):
                    await executor.submit("a", operation, deadline=3.0)
        finally:
            await executor.close()

        self.assertFalse(operation_called)

    async def test_full_queue_submission_is_rejected_immediately(self):
        log = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=1,
            logger=log,
        )
        await executor.start()
        release = asyncio.Event()
        blocker_started = asyncio.Event()
        order = []

        async def blocker():
            blocker_started.set()
            await release.wait()
            return "blocker"

        async def queued():
            order.append("queued")
            return "queued"

        async def rejected():
            order.append("rejected")

        deadline = time.perf_counter() + 2
        first = asyncio.create_task(
            executor.submit("busy", blocker, deadline=deadline)
        )
        await blocker_started.wait()
        second = asyncio.create_task(
            executor.submit("queued", queued, deadline=deadline)
        )
        await wait_until(lambda: executor._queued_count == 1)

        try:
            with self.assertRaisesRegex(
                InferenceUnavailable,
                "capacity_overflow: global inference queue is full",
            ):
                await asyncio.wait_for(
                    executor.submit(
                        "capacity_overflow",
                        rejected,
                        deadline=deadline,
                    ),
                    timeout=0.1,
                )

            self.assertEqual(executor._queued_count, 1)
            self.assertEqual(len(executor._condition._waiters), 0)
            self.assertNotIn("rejected", order)
        finally:
            release.set()
            completed = await asyncio.gather(first, second)
            await executor.close()

        self.assertEqual(completed, ["blocker", "queued"])
        self.assertEqual(order, ["queued"])
        overflow_records = [
            record
            for record in log.records
            if record[0]["model"] == "capacity_overflow"
        ]
        self.assertEqual(len(overflow_records), 1)
        self.assert_completion_log(
            overflow_records[0],
            model="capacity_overflow",
            status="error",
            error_class="InferenceUnavailable",
        )

    async def test_burst_overflow_does_not_create_capacity_waiters(self):
        log = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=2,
            logger=log,
        )
        await executor.start()
        release = asyncio.Event()
        blocker_started = asyncio.Event()
        overflow_called = False

        async def blocker():
            blocker_started.set()
            await release.wait()
            return "blocker"

        async def queued(index):
            return index

        async def overflow_operation():
            nonlocal overflow_called
            overflow_called = True

        deadline = time.perf_counter() + 2
        first = asyncio.create_task(
            executor.submit("busy", blocker, deadline=deadline)
        )
        await blocker_started.wait()
        accepted = [
            asyncio.create_task(
                executor.submit(
                    "accepted",
                    lambda index=index: queued(index),
                    deadline=deadline,
                )
            )
            for index in range(2)
        ]
        await wait_until(lambda: executor._queued_count == 2)

        try:
            overflow_results = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        executor.submit(
                            "overflow",
                            overflow_operation,
                            deadline=deadline,
                        )
                        for _ in range(50)
                    ],
                    return_exceptions=True,
                ),
                timeout=0.1,
            )

            self.assertTrue(
                all(
                    isinstance(result, InferenceUnavailable)
                    for result in overflow_results
                )
            )
            self.assertEqual(executor._queued_count, 2)
            self.assertEqual(len(executor._condition._waiters), 0)
            self.assertFalse(overflow_called)
        finally:
            release.set()
            first_result = await first
            accepted_results = await asyncio.gather(*accepted)
            await executor.close()

        self.assertEqual(first_result, "blocker")
        self.assertEqual(accepted_results, [0, 1])
        overflow_records = [
            record for record in log.records if record[0]["model"] == "overflow"
        ]
        self.assertEqual(len(overflow_records), 50)
        self.assertTrue(
            all(
                record[0]["error_class"] == "InferenceUnavailable"
                for record in overflow_records
            )
        )

    async def test_cancelling_queued_submit_removes_it_without_calling_operation(self):
        log = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=2,
            logger=log,
        )
        await executor.start()
        release = asyncio.Event()
        blocker_started = asyncio.Event()
        cancelled_called = False

        async def blocker():
            blocker_started.set()
            await release.wait()

        async def cancelled_operation():
            nonlocal cancelled_called
            cancelled_called = True

        deadline = time.perf_counter() + 2
        first = asyncio.create_task(
            executor.submit("busy", blocker, deadline=deadline)
        )
        await blocker_started.wait()
        cancelled = asyncio.create_task(
            executor.submit("cancelled", cancelled_operation, deadline=deadline)
        )
        await wait_until(lambda: executor._queued_count == 1)

        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        await wait_until(lambda: executor._queued_count == 0)

        self.assertFalse(cancelled_called)
        release.set()
        await first
        await executor.close()
        cancellation_records = [
            record for record in log.records if record[0]["model"] == "cancelled"
        ]
        self.assertEqual(len(cancellation_records), 1)
        self.assert_completion_log(
            cancellation_records[0],
            model="cancelled",
            status="error",
            error_class="CancelledError",
        )

    async def test_close_drains_accepted_work_and_rejects_new_submissions(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=2)
        await executor.start()
        release = asyncio.Event()
        started = asyncio.Event()
        completed = []

        async def blocking_operation():
            started.set()
            await release.wait()
            completed.append("first")

        async def queued_operation():
            completed.append("second")

        deadline = time.perf_counter() + 2
        first = asyncio.create_task(
            executor.submit("a", blocking_operation, deadline=deadline)
        )
        await started.wait()
        second = asyncio.create_task(
            executor.submit("b", queued_operation, deadline=deadline)
        )
        await wait_until(lambda: executor._queued_count == 1)
        closing = asyncio.create_task(executor.close())
        await wait_until(lambda: executor._state == executor._CLOSING)

        with self.assertRaisesRegex(
            InferenceUnavailable, "inference executor is closing"
        ):
            await executor.submit("c", queued_operation, deadline=deadline)
        with self.assertRaisesRegex(
            InferenceUnavailable, "inference executor is closing"
        ):
            await executor.start()

        release.set()
        await asyncio.gather(first, second, closing)
        self.assertEqual(completed, ["first", "second"])
        await executor.close()

    async def test_cancelling_close_caller_does_not_cancel_shared_shutdown(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=1)
        await executor.start()
        release = asyncio.Event()
        started = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            return "done"

        submission = asyncio.create_task(
            executor.submit("a", operation, deadline=time.perf_counter() + 2)
        )
        await started.wait()
        close_caller = asyncio.create_task(executor.close())
        await wait_until(lambda: executor._state == executor._CLOSING)

        close_caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await close_caller

        release.set()
        self.assertEqual(await submission, "done")
        await executor.close()
        self.assertEqual(executor._state, executor._CLOSED)

    async def test_cancelled_failing_close_waiter_can_replay_failure(self):
        executor = FairInferenceExecutor(concurrency=2, queue_capacity=2)
        await executor.start()
        fatal = SystemExit(61)
        active_started = asyncio.Event()
        active_release = asyncio.Event()
        fatal_started = asyncio.Event()
        fatal_release = asyncio.Event()

        async def active_operation():
            active_started.set()
            await active_release.wait()
            return "active-finished"

        async def fatal_operation():
            fatal_started.set()
            await fatal_release.wait()
            raise fatal

        async def capture_fatal():
            try:
                await executor.submit(
                    "fatal",
                    fatal_operation,
                    deadline=time.perf_counter() + 2,
                )
            except SystemExit as caught:
                return caught
            raise AssertionError("fatal submission unexpectedly returned")

        active = asyncio.create_task(
            executor.submit(
                "active",
                active_operation,
                deadline=time.perf_counter() + 2,
            )
        )
        await active_started.wait()
        fatal_submission = asyncio.create_task(capture_fatal())
        await fatal_started.wait()
        close_caller = None

        try:
            fatal_release.set()
            self.assertIs(await fatal_submission, fatal)
            await wait_until(lambda: executor._state == executor._FAILED)

            close_caller = asyncio.create_task(executor.close())
            await wait_until(lambda: executor._state == executor._CLOSING)
            shared_close = executor._close_task
            self.assertIsNotNone(shared_close)

            close_caller.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await close_caller
            self.assertFalse(shared_close.cancelled())

            active_release.set()
            self.assertEqual(await active, "active-finished")
            await wait_until(shared_close.done)
            self.assertEqual(executor._state, executor._CLOSED)

            replay_callers = [
                asyncio.create_task(executor.close()),
                asyncio.create_task(executor.close()),
            ]
            replayed = await asyncio.gather(
                *replay_callers,
                return_exceptions=True,
            )
            self.assertTrue(
                all(isinstance(error, RuntimeError) for error in replayed)
            )
            self.assertIs(replayed[0], replayed[1])
            self.assertRegex(
                str(replayed[0]),
                "inference executor lane failed",
            )
            self.assertIs(replayed[0].__cause__, fatal)
            self.assertIs(executor._close_task, shared_close)

            with self.assertRaises(RuntimeError) as replayed_again:
                await executor.close()
            self.assertIs(replayed_again.exception, replayed[0])
        finally:
            fatal_release.set()
            active_release.set()
            await asyncio.gather(
                active,
                fatal_submission,
                return_exceptions=True,
            )
            if close_caller is not None and not close_caller.done():
                close_caller.cancel()
                await asyncio.gather(close_caller, return_exceptions=True)

    async def test_cancelled_only_close_waiter_observes_shared_failure(self):
        script = textwrap.dedent(
            """
            import asyncio
            import gc
            import time

            from core.infer.executor import FairInferenceExecutor

            fatal = SystemExit(67)

            class SilentLogger:
                def bind(self, **context):
                    return self

                def info(self, message, *args, **kwargs):
                    pass

            async def wait_until(predicate):
                while not predicate():
                    await asyncio.sleep(0)

            async def main():
                executor = FairInferenceExecutor(
                    concurrency=2,
                    queue_capacity=2,
                    logger=SilentLogger(),
                )
                await executor.start()
                active_started = asyncio.Event()
                active_release = asyncio.Event()
                fatal_started = asyncio.Event()
                fatal_release = asyncio.Event()

                async def active_operation():
                    active_started.set()
                    await active_release.wait()
                    return "active-finished"

                async def fatal_operation():
                    fatal_started.set()
                    await fatal_release.wait()
                    raise fatal

                async def capture_fatal():
                    try:
                        await executor.submit(
                            "fatal",
                            fatal_operation,
                            deadline=time.perf_counter() + 2,
                        )
                    except SystemExit as caught:
                        return caught
                    raise AssertionError("fatal submission returned")

                active = asyncio.create_task(
                    executor.submit(
                        "active",
                        active_operation,
                        deadline=time.perf_counter() + 2,
                    )
                )
                await active_started.wait()
                fatal_submission = asyncio.create_task(capture_fatal())
                await fatal_started.wait()
                fatal_release.set()
                assert await fatal_submission is fatal

                close_caller = asyncio.create_task(executor.close())
                await wait_until(lambda: executor._state == executor._CLOSING)
                shared_close = executor._close_task
                assert shared_close is not None
                close_caller.cancel()
                try:
                    await close_caller
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("close caller was not cancelled")

                active_release.set()
                assert await active == "active-finished"
                await wait_until(shared_close.done)
                assert executor._state == executor._CLOSED
                print("shared-close-failed-without-replay")

            asyncio.run(main())
            gc.collect()
            """
        )
        result = subprocess.run(
            [sys.executable, "-W", "error", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "shared-close-failed-without-replay",
            result.stdout,
        )
        self.assertNotIn(
            "Task exception was never retrieved",
            result.stderr,
        )

    async def test_fatal_lane_exception_is_observed_before_close(self):
        script = textwrap.dedent(
            """
            import asyncio
            import time

            from core.infer.executor import FairInferenceExecutor

            async def main():
                executor = FairInferenceExecutor(concurrency=1, queue_capacity=1)
                await executor.start()
                fatal = SystemExit(9)

                async def fatal_operation():
                    raise fatal

                try:
                    await executor.submit(
                        "fatal",
                        fatal_operation,
                        deadline=time.perf_counter() + 2,
                    )
                except SystemExit as caught:
                    assert caught is fatal
                    print(f"caught-system-exit:{caught.code}")
                else:
                    raise AssertionError("submit did not propagate SystemExit")

                assert executor._active_count == 0
                assert all(lane.done() for lane in executor._lanes)
                lane_failure = executor._lanes[0].exception()
                assert isinstance(lane_failure, Exception)
                assert lane_failure.__cause__ is fatal
                try:
                    await executor.close()
                except RuntimeError as close_error:
                    assert close_error.__cause__ is fatal
                    print("close-observed-fatal")
                else:
                    raise AssertionError("close did not report executor fatal")

            asyncio.run(main())
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("caught-system-exit:9", result.stdout)
        self.assertIn("close-observed-fatal", result.stdout)
        self.assertNotIn("Task exception was never retrieved", result.stderr)

    async def test_fatal_logger_is_transferred_to_submit_and_close(self):
        script = textwrap.dedent(
            """
            import asyncio
            import time

            from core.infer.executor import FairInferenceExecutor

            fatal = KeyboardInterrupt("logger stopped")

            class FatalLogger:
                def bind(self, **context):
                    return self

                def info(self, message, *args, **kwargs):
                    raise fatal

            async def main():
                executor = FairInferenceExecutor(
                    concurrency=1,
                    queue_capacity=1,
                    logger=FatalLogger(),
                )
                await executor.start()

                try:
                    await executor.submit(
                        "fatal_logger",
                        lambda: asyncio.sleep(0, result="not returned"),
                        deadline=time.perf_counter() + 2,
                    )
                except KeyboardInterrupt as caught:
                    assert caught is fatal
                    print("caught-logger-fatal")
                else:
                    raise AssertionError("submit returned before logger fatal")

                assert executor._active_count == 0
                assert all(lane.done() for lane in executor._lanes)
                lane_failure = executor._lanes[0].exception()
                assert isinstance(lane_failure, Exception)
                assert lane_failure.__cause__ is fatal
                try:
                    await executor.close()
                except RuntimeError as close_error:
                    assert close_error.__cause__ is fatal
                    print("close-observed-logger-fatal")
                else:
                    raise AssertionError("close did not report logger fatal")

            asyncio.run(main())
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("caught-logger-fatal", result.stdout)
        self.assertIn("close-observed-logger-fatal", result.stdout)
        self.assertNotIn("Task exception was never retrieved", result.stderr)

    async def test_fatal_logger_on_expired_item_is_transferred(self):
        script = textwrap.dedent(
            """
            import asyncio
            import time
            from unittest.mock import patch

            from core.infer.executor import FairInferenceExecutor

            fatal = SystemExit(31)

            class FatalLogger:
                def bind(self, **context):
                    return self

                def info(self, message, *args, **kwargs):
                    raise fatal

            class PausedExecutor(FairInferenceExecutor):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.lane_waiting = asyncio.Event()
                    self.allow_take = asyncio.Event()

                async def _take_next(self):
                    self.lane_waiting.set()
                    await self.allow_take.wait()
                    return await super()._take_next()

            async def wait_until(predicate):
                while not predicate():
                    await asyncio.sleep(0)

            async def main():
                executor = PausedExecutor(
                    concurrency=1,
                    queue_capacity=1,
                    logger=FatalLogger(),
                )
                await executor.start()
                await executor.lane_waiting.wait()
                deadline = time.perf_counter() + 2

                async def submit_and_capture():
                    try:
                        await executor.submit(
                            "expired",
                            lambda: asyncio.sleep(0, result="not called"),
                            deadline=deadline,
                        )
                    except SystemExit as caught:
                        return caught
                    raise AssertionError("submit did not receive logger fatal")

                submission = asyncio.create_task(
                    submit_and_capture()
                )
                await wait_until(lambda: executor._queued_count == 1)

                with patch(
                    "core.infer.executor.time.perf_counter",
                    return_value=deadline + 1,
                ):
                    executor.allow_take.set()
                    caught = await submission
                    assert caught is fatal
                    print("caught-expired-logger-fatal")

                lane_failure = executor._lanes[0].exception()
                assert isinstance(lane_failure, Exception)
                assert lane_failure.__cause__ is fatal
                try:
                    await executor.close()
                except RuntimeError as close_error:
                    assert close_error.__cause__ is fatal
                    print("close-observed-expired-logger-fatal")
                else:
                    raise AssertionError("close did not report logger fatal")

            asyncio.run(main())
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("caught-expired-logger-fatal", result.stdout)
        self.assertIn(
            "close-observed-expired-logger-fatal",
            result.stdout,
        )
        self.assertNotIn("Task exception was never retrieved", result.stderr)

    async def test_fatal_lane_fails_queue_and_drains_other_active_lane(self):
        script = textwrap.dedent(
            """
            import asyncio
            import time

            from core.infer.errors import InferenceUnavailable
            from core.infer.executor import FairInferenceExecutor

            fatal = SystemExit(47)

            class SilentLogger:
                def bind(self, **context):
                    return self

                def info(self, message, *args, **kwargs):
                    pass

            async def wait_until(predicate):
                while not predicate():
                    await asyncio.sleep(0)

            async def main():
                executor = FairInferenceExecutor(
                    concurrency=2,
                    queue_capacity=2,
                    logger=SilentLogger(),
                )
                await executor.start()
                blocker_started = asyncio.Event()
                blocker_release = asyncio.Event()
                fatal_started = asyncio.Event()
                fatal_release = asyncio.Event()
                queued_called = False

                async def blocker():
                    blocker_started.set()
                    await blocker_release.wait()
                    return "active-finished"

                async def fatal_operation():
                    fatal_started.set()
                    await fatal_release.wait()
                    raise fatal

                async def capture_fatal():
                    try:
                        await executor.submit(
                            "fatal",
                            fatal_operation,
                            deadline=time.perf_counter() + 2,
                        )
                    except SystemExit as caught:
                        return caught
                    raise AssertionError("fatal submit unexpectedly returned")

                async def queued_operation():
                    nonlocal queued_called
                    queued_called = True
                    return "queued-ran"

                active = asyncio.create_task(
                    executor.submit(
                        "active",
                        blocker,
                        deadline=time.perf_counter() + 2,
                    )
                )
                await blocker_started.wait()
                fatal_submission = asyncio.create_task(capture_fatal())
                await fatal_started.wait()
                queued = asyncio.create_task(
                    executor.submit(
                        "queued",
                        queued_operation,
                        deadline=time.perf_counter() + 2,
                    )
                )
                await wait_until(lambda: executor._queued_count == 1)

                fatal_release.set()
                caught = await fatal_submission
                assert caught is fatal
                try:
                    await queued
                except InferenceUnavailable:
                    pass
                else:
                    raise AssertionError("queued operation was not failed")
                assert not queued_called
                assert executor._state == executor._FAILED

                try:
                    await executor.submit(
                        "rejected",
                        queued_operation,
                        deadline=time.perf_counter() + 2,
                    )
                except InferenceUnavailable:
                    pass
                else:
                    raise AssertionError("failed executor accepted new work")

                blocker_release.set()
                assert await active == "active-finished"
                await wait_until(
                    lambda: all(lane.done() for lane in executor._lanes)
                )
                try:
                    await executor.close()
                except RuntimeError as close_error:
                    assert close_error.__cause__ is fatal
                else:
                    raise AssertionError("close did not report fatal")
                print("fatal-drained-other-lane")

            asyncio.run(main())
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fatal-drained-other-lane", result.stdout)
        self.assertNotIn("Task exception was never retrieved", result.stderr)

    async def test_observing_lane_exception_does_not_hide_it_from_close(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=1)
        await executor.start()
        original_lane = executor._lanes[0]
        original_lane.cancel()
        await asyncio.gather(original_lane, return_exceptions=True)
        failure = SyntheticBaseFailure("fatal lane")

        async def failed_lane():
            raise failure

        failed_task = asyncio.create_task(failed_lane())
        failed_task.add_done_callback(executor._observe_task_completion)
        await asyncio.sleep(0)
        executor._lanes = [failed_task]

        with self.assertRaisesRegex(RuntimeError, "inference executor lane failed") as caught:
            await executor.close()
        self.assertIs(caught.exception.__cause__, failure)

    async def test_operation_exception_does_not_terminate_lane_and_is_logged(self):
        log = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1, queue_capacity=2, logger=log
        )
        await executor.start()

        async def failing_operation():
            raise ValueError("bad inference")

        async def successful_operation():
            return "ok"

        with self.assertRaisesRegex(ValueError, "bad inference"):
            await executor.submit(
                "bad_model",
                failing_operation,
                deadline=time.perf_counter() + 2,
            )
        self.assertEqual(
            await executor.submit(
                "good_model",
                successful_operation,
                deadline=time.perf_counter() + 2,
            ),
            "ok",
        )
        await executor.close()

        self.assertEqual(len(log.records), 2)
        required_context = {
            "model",
            "executor_queue_wait_ms",
            "executor_queue_depth",
            "executor_active_slots",
            "executor_concurrency",
            "deadline_remaining_ms",
            "status",
            "error_class",
        }
        for context, message, args, kwargs in log.records:
            self.assertTrue(required_context.issubset(context))
            self.assertEqual(
                message,
                "Triton executor completed: status={}, error_class={}",
            )
            self.assertEqual(kwargs, {})
            self.assertEqual(args, (context["status"], context["error_class"] or ""))

        failure_context, success_context = [record[0] for record in log.records]
        self.assertEqual(failure_context["status"], "error")
        self.assertEqual(failure_context["error_class"], "ValueError")
        self.assertEqual(success_context["status"], "success")
        self.assertIsNone(success_context["error_class"])

    async def test_custom_base_exception_reaches_caller_without_losing_lane(self):
        log = RecordingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=2,
            logger=log,
        )
        await executor.start()
        failure = SyntheticBaseFailure("inference aborted")
        body_completed = False

        async def failing_operation():
            raise failure

        try:
            with self.assertRaises(SyntheticBaseFailure) as caught:
                await asyncio.wait_for(
                    executor.submit(
                        "base_failure",
                        failing_operation,
                        deadline=time.perf_counter() + 2,
                    ),
                    timeout=0.1,
                )
            self.assertIs(caught.exception, failure)
            self.assertEqual(executor._active_count, 0)
            self.assertEqual(
                await asyncio.wait_for(
                    executor.submit(
                        "next",
                        lambda: asyncio.sleep(0, result="lane survived"),
                        deadline=time.perf_counter() + 2,
                    ),
                    timeout=0.1,
                ),
                "lane survived",
            )
            body_completed = True
        finally:
            try:
                await executor.close()
            except RuntimeError:
                if body_completed:
                    raise

        failure_records = [
            record for record in log.records if record[0]["model"] == "base_failure"
        ]
        self.assertEqual(len(failure_records), 1)
        self.assert_completion_log(
            failure_records[0],
            model="base_failure",
            status="error",
            error_class="SyntheticBaseFailure",
        )

    async def test_logger_base_exceptions_do_not_change_results_or_kill_lane(self):
        for logger in (CancelledBindLogger(), SyntheticBaseInfoLogger()):
            with self.subTest(logger=type(logger).__name__):
                executor = FairInferenceExecutor(
                    concurrency=1,
                    queue_capacity=1,
                    logger=logger,
                )
                await executor.start()
                try:
                    self.assertEqual(
                        await asyncio.wait_for(
                            executor.submit(
                                "first",
                                lambda: asyncio.sleep(0, result="first"),
                                deadline=time.perf_counter() + 2,
                            ),
                            timeout=0.1,
                        ),
                        "first",
                    )
                    self.assertEqual(
                        await asyncio.wait_for(
                            executor.submit(
                                "second",
                                lambda: asyncio.sleep(0, result="second"),
                                deadline=time.perf_counter() + 2,
                            ),
                            timeout=0.1,
                        ),
                        "second",
                    )
                finally:
                    await executor.close()

    async def test_logger_exception_does_not_affect_results_or_lane(self):
        log = FailingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=1,
            logger=log,
        )
        await executor.start()

        self.assertEqual(
            await executor.submit(
                "a",
                lambda: asyncio.sleep(0, result="first"),
                deadline=time.perf_counter() + 2,
            ),
            "first",
        )
        self.assertEqual(
            await executor.submit(
                "a",
                lambda: asyncio.sleep(0, result="second"),
                deadline=time.perf_counter() + 2,
            ),
            "second",
        )
        await executor.close()
        self.assertEqual(len(log.contexts), 2)

    async def test_logger_exception_preserves_queued_timeout_and_cancellation(self):
        log = FailingLogger()
        executor = FairInferenceExecutor(
            concurrency=1,
            queue_capacity=2,
            logger=log,
        )
        await executor.start()
        release = asyncio.Event()
        blocker_started = asyncio.Event()

        async def blocker():
            blocker_started.set()
            await release.wait()

        deadline = time.perf_counter() + 2
        blocker_task = asyncio.create_task(
            executor.submit("busy", blocker, deadline=deadline)
        )
        await blocker_started.wait()

        with self.assertRaises(InferenceQueueTimeout):
            await executor.submit(
                "timed_out",
                lambda: asyncio.sleep(0),
                deadline=time.perf_counter() + 0.01,
            )

        cancelled_task = asyncio.create_task(
            executor.submit(
                "cancelled",
                lambda: asyncio.sleep(0),
                deadline=deadline,
            )
        )
        await wait_until(lambda: executor._queued_count == 1)
        cancelled_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_task

        release.set()
        await blocker_task
        await executor.close()
        self.assertEqual(
            [
                context["model"]
                for context in log.contexts
                if context["model"] in {"timed_out", "cancelled"}
            ],
            ["timed_out", "cancelled"],
        )

    async def test_cancelled_operation_does_not_reduce_lane_count(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=2)
        await executor.start()

        async def cancelled_operation():
            raise asyncio.CancelledError

        async def successful_operation():
            return "still running"

        with self.assertRaises(asyncio.CancelledError):
            await executor.submit(
                "a", cancelled_operation, deadline=time.perf_counter() + 2
            )
        self.assertEqual(
            await executor.submit(
                "a", successful_operation, deadline=time.perf_counter() + 2
            ),
            "still running",
        )
        await executor.close()

    async def test_cancelling_started_submit_does_not_cancel_operation_or_lane(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=1)
        await executor.start()
        release = asyncio.Event()
        started = asyncio.Event()
        operation_finished = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            operation_finished.set()

        submission = asyncio.create_task(
            executor.submit("a", operation, deadline=time.perf_counter() + 2)
        )
        await started.wait()
        submission.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await submission

        release.set()
        await operation_finished.wait()
        self.assertEqual(
            await executor.submit(
                "a", lambda: asyncio.sleep(0, result="ok"), deadline=time.perf_counter() + 2
            ),
            "ok",
        )
        await executor.close()

    async def test_cancelling_dequeued_submit_still_runs_failing_operation(self):
        log = RecordingLogger()
        executor = PausingBeforeOperationExecutor(
            concurrency=1,
            queue_capacity=1,
            logger=log,
        )
        await executor.start()
        operation_called = asyncio.Event()

        async def failing_operation():
            operation_called.set()
            raise ValueError("abandoned result")

        submission = asyncio.create_task(
            executor.submit(
                "abandoned",
                failing_operation,
                deadline=time.perf_counter() + 2,
            )
        )
        await executor.item_dequeued.wait()
        submission.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await submission

        executor.allow_operation.set()
        try:
            await asyncio.wait_for(operation_called.wait(), timeout=0.1)
            await wait_until(
                lambda: any(
                    record[0]["model"] == "abandoned" for record in log.records
                )
            )
            self.assertEqual(
                await executor.submit(
                    "next",
                    lambda: asyncio.sleep(0, result="lane survived"),
                    deadline=time.perf_counter() + 2,
                ),
                "lane survived",
            )
        finally:
            executor.allow_operation.set()
            await executor.close()

        abandoned_records = [
            record for record in log.records if record[0]["model"] == "abandoned"
        ]
        self.assertEqual(len(abandoned_records), 1)
        self.assert_completion_log(
            abandoned_records[0],
            model="abandoned",
            status="error",
            error_class="ValueError",
        )

    async def test_committed_operation_runs_after_queue_deadline_passes(self):
        log = RecordingLogger()
        executor = PausingBeforeOperationExecutor(
            concurrency=1,
            queue_capacity=1,
            logger=log,
        )
        await executor.start()
        operation_called = False

        async def operation():
            nonlocal operation_called
            operation_called = True
            return "committed result"

        deadline = time.perf_counter() + 0.1
        submission = asyncio.create_task(
            executor.submit("committed", operation, deadline=deadline)
        )
        await executor.item_dequeued.wait()
        self.assertEqual(executor._active_count, 1)
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()) + 0.01)
        executor.allow_operation.set()

        try:
            self.assertEqual(await submission, "committed result")
        finally:
            executor.allow_operation.set()
            await executor.close()

        self.assertTrue(operation_called)
        committed_records = [
            record for record in log.records if record[0]["model"] == "committed"
        ]
        self.assertEqual(len(committed_records), 1)
        self.assertEqual(committed_records[0][0]["status"], "success")
        self.assertIsNone(committed_records[0][0]["error_class"])

    async def test_expired_item_is_rejected_atomically_when_lane_takes_it(self):
        log = RecordingLogger()
        executor = PausingBeforeTakeExecutor(
            concurrency=1,
            queue_capacity=1,
            logger=log,
        )
        await executor.start()
        await executor.lane_waiting.wait()
        operation_called = False

        async def operation():
            nonlocal operation_called
            operation_called = True

        deadline = time.perf_counter() + 2
        submission = asyncio.create_task(
            executor.submit("expired_at_take", operation, deadline=deadline)
        )
        await wait_until(lambda: executor._queued_count == 1)

        with patch(
            "core.infer.executor.time.perf_counter",
            return_value=deadline + 1,
        ):
            executor.allow_take.set()
            with self.assertRaises(InferenceQueueTimeout):
                await submission

        self.assertFalse(operation_called)
        self.assertEqual(executor._active_count, 0)
        self.assertEqual(executor._queued_count, 0)
        await executor.close()
        records = [
            record
            for record in log.records
            if record[0]["model"] == "expired_at_take"
        ]
        self.assertEqual(len(records), 1)
        self.assert_completion_log(
            records[0],
            model="expired_at_take",
            status="error",
            error_class="InferenceQueueTimeout",
        )

    async def test_constructor_requires_positive_non_bool_integers(self):
        invalid_values = (True, False, 0, -1, 1.5, "1", None)
        for value in invalid_values:
            with self.subTest(concurrency=value):
                with self.assertRaises((TypeError, ValueError)):
                    FairInferenceExecutor(concurrency=value, queue_capacity=1)
            with self.subTest(queue_capacity=value):
                with self.assertRaises((TypeError, ValueError)):
                    FairInferenceExecutor(concurrency=1, queue_capacity=value)

    async def test_submit_validates_model_operation_and_deadline(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=1)
        await executor.start()
        operation = lambda: asyncio.sleep(0)
        deadline = time.perf_counter() + 2

        for model in (None, 1, ""):
            with self.subTest(model=model):
                with self.assertRaises((TypeError, ValueError)):
                    await executor.submit(model, operation, deadline=deadline)

        for invalid_operation in (None, 1, "operation"):
            with self.subTest(operation=invalid_operation):
                with self.assertRaises(TypeError):
                    await executor.submit("a", invalid_operation, deadline=deadline)

        for invalid_deadline in (True, False, math.inf, -math.inf, math.nan):
            with self.subTest(deadline=invalid_deadline):
                with self.assertRaises((TypeError, ValueError)):
                    await executor.submit("a", operation, deadline=invalid_deadline)

        await executor.close()

    async def test_deadline_overflow_is_reported_as_parameter_error(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=1)
        await executor.start()
        try:
            with self.assertRaisesRegex(TypeError, "deadline must be finite"):
                await executor.submit(
                    "a",
                    lambda: asyncio.sleep(0),
                    deadline=10**10000,
                )
        finally:
            await executor.close()

    async def test_start_is_idempotent_and_closed_executor_cannot_restart(self):
        executor = FairInferenceExecutor(concurrency=2, queue_capacity=2)
        await executor.start()
        lanes = list(executor._lanes)
        self.assertEqual(
            [lane.get_name() for lane in lanes],
            ["triton-executor-0", "triton-executor-1"],
        )
        await executor.start()
        self.assertEqual(executor._lanes, lanes)
        await executor.close()

        with self.assertRaisesRegex(
            InferenceUnavailable, "inference executor is closed"
        ):
            await executor.start()
        with self.assertRaisesRegex(
            InferenceUnavailable, "inference executor is closed"
        ):
            await executor.submit(
                "a", lambda: asyncio.sleep(0), deadline=time.perf_counter() + 2
            )

    async def test_close_before_start_closes_executor_idempotently(self):
        executor = FairInferenceExecutor(concurrency=1, queue_capacity=1)
        await executor.close()
        await executor.close()
        self.assertEqual(executor._state, executor._CLOSED)


class _Payload:
    """Weak-referenceable stand-in for a request or response buffer."""


class FairInferenceExecutorReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = FairInferenceExecutor(
            concurrency=1, queue_capacity=2, logger=RecordingLogger()
        )
        await self.executor.start()
        self.addAsyncCleanup(self.executor.close)

    async def assert_released_while_idle(self, *references, executor=None):
        executor = executor or self.executor
        await wait_until(lambda: executor._active_count == 0 and executor._queued_count == 0)
        # Drain completion callbacks, without submitting another job or closing lanes.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gc.collect()
        self.assertTrue(all(not lane.done() for lane in executor._lanes))
        for reference in references:
            self.assertIsNone(reference(), "idle executor retained a completed request buffer")

    async def test_success_releases_input_and_result_without_another_request(self):
        async def submit_payload():
            payload = _Payload()

            async def operation(captured=payload):
                self.assertIsNotNone(captured)
                return _Payload()

            result = await self.executor.submit(
                "person_seg", operation, deadline=time.perf_counter() + 2
            )
            return weakref.ref(payload), result

        input_ref, result = await submit_payload()
        result_ref = weakref.ref(result)
        await self.assert_released_while_idle(input_ref)
        self.assertIs(result_ref(), result)
        del result
        await self.assert_released_while_idle(result_ref)

    async def test_operation_failure_releases_input_after_caller_handles_error(self):
        async def submit_payload():
            payload = _Payload()
            failure = RuntimeError("synthetic inference failure")

            async def operation(captured=payload):
                self.assertIsNotNone(captured)
                raise failure

            try:
                await self.executor.submit(
                    "person_seg", operation, deadline=time.perf_counter() + 2
                )
            except RuntimeError as caught:
                self.assertIs(caught, failure)
            else:
                self.fail("operation failure was not propagated")
            return weakref.ref(payload)

        reference = await submit_payload()
        await self.assert_released_while_idle(reference)

    async def test_cancelled_operation_releases_input_without_stopping_lane(self):
        async def submit_payload():
            payload = _Payload()

            async def operation(captured=payload):
                self.assertIsNotNone(captured)
                raise asyncio.CancelledError

            with self.assertRaises(asyncio.CancelledError):
                await self.executor.submit(
                    "person_seg", operation, deadline=time.perf_counter() + 2
                )
            return weakref.ref(payload)

        reference = await submit_payload()
        await self.assert_released_while_idle(reference)

    async def test_caller_cancellation_keeps_active_input_until_operation_finishes(self):
        started = asyncio.Event()
        release = asyncio.Event()
        self.addCleanup(release.set)
        result_refs = []

        def start_payload():
            payload = _Payload()

            async def operation(captured=payload):
                started.set()
                await release.wait()
                self.assertIsNotNone(captured)
                result = _Payload()
                result_refs.append(weakref.ref(result))
                return result

            submission = asyncio.create_task(self.executor.submit(
                "person_seg", operation, deadline=time.perf_counter() + 2
            ))
            return submission, weakref.ref(payload)

        submission, input_ref = start_payload()
        await asyncio.wait_for(started.wait(), timeout=1)
        submission.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await submission
        del submission
        self.assertIsNotNone(input_ref())
        release.set()
        await wait_until(lambda: len(result_refs) == 1)
        await self.assert_released_while_idle(input_ref, result_refs[0])

    async def test_expired_dequeued_input_is_released_before_next_arrival(self):
        executor = PausingBeforeTakeExecutor(
            concurrency=1, queue_capacity=1, logger=RecordingLogger()
        )
        await executor.start()
        self.addAsyncCleanup(executor.close)
        self.addCleanup(executor.allow_take.set)
        await executor.lane_waiting.wait()
        deadline = time.perf_counter() + 30

        def start_payload():
            payload = _Payload()

            async def operation(captured=payload):
                self.fail("expired operation must not run")

            submission = asyncio.create_task(executor.submit(
                "person_seg", operation, deadline=deadline
            ))
            return submission, weakref.ref(payload)

        submission, reference = start_payload()
        await wait_until(lambda: executor._queued_count == 1)
        with patch("core.infer.executor.time.perf_counter", return_value=deadline + 1):
            executor.allow_take.set()
            # assertRaises clears tracebacks and would hide a retained queue_error.
            try:
                await submission
            except InferenceQueueTimeout:
                pass
            else:
                self.fail("expired submission did not raise InferenceQueueTimeout")
        del submission
        await self.assert_released_while_idle(reference, executor=executor)

    async def test_real_scheduler_releases_images_packed_batch_and_wire_outputs(self):
        import numpy as np

        from core.infer.adapters import build_adapters
        from core.infer.batching import ModelBatchScheduler
        from core.infer.capabilities import ModelCapability

        references = []
        batch_sizes = []

        class WeaklyObservedTransport:
            async def infer(self, contract, inputs, *, request_id):
                batch = inputs["RAW_IMAGE"].shape[0]
                batch_sizes.append(batch)
                references.append(weakref.ref(inputs["RAW_IMAGE"]))
                outputs = {
                    "DETECTIONS": np.tile(np.asarray(
                        [[[40, 40, 80, 140, 0.9, 0]]], dtype=np.float32
                    ), (batch, 1, 1)),
                    "MASK_CROPS_PACKED": np.full((batch, 1, 160, 20), 255, dtype=np.uint8),
                    "NUM_DETECTIONS": np.ones((batch, 1), dtype=np.int32),
                }
                references.extend(weakref.ref(value) for value in outputs.values())
                return outputs

        adapter = build_adapters()["person_seg"]
        scheduler = ModelBatchScheduler(
            adapter=adapter,
            capability=ModelCapability(
                key="person_seg", model_name=adapter.contract.model_name,
                api_contract_version=1, server_max_batch=2, target_batch=2,
            ),
            transport=WeaklyObservedTransport(), executor=self.executor,
            wait_ms=100, queue_timeout=2, logger=RecordingLogger(),
        )
        self.addAsyncCleanup(scheduler.close)

        async def submit_image():
            image = np.zeros((200, 120, 3), dtype=np.uint8)
            references.append(weakref.ref(image))
            return await scheduler.submit(adapter.normalize(image))

        results = await asyncio.gather(submit_image(), submit_image())
        await wait_until(lambda: not scheduler._active_dispatches)
        self.assertEqual(batch_sizes, [2])
        self.assertEqual(len(references), 6)
        await self.assert_released_while_idle(*references)
        for segments in results:
            self.assertEqual(segments[0]["maskPixelCount"], 4000)
            self.assertEqual(segments[0]["mask"].count(), 4000)


if __name__ == "__main__":
    unittest.main()
