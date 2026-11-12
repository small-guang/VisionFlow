import asyncio
import unittest

from core.async_tasks import run_owned_parallel


class OwnedParallelTests(unittest.IsolatedAsyncioTestCase):
    async def test_result_order_is_independent_of_completion_order(self):
        second_done = asyncio.Event()

        async def first():
            await second_done.wait()
            return 1

        async def second():
            second_done.set()
            return 2

        self.assertEqual(await run_owned_parallel(first(), second()), [1, 2])
        self.assertEqual(await run_owned_parallel(), [])

    async def test_failure_cancels_sibling_and_preserves_original_exception(self):
        started, stopped = asyncio.Event(), asyncio.Event()
        failure = RuntimeError("prepare failed")

        async def waiting():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def failing():
            await started.wait()
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            await run_owned_parallel(waiting(), failing())
        self.assertIs(caught.exception, failure)
        self.assertTrue(stopped.is_set())

    async def test_repeated_parent_cancellation_waits_for_child_cleanup(self):
        started, cleaning, release, stopped = (asyncio.Event() for _ in range(4))

        async def child():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                stopped.set()

        task = asyncio.create_task(run_owned_parallel(child()))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            await asyncio.wait_for(cleaning.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(stopped.is_set())
        finally:
            release.set()
            task.cancel()
            result = await asyncio.gather(task, return_exceptions=True)
        self.assertIsInstance(result[0], asyncio.CancelledError)
        self.assertTrue(stopped.is_set())

