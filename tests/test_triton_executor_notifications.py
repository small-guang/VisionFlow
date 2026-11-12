"""执行器通知回归：验证空闲lane只按新增工作唤醒，保留突发并行和取消接力。"""
import asyncio
import time
import unittest

from core.infer.executor import FairInferenceExecutor


class QuietLogger:
    def bind(self, **kwargs): return self
    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass


class WakeCounter(asyncio.Condition):
    def __init__(self):
        super().__init__()
        self.resumed = 0

    async def wait(self):
        result = await super().wait()
        self.resumed += 1
        return result


async def eventually(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("执行器未到达预期状态")


class ExecutorNotificationTests(unittest.IsolatedAsyncioTestCase):
    async def make_executor(self, lanes):
        executor = FairInferenceExecutor(concurrency=lanes, queue_capacity=lanes, logger=QuietLogger())
        executor._condition = WakeCounter()
        await executor.start()
        self.addAsyncCleanup(executor.close)
        await eventually(lambda: len(executor._condition._waiters) == lanes)
        return executor

    async def test_one_request_does_not_wake_other_seventy_nine_idle_lanes(self):
        executor = await self.make_executor(80)
        started, release = asyncio.Event(), asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            return "done"

        submission = asyncio.create_task(executor.submit("single", operation, deadline=time.perf_counter() + 2))
        try:
            await started.wait()
            self.assertEqual(executor._condition.resumed, 1)
            self.assertEqual(executor._active_count, 1)
        finally:
            release.set()
            self.assertEqual(await submission, "done")
        await eventually(lambda: len(executor._condition._waiters) == 80)
        self.assertEqual(executor._condition.resumed, 1)

    async def test_same_turn_burst_uses_every_available_lane(self):
        executor = await self.make_executor(12)
        release = asyncio.Event()
        started = []

        async def operation(index):
            started.append(index)
            await release.wait()
            return index

        submissions = [asyncio.create_task(executor.submit(
            f"model-{index % 3}", lambda index=index: operation(index), deadline=time.perf_counter() + 2))
            for index in range(12)]
        try:
            await eventually(lambda: len(started) == 12)
            self.assertEqual(executor._active_count, 12)
            self.assertEqual(executor._queued_count, 0)
            self.assertEqual(executor._condition.resumed, 12)
        finally:
            release.set()
            self.assertEqual(await asyncio.gather(*submissions), list(range(12)))

    async def test_cancelled_notified_lane_hands_queued_work_to_another_idle_lane(self):
        executor = await self.make_executor(2)

        async def operation():
            return asyncio.current_task().get_name()

        submission = asyncio.create_task(executor.submit("handoff", operation, deadline=time.perf_counter() + 2))
        # 本轮先运行submit，通知产生的lane回调排在当前任务之后，尚未取出item。
        await asyncio.sleep(0)
        self.assertEqual(executor._queued_count, 1)
        executor._lanes[0].cancel()
        self.assertEqual(await asyncio.wait_for(submission, 1), "triton-executor-1")
        self.assertEqual(executor._queued_count, 0)
        self.assertEqual(executor._active_count, 0)


if __name__ == "__main__":
    unittest.main()
