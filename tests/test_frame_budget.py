"""帧容量在等待、异常和取消时保持单一所有权。"""
import asyncio
from types import SimpleNamespace
import unittest

from core.admission import FrameBudget


class FrameBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_reservations_bound_capacity_without_claiming_rss_limit(self):
        budget = FrameBudget(8, memory_bytes=25, reservation_bytes=10)
        self.assertEqual(budget.queue_snapshot()["capacity_frames"], 2)
        async with budget.frame() as first, budget.frame() as second:
            first.observe_image(SimpleNamespace(nbytes=30), b"abc")
            second.observe_image(SimpleNamespace(nbytes=4), b"ab")
            snapshot = budget.queue_snapshot()
            self.assertEqual(snapshot["reserved_bytes"], 20)
            self.assertEqual(snapshot["observed_image_bytes"], 39)
            self.assertEqual(snapshot["images_exceeding_reservation"], 1)
            self.assertFalse(snapshot["hard_rss_limit"])
        self.assertEqual(budget.queue_snapshot()["observed_image_bytes"], 0)
        self.assertEqual(budget.queue_snapshot()["reserved_bytes"], 0)

    async def test_cancelled_waiter_does_not_release_another_frames_slot(self):
        budget = FrameBudget(1)
        entered = asyncio.Event()

        async def wait():
            async with budget.frame():
                entered.set()

        async with budget.frame():
            waiter = asyncio.create_task(wait())
            await asyncio.sleep(0)
            self.assertEqual(budget.queue_snapshot()["waiting_frames"], 1)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertEqual(budget.queue_snapshot()["waiting_frames"], 0)
            self.assertEqual(budget.queue_snapshot()["inflight_frames"], 1)
            successor = asyncio.create_task(wait())
            await asyncio.sleep(0)
            self.assertFalse(entered.is_set())
            self.assertFalse(successor.done())
        await asyncio.wait_for(successor, 1)
        self.assertTrue(entered.is_set())
        self.assertEqual(budget.queue_snapshot()["inflight_frames"], 0)

    async def test_cancelled_owner_releases_lease_and_observed_memory(self):
        budget = FrameBudget(1)
        entered = asyncio.Event()

        async def owner():
            async with budget.frame() as lease:
                lease.observe_image(SimpleNamespace(nbytes=100), b"payload")
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(owner())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(budget.queue_snapshot()["inflight_frames"], 0)
        self.assertEqual(budget.queue_snapshot()["observed_image_bytes"], 0)
        async with asyncio.timeout(1):
            async with budget.frame():
                self.assertEqual(budget.queue_snapshot()["inflight_frames"], 1)

    async def test_exception_returns_slot_for_next_frame(self):
        budget = FrameBudget(1)
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            async with budget.frame():
                raise RuntimeError("decode failed")
        async with budget.frame():
            self.assertEqual(budget.queue_snapshot()["inflight_frames"], 1)

    def test_invalid_budgets_are_rejected_before_resources_are_acquired(self):
        for args in (
            {"capacity": 0}, {"capacity": True}, {"capacity": 1.5},
            {"capacity": 1, "memory_bytes": -1},
            {"capacity": 1, "memory_bytes": True},
            {"capacity": 1, "reservation_bytes": 0},
            {"capacity": 1, "memory_bytes": 3, "reservation_bytes": 4},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                FrameBudget(**args)
