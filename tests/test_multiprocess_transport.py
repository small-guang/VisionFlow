"""真实socket上的背压、取消和双向RPC，不连接外部服务。"""
import asyncio
import pickle
import socket
import struct
import unittest

from core.multiprocess.transport import RpcPeer, PeerClosed, ProtocolError, RemoteError
from core.multiprocess.frames import RemoteFrameBudget


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def pair(self, owner_handler, visual_handler=None, **options):
        left, right = socket.socketpair()
        async def echo(work, method, payload):
            return payload
        self.owner = await RpcPeer.connect(left, owner_handler, **options)
        self.visual = await RpcPeer.connect(right, visual_handler or echo, **options)
        self.addAsyncCleanup(self.owner.close)
        self.addAsyncCleanup(self.visual.close)
        return self.owner, self.visual

    async def test_bidirectional_calls_and_error_do_not_block_replies(self):
        async def handle(work, method, payload):
            if method == "fail":
                raise ValueError("private connection string")
            result = await self.owner.call("nested", payload, work_id=work)
            return {"work": work, "result": result}
        _, visual = await self.pair(handle, max_requests=1)
        result = await asyncio.wait_for(visual.call("outer", {"x": 1}, work_id="frame"), 1)
        self.assertEqual(result, {"work": "frame", "result": {"x": 1}})
        with self.assertRaisesRegex(RemoteError, "^ValueError$"):
            await visual.call("fail", {})
        self.assertFalse(visual.closed.is_set())

    async def test_waiter_cancel_does_not_release_credit_or_cancel_operation(self):
        entered, release = asyncio.Event(), asyncio.Event()
        count = 0
        async def handle(work, method, payload):
            nonlocal count
            count += 1
            entered.set()
            await release.wait()
            return count
        _, visual = await self.pair(handle, max_requests=1)
        first = asyncio.create_task(visual.call("hold", {}))
        await entered.wait()
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(visual.call("next", {}))
        await asyncio.sleep(0.01)
        self.assertEqual(count, 1)
        self.assertEqual(visual.requests.count, 1)
        release.set()
        await asyncio.wait_for(second, 1)
        self.assertEqual(count, 2)
        self.assertEqual(visual.requests.count, 0)

    async def test_frame_release_progresses_while_acquire_lane_is_saturated(self):
        slots = asyncio.Semaphore(1)
        async def handle(work, method, payload):
            if method == "frame.acquire":
                await slots.acquire()
                return work
            if method == "frame.release":
                slots.release()
        _, visual = await self.pair(handle, max_requests=1)
        await visual.call("frame.acquire", {}, work_id="first")
        acquiring = asyncio.create_task(visual.call("frame.acquire", {}, work_id="second"))
        await asyncio.sleep(0.01)
        self.assertFalse(acquiring.done())
        await asyncio.wait_for(visual.call("frame.release", {}), 1)
        self.assertEqual(await asyncio.wait_for(acquiring, 1), "second")
        await visual.call("frame.release", {})

    async def test_cancelled_frame_acquire_returns_late_lease(self):
        gate, entered = asyncio.Event(), asyncio.Event()
        released = []
        async def handle(work, method, payload):
            if method == "frame.acquire":
                entered.set()
                await gate.wait()
                return "lease"
            released.append(payload["lease_id"])
        _, visual = await self.pair(handle)
        budget = RemoteFrameBudget(visual)
        async def frame():
            async with budget.frame():
                self.fail("cancelled frame entered")
        task = asyncio.create_task(frame())
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(released, ["lease"])
        self.assertEqual(budget.waiting, 0)

    async def test_byte_budget_and_cancel_before_send_preserve_request_sequence(self):
        gate, entered = asyncio.Event(), asyncio.Event()
        async def handle(work, method, payload):
            entered.set()
            await gate.wait()
            return None
        _, visual = await self.pair(handle, max_requests=8, max_bytes=512, max_payload_bytes=512)
        first = asyncio.create_task(visual.call("hold", {"value": b"x" * 300}))
        await entered.wait()
        unsent = asyncio.create_task(visual.call("unsent", {"value": b"x" * 300}))
        await asyncio.sleep(0.01)
        self.assertEqual(visual.requests.count, 1)
        unsent.cancel()
        await asyncio.gather(unsent, return_exceptions=True)
        gate.set()
        await first
        await visual.call("after_cancel", {})
        self.assertFalse(visual.closed.is_set())
        with self.assertRaises(ProtocolError):
            await visual.call("too_large", {"value": b"x" * 1024})

    async def test_disconnect_fails_outgoing_but_drains_accepted_side_effect(self):
        entered, gate, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def handle(work, method, payload):
            entered.set()
            await gate.wait()
            completed.set()
        owner, visual = await self.pair(handle)
        request = asyncio.create_task(visual.call("side_effect", {}))
        await entered.wait()
        await visual.close()
        with self.assertRaises(PeerClosed):
            await request
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0.01)
        self.assertFalse(closing.done())
        gate.set()
        await asyncio.wait_for(closing, 1)
        self.assertTrue(completed.is_set())

    async def test_duplicate_request_rejected_before_second_side_effect(self):
        left, right = socket.socketpair()
        calls = []
        async def handler(work, method, payload):
            calls.append(method)
        owner = await RpcPeer.connect(left, handler)
        self.addAsyncCleanup(owner.close)
        right.setblocking(False)
        data = pickle.dumps((1, "request", 1, "work", "commit", {}), protocol=5)
        packet = struct.pack("!I", len(data)) + data
        await asyncio.get_running_loop().sock_sendall(right, packet + packet)
        await asyncio.wait_for(owner.closed.wait(), 1)
        await owner.close()
        right.close()
        self.assertEqual(calls, ["commit"])
        self.assertIsInstance(owner._error, ProtocolError)

    async def test_cancelled_handler_closes_peer_and_releases_sender_credit(self):
        async def handler(work, method, payload):
            raise asyncio.CancelledError()

        owner, visual = await self.pair(handler, max_requests=1)
        with self.assertRaises(PeerClosed):
            await asyncio.wait_for(visual.call("cancel_handler", {}), .5)
        self.assertTrue(owner.closed.is_set())
        self.assertTrue(visual.closed.is_set())
        self.assertEqual(visual.requests.count, 0)
        self.assertEqual(visual.requests.size, 0)

    async def test_unencodable_error_reply_closes_peer_instead_of_losing_response(self):
        async def handler(work, method, payload):
            raise ConnectionAbortedError("private failure details")

        request_size = len(pickle.dumps((1, "request", 1, "", "x", {}), protocol=5))
        error_size = len(pickle.dumps((1, "response", 1, False, "ConnectionAbortedError"), protocol=5))
        self.assertLess(request_size, error_size)
        owner, visual = await self.pair(
            handler, max_requests=1, max_bytes=4096, max_payload_bytes=request_size,
        )
        with self.assertRaises(PeerClosed):
            await asyncio.wait_for(visual.call("x", {}), .5)
        self.assertTrue(owner.closed.is_set())
        self.assertIsInstance(owner._error, ProtocolError)
        self.assertEqual(visual.requests.count, 0)
        self.assertEqual(visual.requests.size, 0)


if __name__ == "__main__":
    unittest.main()
