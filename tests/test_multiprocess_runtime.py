"""真实 spawn 与本机 socket，使用纯 fake 服务验证生命周期和全局容量。"""
from __future__ import annotations

import asyncio
import multiprocessing
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from core.multiprocess.frames import RemoteFrameBudget
from core.multiprocess.runtime import MultiProcessRuntime, OwnerFactories, _run_visual


def runtime_config(**overrides):
    values = dict(
        VISUAL_WORKER_PROCESSES=2,
        ALOG_WORKER_COUNT=4, ALOG_MAX_INFLIGHT_MESSAGES=4,
        ALOG_MAX_INFLIGHT_FRAMES=1, FRAME_MEMORY_BUDGET_BYTES=0,
        FRAME_RESERVATION_BYTES=64,
        MULTIPROCESS_IPC_MAX_REQUESTS=8,
        MULTIPROCESS_IPC_MAX_BYTES=1 << 20,
        MULTIPROCESS_IPC_MAX_PAYLOAD_BYTES=1 << 16,
        MULTIPROCESS_START_TIMEOUT_SECONDS=10,
        MULTIPROCESS_STOP_TIMEOUT_SECONDS=2,
        fail_start_index=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def split_fake_config(config, index):
    return SimpleNamespace(**vars(config), MP_WORKER_INDEX=index)


class FakeVisual:
    def __init__(self, config, peer):
        self.config, self.peer = config, peer
        self.frames = RemoteFrameBudget(peer)

    async def start(self):
        if self.config.fail_start_index == self.config.MP_WORKER_INDEX:
            raise RuntimeError("intentional visual startup failure")

    async def process_message(self, message):
        payload = message.value
        async with self.frames.frame():
            await self.peer.call("test.entered", {
                "label": payload["label"], "pid": os.getpid(),
                "start_method": multiprocessing.get_start_method(),
            })
            action = payload.get("action", "normal")
            if action == "crash":
                os._exit(23)
            if action == "abandon_owner_operation":
                asyncio.create_task(self.peer.call("test.gate", {"name": payload["gate"]}))
                await self.peer.call("test.await_gate", {"name": payload["gate"]})
                os._exit(24)
            if action == "disconnect":
                self.peer.writer.close()
                await asyncio.sleep(3600)
            if action == "hang":
                await asyncio.sleep(3600)
            if payload.get("gate"):
                await self.peer.call("test.gate", {"name": payload["gate"]})
            await self.peer.call("test.finished", {"label": payload["label"]})
        return {"message_completed": 1}

    def metrics(self):
        return {"frames": self.frames.queue_snapshot()}

    async def close(self):
        pass


def fake_visual_factory(config, peer):
    return FakeVisual(config, peer)


class ExplodingMonitorRuntime(MultiProcessRuntime):
    async def _watch(self, worker):
        await super()._watch(worker)
        if worker.index == 0:
            raise OSError("intentional monitor cleanup failure")


class FakeOwner:
    def __init__(self):
        self.events = []
        self.entered = asyncio.Queue()
        self.gates = {}
        self.gate_entered = {}
        self.closed_works = []
        self.runtime = None
        resources = SimpleNamespace(
            vector_write=SimpleNamespace(run=self.run_creation),
            close=self.close_resources, queue_snapshot=lambda: {},
        )
        databases = SimpleNamespace(close=self.close_databases)
        output = SimpleNamespace(start=self.start_output, stop=self.stop_output,
                                 wait_for_capacity=self.wait_for_capacity,
                                 queue_snapshot=lambda: {})
        self.factories = OwnerFactories(
            resources=lambda config: resources,
            databases=lambda config, resources: databases,
            output=lambda config, databases: output,
            handler=lambda config, databases, output: self,
        )

    async def run_creation(self, factory, *args):
        return factory(*args)

    async def start_output(self):
        self.events.append("output.start")

    async def stop_output(self):
        self.events.append("output.stop")

    async def close_resources(self):
        self.events.append("resources.close")

    def close_databases(self):
        self.events.append("databases.close")
        if self.events[-2] != "resources.close":
            raise AssertionError("database closed before resource pool")

    async def wait_for_capacity(self):
        pass

    async def handle(self, worker, work, method, payload):
        if method == "test.entered":
            await self.entered.put(payload | {"worker": worker, "work": work})
        elif method == "test.gate":
            self.gate_entered.setdefault(payload["name"], asyncio.Event()).set()
            await self.gates[payload["name"]].wait()
            self.events.append("gate.finished." + payload["name"])
        elif method == "test.await_gate":
            await self.gate_entered.setdefault(payload["name"], asyncio.Event()).wait()
        elif method == "test.finished":
            self.events.append("finished." + payload["label"])
        else:
            raise ValueError(method)

    async def close_work(self, worker, work):
        self.closed_works.append({
            "worker": worker, "work": work,
            "worker_dead": self.runtime.workers[worker].dead.is_set(),
        })


class MultiProcessRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def make_runtime(self, **overrides):
        fake = FakeOwner()
        runtime = MultiProcessRuntime(
            runtime_config(**overrides), factories=fake.factories,
            visual_factory=fake_visual_factory, config_splitter=split_fake_config,
        )
        fake.runtime = runtime
        self.addAsyncCleanup(self.close_runtime, runtime)
        await runtime.start()
        return runtime, fake

    async def close_runtime(self, runtime):
        try:
            await asyncio.wait_for(runtime.close(), 15)
        except RuntimeError:
            if runtime._close_task is None or not runtime._close_task.done():
                raise

    def message(self, label, **fields):
        return SimpleNamespace(topic="unit-test", partition=0, offset=0,
                               value={"label": label, **fields})

    async def await_drained(self, runtime):
        async with asyncio.timeout(10):
            while runtime._tasks:
                await asyncio.sleep(0.01)

    async def test_spawn_shares_one_global_frame_capacity_and_keeps_owner_services_single(self):
        runtime, fake = await self.make_runtime(MULTIPROCESS_IPC_MAX_REQUESTS=2)
        fake.gates["first"] = asyncio.Event()
        first = asyncio.create_task(runtime.process_message(self.message("first", gate="first")))
        observed = await asyncio.wait_for(fake.entered.get(), 10)
        tail = [asyncio.create_task(runtime.process_message(self.message(f"tail-{i}"))) for i in range(3)]
        try:
            async with asyncio.timeout(5):
                while not runtime.frames.queue_snapshot()["waiting_frames"]:
                    await asyncio.sleep(0.01)
            self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 1)
            self.assertTrue(fake.entered.empty())
            self.assertNotEqual(observed["pid"], os.getpid())
            self.assertEqual(observed["start_method"], "spawn")
        finally:
            fake.gates["first"].set()
        await asyncio.wait_for(asyncio.gather(first, *tail), 10)
        observed_tail = [fake.entered.get_nowait() for _ in range(3)]
        self.assertEqual(len({item["pid"] for item in [observed, *observed_tail]}), 2)
        self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 0)
        self.assertEqual(runtime.frames.queue_snapshot()["waiting_frames"], 0)
        self.assertEqual(len(fake.closed_works), 4)
        self.assertFalse(runtime._works)
        await runtime.close()
        self.assertEqual(fake.events[-3:], ["output.stop", "resources.close", "databases.close"])
        self.assertEqual(fake.events.count("output.start"), 1)
        self.assertTrue(all(worker.dead.is_set() and worker.stopped.is_set() for worker in runtime.workers))

    async def test_one_three_and_four_visuals_keep_global_leases_and_close_the_whole_group(self):
        for count in (1, 3, 4):
            with self.subTest(count=count):
                runtime, fake = await self.make_runtime(VISUAL_WORKER_PROCESSES=count)
                fake.gates["first"] = asyncio.Event()
                first = asyncio.create_task(runtime.process_message(self.message("first", gate="first")))
                observed = await asyncio.wait_for(fake.entered.get(), 10)
                tail = [asyncio.create_task(runtime.process_message(self.message(f"tail-{i}")))
                        for i in range(count)]
                try:
                    async with asyncio.timeout(5):
                        while not runtime.frames.queue_snapshot()["waiting_frames"]:
                            await asyncio.sleep(0.01)
                    self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 1)
                    self.assertTrue(fake.entered.empty())
                finally:
                    fake.gates["first"].set()
                await asyncio.wait_for(asyncio.gather(first, *tail), 10)
                observed_tail = [fake.entered.get_nowait() for _ in range(count)]
                self.assertEqual(len(runtime.workers), count)
                self.assertEqual({item["worker"] for item in [observed, *observed_tail]}, set(range(count)))
                self.assertEqual(len({item["pid"] for item in [observed, *observed_tail]}), count)
                self.assertTrue(all(item["start_method"] == "spawn" for item in [observed, *observed_tail]))
                self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 0)
                self.assertEqual(runtime.frames.queue_snapshot()["waiting_frames"], 0)
                self.assertEqual(len(fake.closed_works), count + 1)
                self.assertFalse(runtime._works)
                await runtime.close()
                self.assertEqual(fake.events.count("output.start"), 1)
                self.assertEqual(fake.events[-3:], ["output.stop", "resources.close", "databases.close"])
                self.assertTrue(all(worker.dead.is_set() and worker.stopped.is_set() for worker in runtime.workers))

    async def test_fourth_visual_startup_failure_closes_every_started_worker(self):
        fake = FakeOwner()
        runtime = MultiProcessRuntime(
            runtime_config(VISUAL_WORKER_PROCESSES=4, fail_start_index=3),
            factories=fake.factories, visual_factory=fake_visual_factory,
            config_splitter=split_fake_config,
        )
        fake.runtime = runtime
        self.addAsyncCleanup(self.close_runtime, runtime)
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(runtime.start(), 15)
        self.assertEqual(len(runtime.workers), 4)
        self.assertTrue(all(worker.dead.is_set() and worker.stopped.is_set() for worker in runtime.workers))
        self.assertEqual(fake.events[-3:], ["output.stop", "resources.close", "databases.close"])

    async def test_caller_cancellation_keeps_lease_and_work_owned_until_real_completion(self):
        runtime, fake = await self.make_runtime()
        fake.gates["release"] = asyncio.Event()
        caller = asyncio.create_task(runtime.process_message(self.message("cancelled-waiter", gate="release")))
        await asyncio.wait_for(fake.entered.get(), 10)
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 1)
        self.assertEqual(len(runtime._works), 1)
        self.assertFalse(fake.closed_works)
        fake.gates["release"].set()
        await self.await_drained(runtime)
        self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 0)
        self.assertEqual(len(fake.closed_works), 1)
        self.assertIn("finished.cancelled-waiter", fake.events)
        await runtime.close()

    async def test_child_crash_releases_frame_only_after_actual_process_exit(self):
        runtime, fake = await self.make_runtime()
        work = asyncio.create_task(runtime.process_message(self.message("crash", action="crash")))
        await asyncio.wait_for(fake.entered.get(), 10)
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(runtime.wait_failure(), 10)
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(work, 10)
        self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 0)
        self.assertEqual(len(fake.closed_works), 1)
        self.assertTrue(fake.closed_works[0]["worker_dead"])
        with self.assertRaises(RuntimeError):
            await runtime.process_message(self.message("after-crash"))
        with self.assertRaises(RuntimeError):
            await runtime.close()
        self.assertTrue(all(worker.dead.is_set() for worker in runtime.workers))
        self.assertFalse(runtime._works)

    async def test_disconnected_live_child_is_terminated_before_lease_recovery(self):
        runtime, fake = await self.make_runtime()
        work = asyncio.create_task(runtime.process_message(self.message("disconnect", action="disconnect")))
        await asyncio.wait_for(fake.entered.get(), 10)
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(work, 10)
        self.assertTrue(fake.closed_works[0]["worker_dead"])
        self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 0)
        with self.assertRaises(RuntimeError):
            await runtime.close()

    async def test_child_death_does_not_abandon_already_accepted_owner_operation(self):
        runtime, fake = await self.make_runtime()
        fake.gates["persist"] = asyncio.Event()
        work = asyncio.create_task(runtime.process_message(self.message(
            "inflight-owner-operation", action="abandon_owner_operation", gate="persist",
        )))
        await asyncio.wait_for(fake.entered.get(), 10)
        try:
            with self.assertRaises(RuntimeError):
                await asyncio.wait_for(runtime.wait_failure(), 10)
            self.assertFalse(work.done())
            self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 1)
            self.assertFalse(fake.closed_works)
            self.assertNotIn("databases.close", fake.events)
        finally:
            fake.gates["persist"].set()
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(work, 10)
        self.assertIn("gate.finished.persist", fake.events)
        self.assertTrue(fake.closed_works[0]["worker_dead"])
        self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 0)
        with self.assertRaises(RuntimeError):
            await runtime.close()

    async def test_shutdown_timeout_terminates_children_and_drains_owner_before_database_close(self):
        runtime, fake = await self.make_runtime(MULTIPROCESS_STOP_TIMEOUT_SECONDS=0.1)
        work = asyncio.create_task(runtime.process_message(self.message("hung", action="hang")))
        await asyncio.wait_for(fake.entered.get(), 10)
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(runtime.close(), 10)
        with self.assertRaises(RuntimeError):
            await work
        self.assertTrue(fake.closed_works[0]["worker_dead"])
        self.assertEqual(runtime.frames.queue_snapshot()["inflight_frames"], 0)
        self.assertFalse(runtime._tasks)
        self.assertEqual(fake.events[-3:], ["output.stop", "resources.close", "databases.close"])

    async def test_partial_startup_failure_closes_both_spawned_children_and_owner_resources(self):
        fake = FakeOwner()
        runtime = MultiProcessRuntime(
            runtime_config(fail_start_index=1), factories=fake.factories,
            visual_factory=fake_visual_factory, config_splitter=split_fake_config,
        )
        fake.runtime = runtime
        self.addAsyncCleanup(self.close_runtime, runtime)
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(runtime.start(), 15)
        self.assertEqual(fake.events[-3:], ["output.stop", "resources.close", "databases.close"])
        self.assertTrue(all(worker.dead.is_set() and worker.stopped.is_set() for worker in runtime.workers))
        self.assertFalse(runtime._works)

    async def test_metrics_remain_readable_after_process_handles_have_been_closed(self):
        runtime, fake = await self.make_runtime()
        pids = [worker.process.pid for worker in runtime.workers]
        await runtime.close()
        snapshot = runtime.queue_snapshot()
        self.assertEqual([worker["pid"] for worker in snapshot["workers"]], pids)
        self.assertTrue(all(not worker["alive"] for worker in snapshot["workers"]))
        self.assertEqual(snapshot["active_work"], 0)
        self.assertEqual(fake.events[-1], "databases.close")

    async def test_monitor_cleanup_error_does_not_skip_other_children_or_owner_cleanup(self):
        fake = FakeOwner()
        runtime = ExplodingMonitorRuntime(
            runtime_config(), factories=fake.factories,
            visual_factory=fake_visual_factory, config_splitter=split_fake_config,
        )
        fake.runtime = runtime
        self.addAsyncCleanup(self.close_runtime, runtime)
        await runtime.start()
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(runtime.close(), 10)
        self.assertTrue(all(worker.dead.is_set() and worker.stopped.is_set() for worker in runtime.workers))
        self.assertTrue(all(worker.process._closed for worker in runtime.workers))
        self.assertEqual(fake.events[-3:], ["output.stop", "resources.close", "databases.close"])

    async def test_dead_process_sentinel_reader_is_removed_before_slow_owner_drain(self):
        runtime, fake = await self.make_runtime()
        fake.gates["persist"] = asyncio.Event()
        sentinel = runtime.workers[0].process.sentinel
        loop = asyncio.get_running_loop()
        with patch.object(loop, "remove_reader", wraps=loop.remove_reader) as removed:
            work = asyncio.create_task(runtime.process_message(self.message(
                "slow-owner", action="abandon_owner_operation", gate="persist",
            )))
            await asyncio.wait_for(fake.entered.get(), 10)
            try:
                await asyncio.wait_for(runtime.workers[0].dead.wait(), 10)
                await asyncio.sleep(0)
                self.assertIn(((sentinel,), {}), [(call.args, call.kwargs) for call in removed.call_args_list])
                self.assertFalse(runtime.workers[0].stopped.is_set())
                self.assertFalse(work.done())
            finally:
                fake.gates["persist"].set()
            with self.assertRaises(RuntimeError):
                await asyncio.wait_for(work, 10)
            with self.assertRaises(RuntimeError):
                await runtime.close()

    async def test_cancelled_owner_cleanup_still_closes_remaining_resources_and_database(self):
        for failing_resource in ("output", "resources"):
            with self.subTest(failing_resource=failing_resource):
                fake = FakeOwner()
                runtime = MultiProcessRuntime(runtime_config(), factories=fake.factories)
                runtime.resources = fake.factories.resources(runtime.config)
                runtime.databases = fake.factories.databases(runtime.config, runtime.resources)
                runtime.output = fake.factories.output(runtime.config, runtime.databases)
                original = (runtime.output.stop if failing_resource == "output"
                            else runtime.resources.close)

                async def cancelled_cleanup():
                    await original()
                    raise asyncio.CancelledError()

                if failing_resource == "output":
                    runtime.output.stop = cancelled_cleanup
                else:
                    runtime.resources.close = cancelled_cleanup
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    await runtime.close()
                self.assertFalse(runtime._close_task.cancelled())
                self.assertEqual(fake.events, ["output.stop", "resources.close", "databases.close"])

    async def test_visual_cleanup_failure_still_closes_ipc_peer(self):
        for error_type in (RuntimeError, asyncio.CancelledError):
            with self.subTest(error_type=error_type.__name__):
                closed = asyncio.Event()
                closed.set()
                peer = SimpleNamespace(closed=closed, call=AsyncMock(), drain=AsyncMock(), close=AsyncMock())
                runtime = SimpleNamespace(start=AsyncMock(), close=AsyncMock(side_effect=error_type()))
                with patch("core.multiprocess.runtime.RpcPeer.connect", AsyncMock(return_value=peer)):
                    with self.assertRaises(error_type):
                        await _run_visual(runtime_config(), object(), lambda *_: runtime)
                runtime.close.assert_awaited_once()
                peer.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
