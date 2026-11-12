"""spawn视觉进程监督、全局帧准入与单一身份/输出资源所有权。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import multiprocessing
import socket
from types import SimpleNamespace
import uuid

from core.admission import AdmissionController, FrameBudget
from core.multiprocess.transport import RpcPeer, PeerClosed, current_work_id


def ipc_options(config):
    return dict(max_requests=config.MULTIPROCESS_IPC_MAX_REQUESTS,
                max_bytes=config.MULTIPROCESS_IPC_MAX_BYTES,
                max_payload_bytes=config.MULTIPROCESS_IPC_MAX_PAYLOAD_BYTES)


def create_runtime(config, *, storage=None):
    if getattr(config, "VISUAL_WORKER_PROCESSES", 0):
        if storage is not None:
            raise ValueError("multiprocess storage must be constructed inside visual processes")
        return MultiProcessRuntime(config)
    from core.application import ApplicationRuntime
    return ApplicationRuntime(config, storage=storage)


def _visual_factory(config, peer):
    from core.multiprocess.visual import visual_runtime_factory
    return visual_runtime_factory(config, peer)


def _owner_resources(config):
    from core.multiprocess.assembly import owner_resources
    return owner_resources(config)


def _owner_handler(config, databases, output):
    from core.multiprocess.identity import OwnerIdentityHandler
    return OwnerIdentityHandler(config, databases, output)


def _databases(config, resources):
    from core.application import _databases as create
    return create(config, resources)


def _output(config, databases):
    from core.application import _output as create
    return create(config, databases)


@dataclass(frozen=True)
class OwnerFactories:
    resources: object = _owner_resources
    databases: object = _databases
    output: object = _output
    handler: object = _owner_handler


async def _run_visual(config, sock, factory):
    stopped = asyncio.Event()
    runtime = None

    async def handle(work, method, payload):
        if method == "process" and work:
            if stopped.is_set():
                raise RuntimeError("visual process is stopping")
            return await runtime.process_message(SimpleNamespace(**payload))
        if method == "metrics" and not work:
            return runtime.metrics()
        if method == "stop" and not work:
            stopped.set()
            return None
        raise ValueError("unknown visual method")

    peer = await RpcPeer.connect(sock, handle, **ipc_options(config))
    try:
        runtime = factory(config, peer)
        await runtime.start()
        await peer.call("ready", {}, work_id="")
        stop_wait = asyncio.create_task(stopped.wait())
        close_wait = asyncio.create_task(peer.closed.wait())
        try:
            await asyncio.wait([stop_wait, close_wait], return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_wait.cancel()
            close_wait.cancel()
            await asyncio.gather(stop_wait, close_wait, return_exceptions=True)
        # 正常stop仅在父端所有process完成后发送；先排空回复避免stop ACK丢失。
        await peer.drain()
    finally:
        try:
            if runtime is not None:
                await runtime.close()
        finally:
            await peer.close()


def visual_process_main(config, sock, factory=_visual_factory):
    # spawn不会继承父loop/channel；只有私有socket和只读配置传入。
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import uvloop
    except ImportError:
        pass
    else:
        uvloop.install()
    asyncio.run(_run_visual(config, sock, factory))


@dataclass
class _Worker:
    index: int
    process: object
    pid: int | None = None
    peer: object = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    dead: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    active: int = 0
    monitor: object = None
    metrics: dict = field(default_factory=dict)


class MultiProcessRuntime:
    def __init__(self, config, *, factories=None, visual_factory=_visual_factory, config_splitter=None):
        self.config = config
        self.factories = factories or OwnerFactories()
        self.visual_factory = visual_factory
        self.config_splitter = config_splitter
        capacity = config.ALOG_MAX_INFLIGHT_MESSAGES or config.ALOG_WORKER_COUNT
        self.frames = FrameBudget(config.ALOG_MAX_INFLIGHT_FRAMES or capacity,
                                  memory_bytes=config.FRAME_MEMORY_BUDGET_BYTES,
                                  reservation_bytes=config.FRAME_RESERVATION_BYTES)
        self._slots = asyncio.Semaphore(capacity)
        self.resources = self.databases = self.output = self.identity = self.admission = None
        self.workers = []
        self._works = {}
        self._tasks = set()
        self._failed = asyncio.Event()
        self._failure = None
        self._start_task = self._close_task = self._metrics_task = None
        self._started = False
        self._stopping = False
        self._epoch = uuid.uuid4().hex
        self._sequence = 0

    def _fail(self, error):
        if not self._failed.is_set():
            self._failure = error
            self._failed.set()

    async def wait_failure(self):
        await self._failed.wait()
        raise self._failure

    async def start(self):
        if self._close_task is not None:
            raise RuntimeError("multiprocess runtime is closing")
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._start())
        try:
            return await asyncio.shield(self._start_task)
        except BaseException:
            try:
                await self.close()
            except BaseException:
                pass
            raise

    async def _start(self):
        if self.config_splitter is None:
            from core.multiprocess.assembly import make_visual_config
            self.config_splitter = make_visual_config
        # 先验证全部分账，失败不建立数据库连接。
        configs = [self.config_splitter(self.config, i)
                   for i in range(self.config.VISUAL_WORKER_PROCESSES)]
        if not configs:
            raise ValueError("multiprocess runtime requires visual workers")
        self.resources = self.factories.resources(self.config)
        creation = asyncio.create_task(self.resources.vector_write.run(
            self.factories.databases, self.config, self.resources))
        try:
            self.databases = await asyncio.shield(creation)
        except asyncio.CancelledError:
            while not creation.done():
                try:
                    await asyncio.shield(creation)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not creation.cancelled() and creation.exception() is None:
                self.databases = creation.result()
            raise
        self.output = self.factories.output(self.config, self.databases)
        await self.output.start()
        self.identity = self.factories.handler(self.config, self.databases, self.output)
        self.admission = AdmissionController(self.output)
        context = multiprocessing.get_context("spawn")
        loop = asyncio.get_running_loop()
        for index, config in enumerate(configs):
            parent_sock, child_sock = socket.socketpair()
            process = context.Process(target=visual_process_main,
                                      args=(config, child_sock, self.visual_factory),
                                      name=f"alog-visual-{index + 1}")
            worker = _Worker(index, process)
            try:
                process.start()
            except BaseException:
                parent_sock.close()
                raise
            finally:
                child_sock.close()
            self.workers.append(worker)
            worker.pid = process.pid
            loop.add_reader(process.sentinel, self._process_exited, worker)
            worker.peer = await RpcPeer.connect(parent_sock,
                lambda work, method, payload, w=worker: self._handle(w, work, method, payload),
                **ipc_options(self.config))
            worker.monitor = asyncio.create_task(self._watch(worker), name=f"visual-monitor-{index}")
        ready = asyncio.gather(*(worker.ready.wait() for worker in self.workers))
        failure = asyncio.create_task(self._failed.wait())
        try:
            async with asyncio.timeout(self.config.MULTIPROCESS_START_TIMEOUT_SECONDS):
                done, _ = await asyncio.wait([ready, failure], return_when=asyncio.FIRST_COMPLETED)
                if failure in done:
                    raise self._failure
                await ready
        finally:
            ready.cancel()
            failure.cancel()
            await asyncio.gather(ready, failure, return_exceptions=True)
        self._started = True
        self._metrics_task = asyncio.create_task(self._sample_metrics(), name="visual-metrics")
        return self

    def _process_exited(self, worker):
        # sentinel死亡后持续可读；必须立即移除，避免排空owner操作期间空转。
        asyncio.get_running_loop().remove_reader(worker.process.sentinel)
        worker.dead.set()

    async def _watch(self, worker):
        disconnected = asyncio.create_task(worker.peer.closed.wait())
        dead = asyncio.create_task(worker.dead.wait())
        try:
            await asyncio.wait([disconnected, dead], return_when=asyncio.FIRST_COMPLETED)
            if not self._stopping:
                self._fail(RuntimeError(f"visual worker {worker.index + 1} exited or disconnected"))
                # 一名视觉进程失联后停止整组，避免入口仍接纳而无法兑现全局额度。
                for sibling in self.workers:
                    if sibling is not worker and sibling.process.is_alive():
                        sibling.process.terminate()
            if not worker.dead.is_set():
                # 通道已丢失，无法安全继续副作用；终止并确认进程退出才回收资源。
                worker.process.terminate()
                try:
                    await asyncio.wait_for(worker.dead.wait(), 5)
                except asyncio.TimeoutError:
                    worker.process.kill()
                    await worker.dead.wait()
            worker.process.join(timeout=0)
            await worker.peer.close()
        finally:
            disconnected.cancel()
            dead.cancel()
            await asyncio.gather(disconnected, dead, return_exceptions=True)
            worker.stopped.set()

    async def _handle(self, worker, work_id, method, payload):
        if method == "ready" and not work_id:
            worker.ready.set()
            return None
        state = self._works.get(work_id)
        if state is None or state["worker"] is not worker:
            raise ValueError("expired or foreign work")
        if method == "frame.acquire":
            if state["lease"] is not None or state["acquiring"]:
                raise ValueError("work already owns a frame")
            state["acquiring"] = True
            scope = self.frames.frame()
            acquire = asyncio.create_task(scope.__aenter__())
            failed = asyncio.create_task(self._failed.wait())
            try:
                done, _ = await asyncio.wait([acquire, failed], return_when=asyncio.FIRST_COMPLETED)
                if failed in done:
                    if acquire.done() and not acquire.cancelled() and acquire.exception() is None:
                        await scope.__aexit__(None, None, None)
                    else:
                        acquire.cancel()
                        await asyncio.gather(acquire, return_exceptions=True)
                    raise PeerClosed("worker runtime failed")
                lease = await acquire
                token = uuid.uuid4().hex
                state["lease"] = (token, scope, lease)
                return token
            finally:
                state["acquiring"] = False
                failed.cancel()
                await asyncio.gather(failed, return_exceptions=True)
        if method == "frame.release":
            owned = state["lease"]
            if owned is None or owned[0] != payload["lease_id"]:
                raise ValueError("unknown frame lease")
            owned[2].image_bytes = max(0, int(payload["observed_image_bytes"]))
            await owned[1].__aexit__(None, None, None)
            state["lease"] = None
            return None
        if method == "frame.complete":
            from core.performance import runtime_performance
            await self.identity.validate_frame_complete(
                worker.index, work_id, payload["uid"], payload["is_pano"])
            runtime_performance.image_completed(payload["topic"], uid=payload["uid"], is_pano=payload["is_pano"])
            return None
        return await self.identity.handle(worker.index, work_id, method, payload)

    async def process_message(self, message):
        if not self._started or self._stopping:
            raise RuntimeError("multiprocess runtime is not accepting work")
        if self._failed.is_set():
            raise self._failure
        await self._slots.acquire()
        task = asyncio.create_task(self._dispatch(message), name="visual-work")
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)

    def _finished(self, task):
        self._tasks.discard(task)
        self._slots.release()
        if not task.cancelled():
            task.exception()

    async def _dispatch(self, message):
        if self._failed.is_set() or self._stopping:
            raise self._failure or RuntimeError("runtime is stopping")
        worker = min(self.workers, key=lambda candidate: candidate.active)
        worker.active += 1
        self._sequence += 1
        work_id = f"{self._epoch}:{self._sequence}"
        state = {"worker": worker, "lease": None, "acquiring": False}
        self._works[work_id] = state
        try:
            await worker.peer.call("process", {
                "topic": getattr(message, "topic", ""), "partition": getattr(message, "partition", 0),
                "offset": getattr(message, "offset", 0), "value": message.value,
            }, work_id=work_id)
            from core.performance import runtime_performance
            runtime_performance.message_completed(getattr(message, "topic", ""))
        finally:
            if worker.peer.closed.is_set():
                await worker.stopped.wait()
            try:
                await self.identity.close_work(worker.index, work_id)
            finally:
                if state["lease"] is not None:
                    await state["lease"][1].__aexit__(None, None, None)
                self._works.pop(work_id, None)
                worker.active -= 1

    async def _sample_metrics(self):
        while True:
            await asyncio.sleep(5)
            for worker in self.workers:
                if not worker.peer.closed.is_set():
                    try:
                        worker.metrics = await worker.peer.call("metrics", {}, work_id="")
                    except Exception:
                        return

    def queue_snapshot(self):
        return {"mode": "shared_identity", "failed": self._failed.is_set(),
                "active_work": len(self._works),
                "workers": [{"index": w.index, "pid": w.pid, "active_work": w.active,
                             "alive": not w.dead.is_set(), "ipc": w.peer.queue_snapshot(),
                             "queues": w.metrics} for w in self.workers if w.peer is not None]}

    def metric_providers(self):
        return {"resources": self.resources.queue_snapshot, "frames": self.frames.queue_snapshot,
                "output": self.output.queue_snapshot, "multiprocess": self.queue_snapshot}

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="multiprocess-close")
        await asyncio.shield(self._close_task)

    async def _close(self):
        self._stopping = True
        if self._start_task and not self._start_task.done():
            self._start_task.cancel()
            await asyncio.gather(self._start_task, return_exceptions=True)
        if self._metrics_task:
            self._metrics_task.cancel()
            await asyncio.gather(self._metrics_task, return_exceptions=True)
        errors = []
        try:
            async with asyncio.timeout(self.config.MULTIPROCESS_STOP_TIMEOUT_SECONDS):
                if self._failed.is_set():
                    raise self._failure
                await asyncio.gather(*(asyncio.shield(t) for t in list(self._tasks)), return_exceptions=True)
                await asyncio.gather(*(w.peer.call("stop", {}, work_id="") for w in self.workers
                                       if w.peer is not None and not w.peer.closed.is_set()))
                await asyncio.gather(*(w.stopped.wait() for w in self.workers))
        except Exception as error:
            errors.append(error)
            self._fail(error)
            for worker in self.workers:
                if worker.process.is_alive():
                    worker.process.terminate()
            for worker in self.workers:
                try:
                    await asyncio.wait_for(worker.dead.wait(), 5)
                except asyncio.TimeoutError:
                    worker.process.kill()
                    await worker.dead.wait()
                if worker.monitor is None or worker.monitor.done():
                    if worker.peer:
                        await worker.peer.close()
                    worker.stopped.set()
            await asyncio.gather(*(w.stopped.wait() for w in self.workers))
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            for worker in self.workers:
                try:
                    if worker.monitor:
                        await worker.monitor
                except BaseException as error:
                    errors.append(error)
                try:
                    if worker.peer is not None:
                        await worker.peer.close()
                except BaseException as error:
                    errors.append(error)
                try:
                    asyncio.get_running_loop().remove_reader(worker.process.sentinel)
                    worker.process.join(timeout=0)
                    worker.process.close()
                except BaseException as error:
                    errors.append(error)
            for resource, method in ((self.output, "stop"), (self.resources, "close")):
                if resource is not None:
                    try:
                        await getattr(resource, method)()
                    except BaseException as error:
                        # 下游清理任务自身取消也必须继续归还其余 owner 资源。
                        errors.append(error)
            if self.databases is not None:
                try:
                    self.databases.close()
                except Exception as error:
                    errors.append(error)
            self._started = False
        if errors:
            raise RuntimeError("multiprocess runtime cleanup failed") from errors[0]

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            await self.close()
        except BaseException:
            if exc is None:
                raise
