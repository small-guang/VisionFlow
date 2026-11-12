"""仅用于本机 spawn socketpair 的有界双向 RPC。

请求和回复各有独立发送额度。读循环不等待业务处理，也不等待发送额度，
因此输出背压不会堵住归还 credit 的回包。pickle 仅跨父子私有 socket 使用，
不暴露网络监听端口；线上方法由两端显式白名单分派。
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from collections import deque
import pickle
import struct


current_work_id: ContextVar[str] = ContextVar("multiprocess_work_id", default="")
VERSION = 1


class PeerClosed(RuntimeError):
    pass


class RemoteError(RuntimeError):
    pass


class ProtocolError(RuntimeError):
    pass


class _Credits:
    def __init__(self, count, size):
        self.limit_count, self.limit_bytes = count, size
        self.count = self.size = 0
        self.condition = asyncio.Condition()
        self.error = None

    async def acquire(self, size):
        async with self.condition:
            await self.condition.wait_for(lambda: self.error or (
                self.count < self.limit_count and self.size + size <= self.limit_bytes
            ))
            if self.error:
                raise self.error
            self.count += 1
            self.size += size

    async def release(self, size):
        async with self.condition:
            self.count -= 1
            self.size -= size
            self.condition.notify_all()

    async def fail(self, error):
        async with self.condition:
            self.error = error
            self.condition.notify_all()


class RpcPeer:
    """call 取消只撤销等待者；已发送操作与额度保留到实际回复或连接终止。"""

    def __init__(self, reader, writer, handler, *, max_requests=128,
                 max_bytes=64 << 20, max_payload_bytes=4 << 20):
        if min(max_requests, max_bytes, max_payload_bytes) < 1 or max_payload_bytes > max_bytes:
            raise ValueError("invalid RPC capacity")
        self.reader, self.writer, self.handler = reader, writer, handler
        self.max_payload = max_payload_bytes
        self.max_requests = max_requests
        self.requests = _Credits(max_requests, max_bytes)
        # 新帧等待不得占满已有帧继续推进/归还租约所需的请求额度。
        self.admissions = _Credits(max_requests, max_bytes)
        self.replies = _Credits(max_requests, max_bytes)
        self._request_queue, self._reply_queue = deque(), deque()
        self._wake = asyncio.Event()
        self._pending = {}
        self._calls = set()
        self._handlers = set()
        self._admission_handlers = set()
        self._sequence = self._received = 0
        self._error = None
        self.closed = asyncio.Event()
        self._reader_task = asyncio.create_task(self._read(), name="ipc-reader")
        self._writer_task = asyncio.create_task(self._write(), name="ipc-writer")
        self.sent_bytes = self.received_bytes = 0

    @classmethod
    async def connect(cls, sock, handler, **options):
        sock.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=sock)
        return cls(reader, writer, handler, **options)

    def _encode(self, message):
        data = pickle.dumps(message, protocol=5)
        if len(data) > self.max_payload:
            raise ProtocolError("IPC payload exceeds configured byte limit")
        return data

    async def call(self, method: str, payload: dict, *, work_id=None):
        # 不在等待credit前另建无限量任务。取得额度前取消没有副作用。
        if self._error:
            raise self._error
        work = current_work_id.get() if work_id is None else work_id
        # 预留整数序号增长的编码空间；取消等待容量不能留下序号空洞。
        provisional = self._encode((VERSION, "request", 0, work, method, payload))
        charged = min(self.max_payload, len(provisional) + 32)
        credits = self.admissions if method == "frame.acquire" else self.requests
        await credits.acquire(charged)
        try:
            request_id = self._sequence + 1
            data = self._encode((VERSION, "request", request_id, work, method, payload))
        except BaseException:
            await credits.release(charged)
            raise
        self._sequence = request_id
        task = asyncio.create_task(self._call(request_id, data, charged, credits), name="ipc-call")
        self._calls.add(task)
        task.add_done_callback(self._finish_call)
        return await asyncio.shield(task)

    def _finish_call(self, task):
        self._calls.discard(task)
        if not task.cancelled():
            task.exception()  # 等待者被取消时也消费真实完成错误。

    async def _call(self, request_id, data, charged, credits):
        try:
            if self._error:
                raise self._error
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            self._request_queue.append(data)
            self._wake.set()
            return await future
        finally:
            self._pending.pop(request_id, None)
            await credits.release(charged)

    async def _write(self):
        try:
            while True:
                await self._wake.wait()
                while self._reply_queue or self._request_queue:
                    reply = bool(self._reply_queue)
                    data = (self._reply_queue if reply else self._request_queue).popleft()
                    self.writer.write(struct.pack("!I", len(data)))
                    self.writer.write(data)
                    await self.writer.drain()
                    self.sent_bytes += len(data)
                    if reply:
                        await self.replies.release(len(data))
                self._wake.clear()
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail(PeerClosed("IPC writer disconnected"))

    async def _read(self):
        try:
            while True:
                length = struct.unpack("!I", await self.reader.readexactly(4))[0]
                if not 0 < length <= self.max_payload:
                    raise ProtocolError("invalid IPC frame length")
                data = await self.reader.readexactly(length)
                self.received_bytes += length
                message = pickle.loads(data)
                if not isinstance(message, tuple) or len(message) < 4 or message[0] != VERSION:
                    raise ProtocolError("incompatible IPC envelope")
                _, kind, request_id, *values = message
                if type(request_id) is not int or request_id <= 0:
                    raise ProtocolError("invalid IPC request id")
                if kind == "response":
                    if len(values) != 2 or type(values[0]) is not bool:
                        raise ProtocolError("invalid IPC response")
                    future = self._pending.get(request_id)
                    if future is None or future.done():
                        raise ProtocolError("unexpected or duplicate IPC reply")
                    ok, result = values
                    if ok:
                        future.set_result(result)
                    else:
                        future.set_exception(RemoteError(str(result)))
                elif kind == "request":
                    if len(values) != 3 or request_id != self._received + 1:
                        raise ProtocolError("duplicate or malformed IPC request")
                    work, method, payload = values
                    if not isinstance(work, str) or not isinstance(method, str) or not isinstance(payload, dict):
                        raise ProtocolError("invalid IPC method payload")
                    handlers = self._admission_handlers if method == "frame.acquire" else self._handlers
                    if len(handlers) >= self.max_requests:
                        raise ProtocolError("peer exceeded request credits")
                    self._received = request_id
                    task = asyncio.create_task(self._handle(request_id, work, method, payload), name="ipc-handler")
                    handlers.add(task)
                    task.add_done_callback(handlers.discard)
                else:
                    raise ProtocolError("unknown IPC envelope")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._fail(error if isinstance(error, ProtocolError) else PeerClosed("IPC reader disconnected"))

    async def _handle(self, request_id, work, method, payload):
        token = current_work_id.set(work)
        try:
            try:
                result = await self.handler(work, method, payload)
                data = self._encode((VERSION, "response", request_id, True, result))
            except Exception as error:
                # 错误只发类型，避免远端异常携带凭据、连接字符串或图像。
                data = self._encode((VERSION, "response", request_id, False, type(error).__name__))
            await self.replies.acquire(len(data))
            self._reply_queue.append(data)
            self._wake.set()
        except asyncio.CancelledError:
            # 接纳后的处理器取消不能静默丢弃回复，否则对端永久持有请求额度。
            await self._fail(PeerClosed("IPC handler cancelled"))
            raise
        except (PeerClosed, ProtocolError) as error:
            # 错误回复也可能超出包上限；无法回复时关闭通道形成明确终态。
            await self._fail(error)
        finally:
            current_work_id.reset(token)

    async def _fail(self, error):
        if self._error is not None:
            return
        self._error = error
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        await self.requests.fail(error)
        await self.admissions.fail(error)
        await self.replies.fail(error)
        self.writer.close()
        self.closed.set()

    async def close(self):
        await self._fail(PeerClosed("IPC peer closed"))
        for task in (self._reader_task, self._writer_task):
            task.cancel()
        await asyncio.gather(self._reader_task, self._writer_task, return_exceptions=True)
        # 已接纳身份/输出不随连接取消；由owner排空后再关闭数据库。
        await asyncio.gather(*self._handlers, *self._admission_handlers, *self._calls, return_exceptions=True)
        try:
            await self.writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass

    async def drain(self):
        """停止新业务后，等待处理器和已排队回包实际写出。"""
        await asyncio.gather(*self._handlers, *self._admission_handlers, return_exceptions=True)
        while (self.replies.count or self._request_queue) and not self.closed.is_set():
            await asyncio.sleep(0.001)

    def queue_snapshot(self):
        return {
            "pending_requests": self.requests.count, "pending_request_bytes": self.requests.size,
            "pending_admissions": self.admissions.count, "pending_admission_bytes": self.admissions.size,
            "queued_replies": self.replies.count, "queued_reply_bytes": self.replies.size,
            "active_handlers": len(self._handlers), "capacity_requests": self.max_requests,
            "capacity_bytes_per_direction": self.requests.limit_bytes,
            "max_payload_bytes": self.max_payload,
            "sent_bytes": self.sent_bytes, "received_bytes": self.received_bytes,
            "closed": self.closed.is_set(),
        }
