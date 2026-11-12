from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from loguru import logger as default_logger

from core.infer.errors import InferenceQueueTimeout, InferenceUnavailable


T = TypeVar("T")


class _LaneFatalError(Exception):
    pass


@dataclass(eq=False)
class _QueuedOperation:
    model: str
    operation: Callable[[], Awaitable[Any]]
    deadline: float
    enqueued_at: float
    future: asyncio.Future[Any]
    started: bool = False


class FairInferenceExecutor:
    _NEW = "new"
    _OPEN = "open"
    _FAILED = "failed"
    _CLOSING = "closing"
    _CLOSED = "closed"

    def __init__(
        self,
        *,
        concurrency: int,
        queue_capacity: int,
        logger: object | None = None,
    ):
        self._validate_positive_integer("concurrency", concurrency)
        self._validate_positive_integer("queue_capacity", queue_capacity)
        self.concurrency = concurrency
        self.queue_capacity = queue_capacity
        self._logger = logger if logger is not None else default_logger
        self._condition = asyncio.Condition()
        self._queues: dict[str, deque[_QueuedOperation]] = {}
        self._ready_models: deque[str] = deque()
        self._queued_count = 0
        self._active_count = 0
        self._active_by_model: dict[str, int] = {}
        self._state = self._NEW
        self._lanes: list[asyncio.Task[None]] = []
        self._close_task: asyncio.Task[None] | None = None

    def queue_snapshot(self) -> dict[str, Any]:
        """在所属事件循环同步读取批次计数，不触发调度或网络请求。

        active 从批次取出队列起计数，包含组包、连接池/SHM 等待和 RPC，
        不表示 GPU 正在执行。调用方取消后仍在执行的批次继续计入 active。
        """
        models = {
            model: {
                "queued_batches": len(self._queues.get(model, ())),
                "active_batches": self._active_by_model.get(model, 0),
            }
            for model in self._queues.keys() | self._active_by_model.keys()
        }
        return {
            "state": self._state,
            "queued_batches": self._queued_count,
            "active_batches": self._active_count,
            "queue_capacity_batches": self.queue_capacity,
            "concurrency_batches": self.concurrency,
            "models": models,
        }

    @staticmethod
    def _validate_positive_integer(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")

    async def start(self) -> None:
        async with self._condition:
            if self._state == self._OPEN:
                return
            if self._state != self._NEW:
                raise InferenceUnavailable(f"inference executor is {self._state}")

            self._state = self._OPEN
            self._lanes = [
                asyncio.create_task(
                    self._run_lane(),
                    name=f"triton-executor-{index}",
                )
                for index in range(self.concurrency)
            ]
            for lane in self._lanes:
                lane.add_done_callback(self._observe_task_completion)

    @staticmethod
    def _observe_task_completion(task: asyncio.Task[None]) -> None:
        try:
            if not task.cancelled():
                task.exception()
        except BaseException:
            pass

    async def submit(
        self,
        model: str,
        operation: Callable[[], Awaitable[T]],
        *,
        deadline: float,
    ) -> T:
        if not isinstance(model, str):
            raise TypeError("model must be a non-empty string")
        if not model:
            raise ValueError("model must be a non-empty string")
        if not callable(operation):
            raise TypeError("operation must be callable")
        if isinstance(deadline, bool):
            raise TypeError("deadline must be finite")
        try:
            deadline_value = float(deadline)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("deadline must be finite") from exc
        if not math.isfinite(deadline_value):
            raise ValueError("deadline must be finite")

        now = time.perf_counter()
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        item = _QueuedOperation(
            model=model,
            operation=cast(Callable[[], Awaitable[Any]], operation),
            deadline=deadline_value,
            enqueued_at=now,
            future=future,
        )

        admitted = False
        try:
            pre_lane_error: InferenceQueueTimeout | InferenceUnavailable | None = None
            pre_lane_error_state: tuple[int, int] | None = None
            async with self._condition:
                if self._state != self._OPEN:
                    raise InferenceUnavailable(
                        f"inference executor is {self._state}"
                    )
                if deadline_value <= time.perf_counter():
                    pre_lane_error = self._queue_timeout(model)
                elif self._queued_count >= self.queue_capacity:
                    pre_lane_error = self._queue_full(model)
                else:
                    self._enqueue_locked(item)
                    admitted = True
                    # 只有空闲 lane 等待此条件；接纳满队列时直接拒绝，不等待容量。
                    # 每个新批次唤醒一个 lane，连续入队也能覆盖多个并行执行槽。
                    self._condition.notify(1)

                if pre_lane_error is not None:
                    pre_lane_error_state = (
                        self._queued_count,
                        self._active_count,
                    )

            if pre_lane_error is not None:
                if not future.done():
                    future.cancel()
                assert pre_lane_error_state is not None
                queue_depth, active_slots = pre_lane_error_state
                self._log_before_start_completion(
                    item=item,
                    queue_depth=queue_depth,
                    active_slots=active_slots,
                    error_class=type(pre_lane_error).__name__,
                )
                raise pre_lane_error

            remaining = deadline_value - time.perf_counter()
            if remaining > 0:
                try:
                    return cast(
                        T,
                        await asyncio.wait_for(
                            asyncio.shield(future),
                            timeout=remaining,
                        ),
                    )
                except TimeoutError:
                    pass

            removed = False
            timeout_state: tuple[int, int] | None = None
            async with self._condition:
                if not item.started:
                    removed = self._remove_locked(item)
                    if not future.done():
                        future.cancel()
                    if removed:
                        timeout_state = (
                            self._queued_count,
                            self._active_count,
                        )

            if removed:
                queue_error = self._queue_timeout(model)
                assert timeout_state is not None
                queue_depth, active_slots = timeout_state
                self._log_before_start_completion(
                    item=item,
                    queue_depth=queue_depth,
                    active_slots=active_slots,
                    error_class=type(queue_error).__name__,
                )
                raise queue_error from None

            return cast(T, await asyncio.shield(future))
        except asyncio.CancelledError:
            should_log = False
            cancellation_state: tuple[int, int] | None = None
            async with self._condition:
                if not item.started:
                    removed = self._remove_locked(item) if admitted else False
                    should_log = removed or not admitted
                    if should_log:
                        cancellation_state = (
                            self._queued_count,
                            self._active_count,
                        )
                if not future.done():
                    future.cancel()

            if should_log:
                assert cancellation_state is not None
                queue_depth, active_slots = cancellation_state
                self._log_before_start_completion(
                    item=item,
                    queue_depth=queue_depth,
                    active_slots=active_slots,
                    error_class="CancelledError",
                )
            raise

    async def close(self) -> None:
        close_task: asyncio.Task[None] | None
        async with self._condition:
            if self._state == self._CLOSED:
                close_task = self._close_task
            elif self._state == self._NEW:
                self._state = self._CLOSED
                self._condition.notify_all()
                return
            else:
                if self._state in (self._OPEN, self._FAILED):
                    self._state = self._CLOSING
                    self._condition.notify_all()
                    if self._close_task is None:
                        self._close_task = asyncio.create_task(
                            self._finish_close(),
                            name="triton-executor-close",
                        )
                        self._close_task.add_done_callback(
                            self._observe_task_completion
                        )

                close_task = self._close_task

        if close_task is not None:
            await asyncio.shield(close_task)

    def _enqueue_locked(self, item: _QueuedOperation) -> None:
        queue = self._queues.get(item.model)
        if queue is None:
            queue = deque()
            self._queues[item.model] = queue
            self._ready_models.append(item.model)
        queue.append(item)
        self._queued_count += 1

    def _dequeue_locked(self) -> _QueuedOperation:
        model = self._ready_models.popleft()
        queue = self._queues[model]
        item = queue.popleft()
        if queue:
            self._ready_models.append(model)
        else:
            del self._queues[model]
        self._queued_count -= 1
        return item

    def _remove_locked(self, item: _QueuedOperation) -> bool:
        queue = self._queues.get(item.model)
        if queue is None:
            return False
        try:
            queue.remove(item)
        except ValueError:
            return False

        self._queued_count -= 1
        if not queue:
            del self._queues[item.model]
            try:
                self._ready_models.remove(item.model)
            except ValueError:
                pass
        return True

    async def _take_next(self) -> _QueuedOperation | None:
        while True:
            expired_item: _QueuedOperation | None = None
            expired_state: tuple[int, int] | None = None
            async with self._condition:
                try:
                    await self._condition.wait_for(
                        lambda: bool(self._ready_models)
                        or self._state != self._OPEN
                    )
                except asyncio.CancelledError:
                    # Condition.wait 取消后会重新持锁；将已选中的唤醒交给另一lane，
                    # 避免该lane退出时留下有工作却全部休眠的队列。
                    if self._ready_models:
                        self._condition.notify(1)
                    raise
                if not self._ready_models:
                    return None

                item = self._dequeue_locked()
                if item.deadline <= time.perf_counter():
                    expired_item = item
                    expired_state = (
                        self._queued_count,
                        self._active_count,
                    )
                else:
                    item.started = True
                    self._active_count += 1
                    self._active_by_model[item.model] = (
                        self._active_by_model.get(item.model, 0) + 1
                    )
                    return item

            assert expired_item is not None
            assert expired_state is not None
            queue_depth, active_slots = expired_state
            queue_error = self._queue_timeout(expired_item.model)
            try:
                self._log_before_start_completion(
                    item=expired_item,
                    queue_depth=queue_depth,
                    active_slots=active_slots,
                    error_class=type(queue_error).__name__,
                )
                if not expired_item.future.done():
                    expired_item.future.set_exception(queue_error)
            except (KeyboardInterrupt, SystemExit) as fatal:
                await self._fail_after_fatal(fatal)
                if not expired_item.future.done():
                    expired_item.future.set_exception(fatal)
                raise _LaneFatalError(
                    f"inference executor lane stopped after "
                    f"{type(fatal).__name__}"
                ) from fatal
            finally:
                # The next condition wait must not retain an expired request or
                # its exception traceback after the caller has handled it.
                del item, expired_item, queue_error

    async def _run_lane(self) -> None:
        while True:
            item = await self._take_next()
            if item is None:
                return
            try:
                await self._execute(item)
            finally:
                # Release the operation closure and future before waiting idle.
                del item

    async def _fail_after_fatal(self, fatal: BaseException) -> None:
        async with self._condition:
            queued = [
                item
                for queue in self._queues.values()
                for item in queue
            ]
            self._queues.clear()
            self._ready_models.clear()
            self._queued_count = 0
            self._state = self._FAILED
            active_slots = self._active_count
            self._condition.notify_all()

        for index, queued_item in enumerate(queued):
            error = InferenceUnavailable(
                f"{queued_item.model}: inference executor stopped after "
                f"{type(fatal).__name__}"
            )
            if not queued_item.future.done():
                queued_item.future.set_exception(error)
            try:
                self._log_before_start_completion(
                    item=queued_item,
                    queue_depth=len(queued) - index - 1,
                    active_slots=active_slots,
                    error_class=type(error).__name__,
                )
            except BaseException:
                pass

    async def _execute(self, item: _QueuedOperation) -> None:
        started_at = time.perf_counter()
        queue_wait_ms = (started_at - item.enqueued_at) * 1000
        deadline_remaining_ms = max(0.0, (item.deadline - started_at) * 1000)
        async with self._condition:
            queue_depth = self._queued_count
            active_slots = self._active_count

        status = "success"
        error_class: str | None = None
        result: Any = None
        failure: BaseException | None = None
        cancelled = False
        fatal: KeyboardInterrupt | SystemExit | None = None
        try:
            result = await item.operation()
        except asyncio.CancelledError:
            status = "cancelled"
            error_class = "CancelledError"
            cancelled = True
        except (KeyboardInterrupt, SystemExit) as exc:
            status = "error"
            error_class = type(exc).__name__
            fatal = exc
        except BaseException as exc:
            status = "error"
            error_class = type(exc).__name__
            failure = exc

        async with self._condition:
            self._active_count -= 1
            remaining_active = self._active_by_model[item.model] - 1
            if remaining_active:
                self._active_by_model[item.model] = remaining_active
            else:
                del self._active_by_model[item.model]

        try:
            self._log_completion(
                item=item,
                queue_wait_ms=queue_wait_ms,
                queue_depth=queue_depth,
                active_slots=active_slots,
                deadline_remaining_ms=deadline_remaining_ms,
                status=status,
                error_class=error_class,
            )
        except (KeyboardInterrupt, SystemExit) as exc:
            if fatal is None:
                fatal = exc
                status = "error"
                error_class = type(exc).__name__

        if fatal is not None:
            await self._fail_after_fatal(fatal)

        if not item.future.done():
            if fatal is not None:
                item.future.set_exception(fatal)
            elif cancelled:
                item.future.cancel()
            elif failure is not None:
                item.future.set_exception(failure)
            else:
                item.future.set_result(result)

        if fatal is not None:
            raise _LaneFatalError(
                f"inference executor lane stopped after "
                f"{type(fatal).__name__}"
            ) from fatal

    async def _finish_close(self) -> None:
        results = await asyncio.gather(*self._lanes, return_exceptions=True)
        errors = [
            result
            for result in results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ]
        async with self._condition:
            self._state = self._CLOSED
            self._condition.notify_all()

        if errors:
            error = errors[0]
            if isinstance(error, _LaneFatalError) and error.__cause__ is not None:
                error = error.__cause__
            raise RuntimeError("inference executor lane failed") from error

    @staticmethod
    def _queue_timeout(model: str) -> InferenceQueueTimeout:
        return InferenceQueueTimeout(
            f"{model}: request expired in global inference queue"
        )

    @staticmethod
    def _queue_full(model: str) -> InferenceUnavailable:
        return InferenceUnavailable(
            f"{model}: global inference queue is full"
        )

    def _log_completion(
        self,
        *,
        item: _QueuedOperation,
        queue_wait_ms: float,
        queue_depth: int,
        active_slots: int,
        deadline_remaining_ms: float,
        status: str,
        error_class: str | None,
    ) -> None:
        try:
            completion_logger = self._logger.bind(
                model=item.model,
                executor_queue_wait_ms=queue_wait_ms,
                executor_queue_depth=queue_depth,
                executor_active_slots=active_slots,
                executor_concurrency=self.concurrency,
                deadline_remaining_ms=deadline_remaining_ms,
                status=status,
                error_class=error_class,
            )
            level = "debug" if error_class is None else "warning"
            getattr(completion_logger, level, completion_logger.info)(
                "Triton executor completed: status={}, error_class={}",
                status,
                error_class or "",
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            pass

    def _log_before_start_completion(
        self,
        *,
        item: _QueuedOperation,
        queue_depth: int,
        active_slots: int,
        error_class: str,
    ) -> None:
        completed_at = time.perf_counter()
        self._log_completion(
            item=item,
            queue_wait_ms=(completed_at - item.enqueued_at) * 1000,
            queue_depth=queue_depth,
            active_slots=active_slots,
            deadline_remaining_ms=max(
                0.0,
                (item.deadline - completed_at) * 1000,
            ),
            status="error",
            error_class=error_class,
        )
