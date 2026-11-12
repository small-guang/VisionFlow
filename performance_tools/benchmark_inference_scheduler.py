"""无外部服务的推理执行器微基准；原生计时与Condition唤醒计数分开运行。"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class QuietLogger:
    def bind(self, **kwargs): return self
    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass


class CountingCondition(asyncio.Condition):
    def __init__(self, executor):
        super().__init__()
        self.executor = executor
        self.counts = Counter()

    async def wait(self):
        self.counts["wait_calls"] += 1
        result = await super().wait()
        self.counts["wait_resumed"] += 1
        if not self.executor._ready_models and self.executor._state == self.executor._OPEN:
            self.counts["resumed_without_work_while_open"] += 1
        return result

    async def wait_for(self, predicate):
        self.counts["wait_for_calls"] += 1
        return await super().wait_for(predicate)

    def notify(self, n=1):
        self.counts["notify_calls"] += 1
        return super().notify(n)

    def notify_all(self):
        self.counts["notify_all_calls"] += 1
        return super().notify_all()


def load_executor(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.FairInferenceExecutor


async def sample(executor_type, *, operations, concurrency, producers, models, yields, count=False):
    executor = executor_type(concurrency=concurrency, queue_capacity=max(64, producers), logger=QuietLogger())
    if count:
        executor._condition = CountingCondition(executor)
    await executor.start()
    # 先让所有lane进入空闲等待；启动/关闭广播不混入业务窗口。
    await asyncio.sleep(0)
    if count:
        executor._condition.counts.clear()
    next_index = completed = active = peak = 0

    async def operation(index):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            for _ in range(yields):
                await asyncio.sleep(0)
            return index
        finally:
            active -= 1

    async def producer():
        nonlocal next_index, completed
        while next_index < operations:
            index = next_index
            next_index += 1
            result = await executor.submit(f"model-{index % models}",
                                           lambda index=index: operation(index),
                                           deadline=time.perf_counter() + 60)
            if result != index:
                raise AssertionError("执行器结果错配")
            completed += 1

    try:
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        await asyncio.gather(*(producer() for _ in range(producers)))
        # 等待本轮已排队的空唤醒重新休眠，使旧版的收尾通知计入窗口。
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        wall_seconds = time.perf_counter() - wall_started
        cpu_seconds = time.process_time() - cpu_started
        if completed != operations or executor._queued_count or executor._active_count:
            raise AssertionError("采样结束时仍有未完成操作")
        return {"wall_seconds": wall_seconds, "process_cpu_seconds": cpu_seconds,
                "operations_completed": completed, "peak_active": peak,
                "condition": dict(executor._condition.counts) if count else None}
    finally:
        await executor.close()


async def benchmark(args):
    sources = {"baseline": args.baseline_executor, "candidate": args.candidate_executor}
    classes = {name: load_executor(path, "_scheduler_benchmark_" + name) for name, path in sources.items()}
    cases = []
    for producers in args.producers:
        parameters = dict(operations=args.operations, concurrency=args.concurrency, producers=producers,
                          models=args.models, yields=args.operation_yields)
        # 相同输入预热，随后交替测量顺序；不用cProfile，不把计数器开销算作原生时间。
        for cls in classes.values():
            await sample(cls, **{**parameters, "operations": min(100, args.operations)})
        samples = {name: [] for name in classes}
        for iteration in range(args.repeats):
            for name in (tuple(classes) if iteration % 2 == 0 else tuple(reversed(classes))):
                samples[name].append(await sample(classes[name], **parameters))
        measured = {}
        for name, cls in classes.items():
            counted = await sample(cls, **parameters, count=True)
            measured[name] = {
                "wall_seconds_median": statistics.median(row["wall_seconds"] for row in samples[name]),
                "process_cpu_seconds_median": statistics.median(row["process_cpu_seconds"] for row in samples[name]),
                "peak_active": max(row["peak_active"] for row in samples[name]),
                "unprofiled_samples": samples[name], "separate_counted_run": counted["condition"],
            }
        cases.append({"producers": producers, **measured})
    return {"schema": 1, "loop": args.loop, "operations_per_sample": args.operations,
            "concurrency": args.concurrency, "models": args.models, "operation_yields": args.operation_yields,
            "repeats": args.repeats,
            "executor_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
            "cases": cases,
            "notes": ["仅比较执行器调度；无图片、GPU、网络、数据库和真实RPC，不代表端到端吞吐。",
                      "计时使用独立未插桩运行的perf_counter/process_time；Condition计数另跑一次。",
                      "每次操作通过sleep(0)模拟让出事件循环；所有计数排除启动和关闭广播。",
                      "不同生产者数用于比较低占用与并发场景，不能用profile累计时间推断硬件CPU归因。"]}


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-executor", type=Path, required=True)
    parser.add_argument("--candidate-executor", type=Path, default=PROJECT_ROOT / "core/infer/executor.py")
    parser.add_argument("--concurrency", type=positive_int, default=80)
    parser.add_argument("--operations", type=positive_int, default=2305)
    parser.add_argument("--producers", type=lambda text: [positive_int(value) for value in text.split(",")], default=[1, 8, 80])
    parser.add_argument("--models", type=positive_int, default=8)
    parser.add_argument("--operation-yields", type=positive_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--loop", choices=("asyncio", "uvloop"), default="asyncio")
    args = parser.parse_args()
    if args.loop == "uvloop":
        import uvloop
        result = uvloop.run(benchmark(args))
    else:
        result = asyncio.run(benchmark(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
