"""Bounded frame inference and ordered video output, independent of worker services."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
import time
from typing import Any

import numpy as np

from core.ordered_pipeline import run_ordered_pipeline
from core.queue_metrics import PipelineMetrics


async def process_video_frames(
    frames: Iterable[tuple[int, np.ndarray]],
    infer: Callable[[np.ndarray], Awaitable[dict[str, Any]]],
    commit: Callable[[int, np.ndarray, dict[str, Any]], Awaitable[None]],
    *,
    concurrency: int = 2,
    lookahead: int = 2,
    metrics: PipelineMetrics | None = None,
) -> dict[str, Any]:
    """Infer concurrently and commit frames in their original iterable order.

    At most ``concurrency`` inference callbacks run at once; at most
    ``concurrency + lookahead`` frames remain uncommitted. The caller owns
    decoding, color format, the real GPU client, encoding and their resources. Exceptions from
    inference, output or input iteration propagate after owned tasks are drained.

    Latencies measure the successful ``await infer(frame)`` wall time, including
    client/transport waits, rather than device kernel time. Exact percentiles
    retain one float per successful frame; image payloads use the bounded window.
    Supplied metrics remain reusable and their counters are cumulative.
    """
    run_metrics = metrics if metrics is not None else PipelineMetrics()
    started = time.perf_counter()
    latencies: list[float] = []
    committed = 0

    async def prepare(item: tuple[int, np.ndarray]):
        index, frame = item
        with run_metrics.activity("video_inference"):
            infer_started = time.perf_counter()
            result = await infer(frame)
            latencies.append((time.perf_counter() - infer_started) * 1000)
        return index, frame, result

    async def write(prepared):
        nonlocal committed
        with run_metrics.activity("video_commit"):
            await commit(*prepared)
            committed += 1

    await run_ordered_pipeline(
        frames,
        prepare,
        write,
        concurrency=concurrency,
        lookahead=lookahead,
        stage_name="video_ordered_commit",
        metrics=run_metrics,
    )
    return {
        "frames": committed,
        "elapsed_seconds": time.perf_counter() - started,
        "infer_latency_ms": {
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
        },
        "pipeline_metrics": run_metrics.snapshot(final=True),
    }
