"""Benchmarker tool (plan 4.6).

Public API:
    from profine.benchmarker import benchmark

    result = benchmark("train.py", optimized_source, hardware="1x_a100")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from profine.benchmarker.benchmarker import Benchmarker, BenchmarkResult
from profine.schema.hardware import HardwareConfig


def benchmark(
    baseline_path: str | Path,
    optimized_source: str,
    *,
    hardware: str | HardwareConfig = "1x_a100",
    steps: int = 60,
    warmup_steps: int = 30,
    rtol: float = 1e-2,
    atol: float = 1e-4,
    optimization_name: str = "",
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Convenience entry point for the Benchmarker tool."""
    b = Benchmarker(provider=provider, api_key=api_key, model=model, **kwargs)
    return b.benchmark(
        baseline_path,
        optimized_source,
        hardware=hardware,
        steps=steps,
        warmup_steps=warmup_steps,
        rtol=rtol,
        atol=atol,
        optimization_name=optimization_name,
    )
