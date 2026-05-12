"""Metric comparison and correctness checking for benchmarks.

Compares two profiler payloads (baseline vs. optimized) and produces
structured deltas + a pass/fail correctness verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MetricDelta:
    """Comparison of a single metric between baseline and candidate."""
    name: str
    baseline: float = 0.0
    candidate: float = 0.0
    delta: float = 0.0          # candidate - baseline
    delta_pct: float = 0.0      # (candidate - baseline) / baseline * 100
    improved: bool = False       # True if delta is in the "good" direction


@dataclass(slots=True)
class CorrectnessVerdict:
    """Pass/fail correctness check based on loss curve matching."""
    passed: bool = True
    loss_match: bool = True
    max_loss_diff: float = 0.0
    rtol: float = 1e-2
    atol: float = 1e-4
    notes: str = ""


@dataclass(slots=True)
class BenchmarkComparison:
    """Full comparison between baseline and optimized runs."""
    metrics: list[MetricDelta] = field(default_factory=list)
    correctness: CorrectnessVerdict = field(default_factory=CorrectnessVerdict)
    speedup_pct: float = 0.0         # end-to-end step time improvement
    memory_delta_pct: float = 0.0    # memory change (negative = saved)
    util_delta_pct: float = 0.0      # GPU util change (negative = regressed)
    verdict: str = "NO-OP"           # "PASS" | "NO-OP" | "REGRESSION"
    summary: str = ""


# Asymmetric thresholds: telling a customer "PASS" on a no-op is worse
# than telling them "REGRESSION" on a tied run. Util-drop alone counts
# as regression because same throughput at lower util scales worse.
SPEEDUP_PASS_PCT = 3.0
SPEEDUP_REG_PCT = -2.0
UTIL_DROP_REG_PCT = -15.0


def _decide_verdict(speedup_pct: float, util_delta_pct: float, correctness_passed: bool) -> str:
    # Determine speed verdict independently of correctness.
    #
    # Util-drop only counts as a regression when throughput is NOT improving.
    # The original rationale: "same step-time at lower util" suggests the GPU
    # is sitting idle on something (e.g. CPU-bound dataloader regression). But
    # a util drop alongside a clear speedup just means the optimized code does
    # each step faster, so the GPU sits idle more between steps. That's a
    # feature — flagging it REGRESSION misleads the user.
    speed_regression = speedup_pct <= SPEEDUP_REG_PCT
    util_regression = (
        util_delta_pct <= UTIL_DROP_REG_PCT
        and speedup_pct < SPEEDUP_PASS_PCT
    )

    if speed_regression or util_regression:
        speed = "REGRESSION"
    elif speedup_pct >= SPEEDUP_PASS_PCT:
        speed = "PASS"
    else:
        speed = "NO-OP"

    if not correctness_passed:
        # Report the speed result but flag correctness failure
        if speed == "PASS":
            return "PASS (correctness: FAIL)"
        return "REGRESSION"
    return speed


def compare_payloads(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rtol: float = 1e-2,
    atol: float = 1e-4,
) -> BenchmarkComparison:
    """Compare two profiler payloads and produce a structured comparison.

    Args:
        baseline: Raw payload from baseline profiler run.
        candidate: Raw payload from optimized profiler run.
        rtol: Relative tolerance for loss curve matching.
        atol: Absolute tolerance for loss curve matching.

    Returns:
        BenchmarkComparison with metrics, correctness, and summary.
    """
    metrics: list[MetricDelta] = []

    # Step time (lower is better)
    b_step = _median(baseline.get("step_times_ms", []))
    c_step = _median(candidate.get("step_times_ms", []))
    metrics.append(_make_delta("step_time_median_ms", b_step, c_step, lower_is_better=True))

    # Throughput (higher is better) — inverse of step time
    if b_step > 0 and c_step > 0:
        b_tps = 1000.0 / b_step
        c_tps = 1000.0 / c_step
        metrics.append(_make_delta("throughput_steps_per_sec", b_tps, c_tps, lower_is_better=False))

    # Peak memory (lower is better)
    b_mem = baseline.get("memory_peak_bytes", 0) / (1024**3)
    c_mem = candidate.get("memory_peak_bytes", 0) / (1024**3)
    metrics.append(_make_delta("memory_peak_gb", b_mem, c_mem, lower_is_better=True))

    # GPU utilization (higher is better)
    b_gpu = _mean(baseline.get("gpu_utilization_samples", []))
    c_gpu = _mean(candidate.get("gpu_utilization_samples", []))
    metrics.append(_make_delta("gpu_util_mean_pct", b_gpu, c_gpu, lower_is_better=False))

    # Correctness check
    correctness = _check_correctness(
        baseline.get("loss_values", []),
        candidate.get("loss_values", []),
        rtol=rtol,
        atol=atol,
    )

    speedup_pct = metrics[0].delta_pct * -1 if metrics else 0.0
    memory_delta_pct = metrics[2].delta_pct if len(metrics) > 2 else 0.0
    util_delta_pct = (c_gpu - b_gpu) if (b_gpu or c_gpu) else 0.0  # absolute pp, not %

    verdict = _decide_verdict(speedup_pct, util_delta_pct, correctness.passed)
    summary = _build_summary(verdict, speedup_pct, memory_delta_pct,
                              util_delta_pct, correctness)

    return BenchmarkComparison(
        metrics=metrics,
        correctness=correctness,
        speedup_pct=speedup_pct,
        memory_delta_pct=memory_delta_pct,
        util_delta_pct=util_delta_pct,
        verdict=verdict,
        summary=summary,
    )


def _make_delta(name: str, baseline: float, candidate: float, *, lower_is_better: bool) -> MetricDelta:
    delta = candidate - baseline
    delta_pct = (delta / baseline * 100) if baseline != 0 else 0.0
    improved = delta < 0 if lower_is_better else delta > 0
    return MetricDelta(
        name=name,
        baseline=round(baseline, 4),
        candidate=round(candidate, 4),
        delta=round(delta, 4),
        delta_pct=round(delta_pct, 2),
        improved=improved,
    )


def _check_correctness(
    baseline_losses: list[float],
    candidate_losses: list[float],
    rtol: float,
    atol: float,
) -> CorrectnessVerdict:
    """Check if loss curves match within tolerance."""
    if not baseline_losses or not candidate_losses:
        return CorrectnessVerdict(
            passed=True,
            loss_match=True,
            notes="No loss values captured — correctness not verified",
        )

    # Compare overlapping steps
    n = min(len(baseline_losses), len(candidate_losses))
    max_diff = 0.0

    for i in range(n):
        b, c = baseline_losses[i], candidate_losses[i]
        diff = abs(b - c)
        max_diff = max(max_diff, diff)

        # numpy-style allclose: |b - c| <= atol + rtol * |b|
        threshold = atol + rtol * abs(b)
        if diff > threshold:
            return CorrectnessVerdict(
                passed=False,
                loss_match=False,
                max_loss_diff=max_diff,
                rtol=rtol,
                atol=atol,
                notes=f"Loss diverged at step {i}: baseline={b:.6f}, candidate={c:.6f}, diff={diff:.6f}",
            )

    return CorrectnessVerdict(
        passed=True,
        loss_match=True,
        max_loss_diff=max_diff,
        rtol=rtol,
        atol=atol,
        notes=f"Loss curves match across {n} steps (max diff: {max_diff:.6f})",
    )


def _build_summary(
    verdict: str,
    speedup_pct: float,
    memory_delta_pct: float,
    util_delta_pct: float,
    correctness: CorrectnessVerdict,
) -> str:
    parts: list[str] = [verdict]

    if speedup_pct > 0:
        parts.append(f"{speedup_pct:.1f}% faster")
    elif speedup_pct < 0:
        parts.append(f"{abs(speedup_pct):.1f}% slower")
    else:
        parts.append("no step-time change")

    if memory_delta_pct < -1:
        parts.append(f"{abs(memory_delta_pct):.1f}% less memory")
    elif memory_delta_pct > 1:
        parts.append(f"{memory_delta_pct:.1f}% more memory")

    if util_delta_pct <= -5:
        parts.append(f"GPU util -{abs(util_delta_pct):.0f}pp")
    elif util_delta_pct >= 5:
        parts.append(f"GPU util +{util_delta_pct:.0f}pp")

    parts.append(f"correctness: {'PASS' if correctness.passed else 'FAIL'}")
    return " | ".join(parts)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
