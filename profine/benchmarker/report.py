"""Side-by-side benchmark report generation."""

from __future__ import annotations

from profine.benchmarker.comparator import BenchmarkComparison, MetricDelta


def generate_report(comparison: BenchmarkComparison, optimization_name: str = "") -> str:
    """Generate a human-readable side-by-side benchmark report."""
    lines: list[str] = ["# Benchmark Report", ""]

    if optimization_name:
        lines.append(f"**Optimization:** {optimization_name}")
        lines.append("")

    lines.append(f"**Verdict:** {comparison.summary}")
    lines.append("")

    # Headline
    lines.append("---")
    lines.append("")

    if comparison.speedup_pct > 0:
        lines.append(f"**Speedup: {comparison.speedup_pct:.1f}%**")
    elif comparison.speedup_pct < 0:
        lines.append(f"**Regression: {abs(comparison.speedup_pct):.1f}% slower**")
    lines.append("")

    # Metrics table
    lines.append("## Metrics Comparison")
    lines.append("")
    lines.append("| Metric | Baseline | Optimized | Delta | Change |")
    lines.append("|--------|----------|-----------|-------|--------|")

    for m in comparison.metrics:
        arrow = _arrow(m)
        lines.append(
            f"| {_friendly_name(m.name)} "
            f"| {_fmt(m.baseline, m.name)} "
            f"| {_fmt(m.candidate, m.name)} "
            f"| {m.delta_pct:+.1f}% "
            f"| {arrow} |"
        )

    lines.append("")

    # Correctness
    lines.append("## Correctness Check")
    lines.append("")
    c = comparison.correctness
    lines.append(f"- **Loss match:** {'Yes' if c.loss_match else 'No'}")
    lines.append(f"- **Max loss diff:** {c.max_loss_diff:.6f}")
    lines.append(f"- **Tolerance:** rtol={c.rtol}, atol={c.atol}")
    if c.notes:
        lines.append(f"- **Notes:** {c.notes}")
    lines.append("")

    return "\n".join(lines)


def _arrow(m: MetricDelta) -> str:
    if m.improved:
        return "improved"
    if abs(m.delta_pct) < 1:
        return "~same"
    return "regressed"


def _friendly_name(name: str) -> str:
    names = {
        "step_time_median_ms": "Step Time (ms)",
        "throughput_steps_per_sec": "Throughput (steps/s)",
        "memory_peak_gb": "Peak Memory (GB)",
        "gpu_util_mean_pct": "GPU Utilization (%)",
    }
    return names.get(name, name)


def _fmt(val: float, name: str) -> str:
    if "pct" in name or "util" in name:
        return f"{val:.1f}"
    if "gb" in name.lower() or "memory" in name.lower():
        return f"{val:.2f}"
    return f"{val:.2f}"
