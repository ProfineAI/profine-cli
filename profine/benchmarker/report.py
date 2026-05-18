"""Side-by-side benchmark report generation."""

from __future__ import annotations

from profine.benchmarker.comparator import BenchmarkComparison, MetricDelta


def generate_report(
    comparison: BenchmarkComparison,
    optimization_name: str = "",
    *,
    hardware: str | None = None,
    cost_per_hour: float | None = None,
) -> str:
    """Generate a human-readable side-by-side benchmark report.

    Args:
        comparison: Computed comparison between baseline and optimized runs.
        optimization_name: Human label for the optimization(s) applied.
        hardware: Hardware preset name (used in the cost-savings section).
        cost_per_hour: USD/hour for the hardware (used to project savings).
    """
    lines: list[str] = ["# Benchmark Report", ""]

    # TL;DR headline — what the user wants to know in 1 line.
    lines.append(_tldr(comparison, optimization_name))
    lines.append("")

    if optimization_name:
        lines.append(f"**Optimization applied:** {optimization_name}")
    if hardware:
        lines.append(f"**Hardware:** {hardware}" + (f" (${cost_per_hour:.2f}/hr)" if cost_per_hour else ""))
    lines.append(f"**Verdict:** {comparison.verdict}")
    if comparison.summary and comparison.summary != comparison.verdict:
        lines.append(f"**Notes:** {comparison.summary}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Metrics table
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Optimized | Δ | |")
    lines.append("|---|---|---|---|---|")
    for m in comparison.metrics:
        lines.append(
            f"| {_friendly_name(m.name)} "
            f"| {_fmt(m.baseline, m.name)} "
            f"| {_fmt(m.candidate, m.name)} "
            f"| {m.delta_pct:+.1f}% "
            f"| {_arrow(m)} |"
        )
    lines.append("")

    b = comparison.baseline_step_stats
    c_stats = comparison.candidate_step_stats
    if b.n_samples and c_stats.n_samples:
        lines.append("### Measurement stability")
        lines.append("")
        lines.append("| Run | p25 (ms) | p50 (ms) | p75 (ms) | CV | N |")
        lines.append("|---|---|---|---|---|---|")
        lines.append(
            f"| Baseline  | {b.p25:.2f} | {b.p50:.2f} | {b.p75:.2f} "
            f"| {b.cv*100:.1f}% | {b.n_samples} |"
        )
        lines.append(
            f"| Optimized | {c_stats.p25:.2f} | {c_stats.p50:.2f} | {c_stats.p75:.2f} "
            f"| {c_stats.cv*100:.1f}% | {c_stats.n_samples} |"
        )
        lines.append("")
        if comparison.noisy:
            lines.append(
                "> ⚠️  High step-time variance — consider bumping `--steps` for a tighter estimate."
            )
            lines.append("")


    # Projected savings — translate step-time improvement into hours and dollars.
    savings = _projected_savings(comparison, cost_per_hour)
    if savings:
        lines.append("## Projected Savings")
        lines.append("")
        lines.extend(savings)
        lines.append("")

    # Correctness
    lines.append("## Correctness")
    lines.append("")
    c = comparison.correctness
    lines.append(f"- **Loss curves match:** {'Yes ✓' if c.loss_match else 'No ✗'}")
    lines.append(f"- **Max loss diff:** {c.max_loss_diff:.6f}")
    lines.append(f"- **Tolerance:** rtol={c.rtol}, atol={c.atol}")
    if c.notes:
        lines.append(f"- **Notes:** {c.notes}")
    lines.append("")

    # Ship-it guidance — actionable line at the end.
    lines.append("## Recommendation")
    lines.append("")
    lines.append(_recommendation(comparison))
    lines.append("")

    return "\n".join(lines)


def _tldr(comparison: BenchmarkComparison, optimization_name: str) -> str:
    """One-line headline: did it work, by how much."""
    speedup = comparison.speedup_pct
    c = comparison.correctness
    if c.tolerance_widened:
        cat = c.tolerance_widened_for or "this optimization"
        correctness_note = (
            f"correctness preserved (at relaxed tolerance for {cat}: "
            f"rtol={c.rtol:g}, atol={c.atol:g})"
        )
    else:
        correctness_note = "correctness preserved"
    band = _speedup_band_suffix(comparison)
    if speedup >= 3.0 and c.passed:
        mult = 100.0 / (100.0 - speedup) if speedup < 100 else float("inf")
        return f"## ✅ {speedup:.1f}% faster ({mult:.2f}× speedup{band}), {correctness_note}."
    if speedup >= 3.0 and not c.passed:
        return f"## ⚠️ {speedup:.1f}% faster, but loss curves diverge — review before shipping."
    if speedup <= -2.0:
        return f"## ❌ {abs(speedup):.1f}% regression — do not ship."
    return f"## ➖ No meaningful change ({speedup:+.1f}%)."


def _speedup_band_suffix(c: BenchmarkComparison) -> str:
    """Inline confidence band on the headline, hidden when the run is tight."""
    headline = c.speedup_pct
    lo = c.speedup_pct_p25
    hi = c.speedup_pct_p75
    if not c.noisy and (hi - lo) <= 4.0:
        return ""
    if lo <= 0 and headline > 0:
        lo_str = "no change"
    else:
        lo_mult = 100.0 / (100.0 - lo) if lo < 100 else float("inf")
        lo_str = f"{lo_mult:.2f}×"
    hi_mult = 100.0 / (100.0 - hi) if hi < 100 else float("inf")
    return f"; {lo_str}–{hi_mult:.2f}× across step-time p25–p75"


def _projected_savings(
    comparison: BenchmarkComparison, cost_per_hour: float | None
) -> list[str]:
    """Translate step-time improvement into time/$ saved at scale."""
    speedup = comparison.speedup_pct
    if speedup <= 0:
        return []
    fraction_saved = min(speedup / 100.0, 0.99)
    out: list[str] = [
        f"For every **100 hours** of training time saved at the optimized step time, "
        f"you'd have spent **{100.0 / (1.0 - fraction_saved):.0f} hours** on the baseline.",
        "",
        "| Baseline run length | Time saved | Cost saved |",
        "|---|---|---|",
    ]
    for baseline_hours in (1, 10, 100, 1000):
        time_saved_hours = baseline_hours * fraction_saved
        cost_saved = (
            f"${time_saved_hours * cost_per_hour:.2f}"
            if cost_per_hour
            else "—"
        )
        out.append(
            f"| {baseline_hours} hr | {time_saved_hours:.2f} hr "
            f"({time_saved_hours * 60:.0f} min) | {cost_saved} |"
        )
    return out


def _recommendation(comparison: BenchmarkComparison) -> str:
    if not comparison.correctness.passed:
        if comparison.speedup_pct >= 3.0:
            return (
                "**Hold.** Speedup is real, but loss curves diverged beyond tolerance. "
                "Either widen `--rtol`/`--atol` if your model is known-perturbative (e.g. quantization), "
                "or investigate the divergence before merging."
            )
        return "**Reject.** Correctness check failed and there's no compensating speedup."
    if comparison.speedup_pct >= 3.0:
        return "**Ship it.** Speedup exceeds the 3% threshold and correctness passed."
    if comparison.speedup_pct <= -2.0:
        return "**Revert.** Step time regressed beyond noise."
    return (
        "**No-op.** Change isn't worth merging on its own — but it may compose with "
        "future optimizations (run `profine suggest` again with the latest profile)."
    )


def _arrow(m: MetricDelta) -> str:
    if m.improved:
        return "↑ improved"
    if abs(m.delta_pct) < 1:
        return "~ same"
    return "↓ regressed"


def _friendly_name(name: str) -> str:
    names = {
        "step_time_median_ms": "Step time (ms)",
        "throughput_steps_per_sec": "Throughput (steps/s)",
        "memory_peak_gb": "Peak memory (GB)",
        "gpu_util_mean_pct": "GPU utilization (%)",
    }
    return names.get(name, name)


def _fmt(val: float, name: str) -> str:
    if "pct" in name or "util" in name:
        return f"{val:.1f}"
    if "gb" in name.lower() or "memory" in name.lower():
        return f"{val:.2f}"
    return f"{val:.2f}"
