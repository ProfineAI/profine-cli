"""Generate human-readable markdown profile report."""

from __future__ import annotations

from profine.schema.profile_record import ProfileRecord


def generate_report(record: ProfileRecord) -> str:
    """Build a markdown profile report from a ProfileRecord."""
    sections = [
        _header(record),
        _step_time_summary(record),
        _gpu_utilization(record),
        _memory_summary(record),
        _top_kernels(record),
        _kernel_breakdown(record),
        _phase_breakdown(record),
        _communication(record),
        _data_loading(record),
        _warnings(record),
    ]
    return "\n\n".join(s for s in sections if s)


def _header(r: ProfileRecord) -> str:
    lines = [
        f"# Profile Report: {r.script_path}",
        "",
        f"- **Hardware**: {r.hardware_name}",
        f"- **Status**: {r.status}",
        f"- **Steps**: {r.steps_completed}/{r.steps_requested} "
        f"(warmup: {r.warmup_steps_effective})",
        f"- **Runtime**: {r.runtime_seconds:.1f}s",
    ]
    if r.error:
        lines.append(f"- **Error**: {r.error}")
    return "\n".join(lines)


def _step_time_summary(r: ProfileRecord) -> str:
    if not r.step_times_ms:
        return ""
    median = r.step_time_median_ms
    if median is None:
        return ""
    lines = [
        "## Step Time",
        "",
        f"- **Median (steady-state)**: {median:.2f} ms",
        f"- **Steps measured**: {len(r.step_times_ms)}",
    ]
    if r.warmup_step_times_ms:
        warmup_median = sorted(r.warmup_step_times_ms)[len(r.warmup_step_times_ms) // 2]
        overhead = warmup_median / median if median > 0 else 0
        lines.append(f"- **Warmup median**: {warmup_median:.2f} ms ({overhead:.1f}x steady-state)")
    if r.loss_values:
        lines.append(f"- **Final loss**: {r.loss_values[-1]:.4f}")
    return "\n".join(lines)


def _gpu_utilization(r: ProfileRecord) -> str:
    if not r.gpu_util_samples:
        return ""
    return "\n".join([
        "## GPU Utilization",
        "",
        f"- **Mean**: {r.gpu_util_mean:.1f}%",
        f"- **Pattern**: {r.gpu_util_pattern}",
        f"- **Samples**: {len(r.gpu_util_samples)}",
    ])


def _memory_summary(r: ProfileRecord) -> str:
    if r.memory_peak_bytes <= 0:
        return ""
    return "\n".join([
        "## Memory",
        "",
        f"- **Peak**: {r.memory_peak_gb:.2f} GB",
        f"- **Headroom**: {r.memory_headroom_pct:.1f}%",
    ])


def _top_kernels(r: ProfileRecord) -> str:
    if not r.top_kernels:
        return ""
    lines = [
        "## Top Kernels by CUDA Time",
        "",
        "| Kernel | Category | Time (%) | Count |",
        "|--------|----------|----------|-------|",
    ]
    for k in r.top_kernels[:10]:
        name = k.name[:50] + "..." if len(k.name) > 50 else k.name
        lines.append(f"| {name} | {k.category} | {k.pct_of_total:.1f}% | {k.count} |")
    return "\n".join(lines)


def _kernel_breakdown(r: ProfileRecord) -> str:
    bd = r.kernel_breakdown
    if not bd:
        return ""
    categories = [
        ("Matmul", bd.matmul_pct),
        ("Attention", bd.attention_pct),
        ("Elementwise", bd.elementwise_pct),
        ("Normalization", bd.normalization_pct),
        ("Optimizer", bd.optimizer_pct),
        ("Communication", bd.communication_pct),
        ("Memory", bd.memory_pct),
        ("DataLoader", bd.dataloader_pct),
        ("Other", bd.other_pct),
    ]
    lines = ["## Kernel Category Breakdown", ""]
    max_bar = 40
    for name, pct in sorted(categories, key=lambda x: x[1], reverse=True):
        if pct < 0.5:
            continue
        bar_len = int(pct / 100 * max_bar)
        bar = "█" * bar_len
        lines.append(f"  {name:<15} {bar} {pct:.1f}%")
    return "\n".join(lines)


def _phase_breakdown(r: ProfileRecord) -> str:
    pb = r.phase_breakdown
    if not pb:
        return ""
    return "\n".join([
        "## Phase Breakdown",
        "",
        f"- **Forward**: {pb.forward_pct:.1f}%",
        f"- **Backward**: {pb.backward_pct:.1f}%",
        f"- **Optimizer**: {pb.optimizer_pct:.1f}%",
        f"- **DataLoader**: {pb.dataloader_pct:.1f}%",
        f"- **Other**: {pb.other_pct:.1f}%",
    ])


def _communication(r: ProfileRecord) -> str:
    if r.communication_overhead_pct <= 0:
        return ""
    overlap = "yes" if r.communication_overlapped else "no"
    return "\n".join([
        "## Communication",
        "",
        f"- **Overhead**: {r.communication_overhead_pct:.1f}%",
        f"- **Overlapped with compute**: {overlap}",
    ])


def _data_loading(r: ProfileRecord) -> str:
    if r.dataloader_stall_pct <= 0:
        return ""
    lines = [
        "## Data Loading",
        "",
        f"- **Stall**: {r.dataloader_stall_pct:.1f}% of step time",
    ]
    dl_config = r.metadata.get("dataloader_config")
    if dl_config and isinstance(dl_config, dict):
        lines.append(f"- **num_workers**: {dl_config.get('num_workers', 'unknown')}")
        lines.append(f"- **pin_memory**: {dl_config.get('pin_memory', 'unknown')}")
    return "\n".join(lines)



def _warnings(r: ProfileRecord) -> str:
    warnings = []
    if r.status != "ok":
        warnings.append(f"Run status: {r.status}")
    if r.error:
        warnings.append(f"Error: {r.error}")
    if r.steps_completed < r.steps_requested:
        warnings.append(f"Only {r.steps_completed}/{r.steps_requested} steps completed")
    if r.gpu_util_mean < 30 and r.gpu_util_samples:
        warnings.append(f"Very low GPU utilization ({r.gpu_util_mean:.0f}%)")

    if not warnings:
        return ""
    lines = ["## Warnings", ""]
    for w in warnings:
        lines.append(f"- {w}")
    return "\n".join(lines)
