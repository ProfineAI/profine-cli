"""Compute performance heuristics from raw profiler data.

Combines GPU utilization analysis, kernel categorization, and
phase breakdown into a single module.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from profine.config.yaml_loader import (
    get_attention_impl_map,
    get_categories,
    get_exclude_patterns,
    get_precision_map,
)
from profine.schema.profile_record import (
    KernelCategoryBreakdown,
    KernelSummary,
    PhaseBreakdown,
    ProfilerEvent,
)

def _is_excluded(name: str) -> bool:
    """Check if an event is a profiler-internal marker to skip."""
    return any(p in name for p in get_exclude_patterns())


def _categorize_kernel(name: str) -> str:
    lower = name.lower()
    for category, patterns in get_categories().items():
        if any(p.lower() in lower for p in patterns):
            return category
    return "other"


def classify_gpu_pattern(samples: list[float]) -> str:
    if not samples:
        return "unknown"
    normalized = [s / 100.0 if max(samples) > 1.0 else s for s in samples]
    high_pct = sum(1 for s in normalized if s > 0.8) / len(normalized)
    if high_pct > 0.7:
        return "sustained"
    low_pct = sum(1 for s in normalized if s < 0.6) / len(normalized)
    if low_pct > 0.9:
        return "low_flat"
    transitions = sum(1 for i in range(1, len(normalized))
                      if abs(normalized[i] - normalized[i-1]) > 0.3)
    if transitions >= 4:
        return "periodic_gaps"
    return "mixed"


def compute_gpu_mean(samples: list[float]) -> float:
    if not samples:
        return 0.0
    vals = [s / 100.0 if max(samples) > 1.0 else s for s in samples]
    return sum(vals) / len(vals) * 100.0


def compute_top_kernels(events: list[ProfilerEvent], top_k: int = 15) -> list[KernelSummary]:
    aggregated: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cuda_us": 0.0, "count": 0, "flops": 0.0, "bytes": 0.0}
    )
    for e in events:
        cuda = e.self_cuda_time_total_us or 0.0
        if cuda <= 0 or _is_excluded(e.name):
            continue
        agg = aggregated[e.name]
        agg["cuda_us"] += cuda
        agg["count"] += e.count
        agg["flops"] += e.flops
        agg["bytes"] += e.bytes_moved

    total_cuda = sum(a["cuda_us"] for a in aggregated.values())
    if total_cuda <= 0:
        return []

    sorted_kernels = sorted(aggregated.items(), key=lambda x: x[1]["cuda_us"], reverse=True)
    return [
        KernelSummary(
            name=name,
            category=_categorize_kernel(name),
            cuda_time_us=data["cuda_us"],
            pct_of_total=data["cuda_us"] / total_cuda * 100.0,
            count=data["count"],
            flops=data["flops"],
            bytes_moved=data["bytes"],
        )
        for name, data in sorted_kernels[:top_k]
    ]


def compute_category_breakdown(events: list[ProfilerEvent]) -> KernelCategoryBreakdown:
    category_time: dict[str, float] = defaultdict(float)
    for e in events:
        cuda = e.self_cuda_time_total_us or 0.0
        if cuda <= 0 or _is_excluded(e.name):
            continue
        category_time[_categorize_kernel(e.name)] += cuda

    total = sum(category_time.values())
    if total <= 0:
        return KernelCategoryBreakdown()

    def pct(cat: str) -> float:
        return category_time.get(cat, 0.0) / total * 100.0

    return KernelCategoryBreakdown(
        matmul_pct=pct("matmul"),
        attention_pct=pct("attention"),
        elementwise_pct=pct("elementwise"),
        normalization_pct=pct("normalization"),
        optimizer_pct=pct("optimizer"),
        communication_pct=pct("communication"),
        memory_pct=pct("memory"),
        dataloader_pct=pct("dataloader"),
        other_pct=pct("other"),
    )


def compute_phase_breakdown(events: list[ProfilerEvent]) -> PhaseBreakdown:
    phase_time: dict[str, float] = defaultdict(float)
    for e in events:
        t = e.self_cuda_time_total_us or e.self_cpu_time_total_us or 0.0
        if t <= 0 or _is_excluded(e.name):
            continue
        lower = e.name.lower()
        if any(p in lower for p in ("backward", "autograd", "bwd", "grad")):
            phase_time["backward"] += t
        elif _categorize_kernel(e.name) == "optimizer":
            phase_time["optimizer"] += t
        elif _categorize_kernel(e.name) == "dataloader":
            phase_time["dataloader"] += t
        else:
            phase_time["forward"] += t

    total = sum(phase_time.values())
    if total <= 0:
        return PhaseBreakdown()

    return PhaseBreakdown(
        forward_pct=phase_time.get("forward", 0) / total * 100,
        backward_pct=phase_time.get("backward", 0) / total * 100,
        optimizer_pct=phase_time.get("optimizer", 0) / total * 100,
        dataloader_pct=phase_time.get("dataloader", 0) / total * 100,
        other_pct=phase_time.get("other", 0) / total * 100,
    )


def compute_dataloader_stall_pct(events: list[ProfilerEvent], step_time_total_us: float) -> float:
    if step_time_total_us <= 0:
        return 0.0
    stall_us = sum(
        e.self_cpu_time_total_us for e in events
        if any(p in e.name.lower() for p in ("dataloader", "__next__", "pin_memory"))
    )
    return min(stall_us / step_time_total_us * 100.0, 100.0)


def compute_arithmetic_intensity(events: list[ProfilerEvent]) -> float | None:
    total_flops = sum(e.flops for e in events if e.flops > 0)
    total_bytes = sum(e.bytes_moved for e in events if e.bytes_moved > 0)
    if total_flops <= 0 or total_bytes <= 0:
        return None
    return total_flops / total_bytes


def detect_attention_impl(events: list[ProfilerEvent]) -> str:
    impl_map = get_attention_impl_map()
    for impl_name, patterns in impl_map.items():
        for e in events:
            lower = e.name.lower()
            if any(p.lower() in lower for p in patterns):
                return impl_name
    return "unknown"


def detect_precision(events: list[ProfilerEvent]) -> str:
    # input_dtypes are the most reliable precision signal.
    dtype_counts: dict[str, int] = defaultdict(int)
    for e in events:
        for dt in e.input_dtypes:
            dtype_counts[dt.lower()] += 1
    if dtype_counts:
        most_common = max(dtype_counts, key=dtype_counts.get)
        if "bfloat16" in most_common:
            return "bf16"
        if "float16" in most_common or "half" in most_common:
            return "fp16"
        if "float32" in most_common:
            return "fp32"
        return most_common

    # Fallback: infer from kernel names (e.g. sgemm = fp32)
    precision_map = get_precision_map()
    kernel_names = " ".join(e.name.lower() for e in events if e.self_cuda_time_total_us > 0)
    for precision, patterns in precision_map.items():
        if any(p.lower() in kernel_names for p in patterns):
            return precision

    return "unknown"


def compute_communication_overhead(events: list[ProfilerEvent]) -> tuple[float, bool]:
    """Returns (overhead_pct, is_overlapped)."""
    total_cuda = sum(e.self_cuda_time_total_us for e in events if e.self_cuda_time_total_us > 0)
    if total_cuda <= 0:
        return 0.0, False
    comm_patterns = {"nccl", "all_reduce", "reduce_scatter", "all_gather", "broadcast"}
    comm_time = sum(
        e.self_cuda_time_total_us for e in events
        if any(p in e.name.lower() for p in comm_patterns)
    )
    overhead_pct = comm_time / total_cuda * 100.0
    # Rough overlap detection: if comm is present but < 5% it's likely overlapped
    overlapped = 0 < comm_time and overhead_pct < 5.0
    return overhead_pct, overlapped


def compute_memory_headroom(peak_bytes: int, vram_gb: float) -> float:
    if vram_gb <= 0:
        return 0.0
    peak_gb = peak_bytes / (1024 ** 3)
    return max(0.0, (1 - peak_gb / vram_gb)) * 100.0
