"""Collect and normalize events from torch.profiler.

Extracts kernel-level metrics from profiler.key_averages() and converts
them into ProfilerEvent dataclasses.
"""

from __future__ import annotations

from typing import Any

from profine.schema.profile_record import ProfilerEvent

# Cap profiled steps to avoid excessive overhead; scale metrics if capped
PROFILER_ACTIVE_STEPS_CAP = 20

# Fields that get scaled when active steps are capped
_SCALE_FIELDS = (
    "self_cpu_time_total_us",
    "self_cuda_time_total_us",
    "flops",
    "bytes_moved",
)


def collect_events_from_profiler(
    profiler: Any,
    scale_factor: float = 1.0,
) -> list[ProfilerEvent]:
    """Extract events from a torch.profiler.profile instance.

    Args:
        profiler: A torch.profiler.profile context manager (after exiting).
        scale_factor: Multiply time/flops/bytes by this if active steps were capped.

    Returns:
        List of ProfilerEvent dataclasses.
    """
    events: list[ProfilerEvent] = []

    try:
        averages = profiler.key_averages()
    except Exception:
        return events

    for avg in averages:
        name = getattr(avg, "key", "") or str(avg)
        cpu_time = getattr(avg, "self_cpu_time_total", 0.0) or 0.0
        cuda_time = getattr(avg, "self_device_time_total", 0.0) or getattr(avg, "self_cuda_time_total", 0.0) or 0.0
        flops = getattr(avg, "flops", 0.0) or 0.0
        count = getattr(avg, "count", 1) or 1

        # bytes_moved: try .self_device_memory_usage first, fall back to
        # sum of input_shapes sizes × dtype bytes
        bytes_moved = 0.0
        if hasattr(avg, "self_device_memory_usage"):
            bytes_moved = float(getattr(avg, "self_device_memory_usage", 0) or 0)

        input_dtypes: list[str] = []
        raw_dtypes = getattr(avg, "input_type", None) or getattr(avg, "input_dtypes", None)
        if raw_dtypes:
            if isinstance(raw_dtypes, (list, tuple)):
                input_dtypes = [str(d) for d in raw_dtypes]
            else:
                input_dtypes = [str(raw_dtypes)]

        event = ProfilerEvent(
            name=name,
            self_cpu_time_total_us=cpu_time * scale_factor,
            self_cuda_time_total_us=cuda_time * scale_factor,
            flops=flops * scale_factor,
            bytes_moved=bytes_moved * scale_factor,
            count=int(count * scale_factor) if scale_factor != 1.0 else count,
            input_dtypes=input_dtypes,
        )
        events.append(event)

    return events


def compute_scale_factor(active_steps: int) -> float:
    """Compute scale factor if active steps exceed the profiler cap."""
    if active_steps <= PROFILER_ACTIVE_STEPS_CAP:
        return 1.0
    return active_steps / PROFILER_ACTIVE_STEPS_CAP


def compute_profiler_schedule(
    warmup_steps: int,
    active_steps: int,
) -> dict[str, int]:
    """Compute torch.profiler.schedule kwargs.

    Returns dict with wait, warmup, active, repeat suitable for
    torch.profiler.schedule(**result).
    """
    capped_active = min(active_steps, PROFILER_ACTIVE_STEPS_CAP)
    return {
        "wait": 0,
        "warmup": warmup_steps,
        "active": max(capped_active, 1),
        "repeat": 1,
    }


def parse_events_from_payload(raw_events: list[dict[str, Any]]) -> list[ProfilerEvent]:
    """Parse profiler events from a remote payload dict."""
    events: list[ProfilerEvent] = []
    for raw in raw_events:
        events.append(ProfilerEvent(
            name=raw.get("name", ""),
            category=raw.get("category", ""),
            self_cpu_time_total_us=raw.get("self_cpu_time_total_us", 0.0),
            self_cuda_time_total_us=raw.get("self_cuda_time_total_us", 0.0),
            flops=raw.get("flops", 0.0),
            bytes_moved=raw.get("bytes_moved", 0.0),
            count=raw.get("count", 1),
            input_dtypes=raw.get("input_dtypes", []),
        ))
    return events
