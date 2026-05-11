"""Tests for profiler heuristic functions.

These are pure functions over ProfilerEvent lists / sample arrays,
so we exercise them directly without going through Modal.
"""

from __future__ import annotations

from profine.profiler.heuristics import (
    classify_gpu_pattern,
    compute_arithmetic_intensity,
    compute_category_breakdown,
    compute_communication_overhead,
    compute_dataloader_stall_pct,
    compute_gpu_mean,
    compute_memory_headroom,
    compute_phase_breakdown,
    compute_top_kernels,
    detect_attention_impl,
    detect_precision,
)
from profine.schema.profile_record import ProfilerEvent


def _ev(name: str, cuda_us: float = 0.0, cpu_us: float = 0.0, flops: float = 0.0,
        bytes_moved: float = 0.0, count: int = 1, dtypes: list[str] | None = None) -> ProfilerEvent:
    return ProfilerEvent(
        name=name,
        self_cuda_time_total_us=cuda_us,
        self_cpu_time_total_us=cpu_us,
        flops=flops,
        bytes_moved=bytes_moved,
        count=count,
        input_dtypes=dtypes or [],
    )


def test_classify_gpu_pattern_empty():
    assert classify_gpu_pattern([]) == "unknown"


def test_classify_gpu_pattern_sustained():
    assert classify_gpu_pattern([90.0] * 10) == "sustained"


def test_classify_gpu_pattern_low_flat():
    assert classify_gpu_pattern([20.0] * 10) == "low_flat"


def test_classify_gpu_pattern_periodic():
    assert classify_gpu_pattern([90.0, 10.0] * 5) == "periodic_gaps"


def test_compute_gpu_mean_normalizes_percent_scale():
    # When samples are in 0-100 percent we get the percent back
    assert abs(compute_gpu_mean([50.0, 50.0, 50.0]) - 50.0) < 0.01


def test_compute_gpu_mean_empty():
    assert compute_gpu_mean([]) == 0.0


def test_compute_top_kernels_orders_by_cuda_time():
    events = [
        _ev("small", cuda_us=10.0),
        _ev("big", cuda_us=1000.0),
        _ev("medium", cuda_us=100.0),
    ]
    top = compute_top_kernels(events, top_k=2)
    assert top[0].name == "big"
    assert top[1].name == "medium"


def test_compute_top_kernels_pct_sums_to_at_most_100():
    events = [_ev("a", cuda_us=100.0), _ev("b", cuda_us=200.0)]
    top = compute_top_kernels(events)
    total_pct = sum(k.pct_of_total for k in top)
    assert total_pct <= 100.01


def test_compute_category_breakdown_empty():
    bd = compute_category_breakdown([])
    assert bd.matmul_pct == 0.0
    assert bd.other_pct == 0.0


def test_compute_phase_breakdown_empty():
    pb = compute_phase_breakdown([])
    assert pb.forward_pct == 0.0
    assert pb.backward_pct == 0.0


def test_compute_dataloader_stall_pct_zero_total():
    assert compute_dataloader_stall_pct([], step_time_total_us=0.0) == 0.0


def test_compute_arithmetic_intensity_no_data():
    assert compute_arithmetic_intensity([]) is None


def test_compute_arithmetic_intensity_basic():
    # 1000 flops over 100 bytes = 10 flops/byte
    events = [_ev("matmul", flops=1000.0, bytes_moved=100.0)]
    ai = compute_arithmetic_intensity(events)
    assert ai is not None
    assert ai > 0


def test_detect_attention_impl_unknown_when_no_match():
    assert detect_attention_impl([_ev("aten::matmul")]) == "unknown"


def test_detect_precision_unknown_when_no_dtypes():
    assert detect_precision([_ev("op")]) == "unknown"


def test_compute_communication_overhead_no_comm_kernels():
    overhead, overlapped = compute_communication_overhead([_ev("matmul", cuda_us=100.0)])
    assert overhead == 0.0
    assert overlapped is False


def test_compute_memory_headroom():
    # Used 16 GB out of 80 GB → 80% headroom
    headroom = compute_memory_headroom(peak_bytes=16 * 1024**3, vram_gb=80.0)
    assert 70.0 <= headroom <= 90.0


def test_compute_memory_headroom_zero_vram():
    assert compute_memory_headroom(peak_bytes=1, vram_gb=0.0) == 0.0
