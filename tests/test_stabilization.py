"""Tests for warmup stabilization detection."""

from __future__ import annotations

from profine.profiler.stabilization import detect_stabilization_point


def test_short_trace_returns_min_warmup():
    # Trace shorter than window+consecutive — falls back to min_warmup
    assert detect_stabilization_point([1.0, 1.0, 1.0], min_warmup=2) == 2


def test_perfectly_stable_trace_detects_immediately():
    # Once we have window_size + required_consecutive stable points
    # the detector should pick a low index, not the end.
    times = [10.0] * 30
    idx = detect_stabilization_point(times, min_warmup=0)
    assert 0 <= idx < 5


def test_warmup_then_stable_detects_after_warmup():
    # First 10 steps are slow (compile), then stable
    warmup = [50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 12.0, 11.0]
    stable = [10.0] * 30
    times = warmup + stable
    idx = detect_stabilization_point(times, min_warmup=0)
    # Stabilization should land after the warmup section
    assert idx >= 5
    assert idx < len(times)


def test_min_warmup_is_respected_as_floor():
    # Even if stable from step 0, we shouldn't return below min_warmup
    times = [10.0] * 50
    idx = detect_stabilization_point(times, min_warmup=8)
    assert idx >= 8


def test_noisy_trace_falls_back_to_min_warmup():
    # Bouncing between 10 and 30 — never satisfies CV<10%
    times = [10.0, 30.0] * 20
    idx = detect_stabilization_point(times, min_warmup=5)
    assert idx == 5


def test_zero_tail_median_returns_min_warmup():
    # All zero — pathological, should not crash
    idx = detect_stabilization_point([0.0] * 30, min_warmup=3)
    assert idx == 3


def test_strictly_decreasing_warmup_then_flat():
    # Realistic torch.compile pattern: 5x slower at start, flat after step 15
    times = list(range(50, 9, -3)) + [10.0] * 25
    idx = detect_stabilization_point(times, min_warmup=0)
    assert idx >= 10  # well past the warmup section
