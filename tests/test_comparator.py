"""Tests for benchmark comparison & verdict logic."""

from __future__ import annotations

from profine.benchmarker.comparator import (
    SPEEDUP_PASS_PCT,
    SPEEDUP_REG_PCT,
    UTIL_DROP_REG_PCT,
    _check_correctness,
    _decide_verdict,
    _make_delta,
    compare_payloads,
)


def test_verdict_correctness_failure_overrides_speedup():
    # Correctness fails on a winning speedup → surface both signals
    assert _decide_verdict(speedup_pct=50.0, util_delta_pct=0, correctness_passed=False) == "FAIL (correctness; speedup measured but loss diverged)"
    # Correctness fails without a winning speedup → REGRESSION
    assert _decide_verdict(speedup_pct=1.0, util_delta_pct=0, correctness_passed=False) == "REGRESSION"


def test_verdict_pass_threshold():
    assert _decide_verdict(SPEEDUP_PASS_PCT, 0, True) == "PASS"
    assert _decide_verdict(SPEEDUP_PASS_PCT - 0.1, 0, True) == "NO-OP"


def test_verdict_regression_threshold():
    assert _decide_verdict(SPEEDUP_REG_PCT, 0, True) == "REGRESSION"
    assert _decide_verdict(SPEEDUP_REG_PCT + 0.1, 0, True) == "NO-OP"


def test_verdict_util_drop_is_regression_only_without_speedup():
    # Util drop with a flat speedup = REGRESSION (suspicious: GPU idle for no reason).
    assert _decide_verdict(
        speedup_pct=1.0, util_delta_pct=UTIL_DROP_REG_PCT, correctness_passed=True,
    ) == "REGRESSION"
    # Util drop with a clear speedup = PASS. The util drop is explained by the
    # optimization finishing each step faster, leaving the GPU idle longer
    # between steps. Flagging this as REGRESSION misleads users.
    assert _decide_verdict(
        speedup_pct=50.0, util_delta_pct=UTIL_DROP_REG_PCT, correctness_passed=True,
    ) == "PASS"


def test_verdict_no_op_band():
    assert _decide_verdict(speedup_pct=1.0, util_delta_pct=0, correctness_passed=True) == "NO-OP"
    assert _decide_verdict(speedup_pct=0.0, util_delta_pct=0, correctness_passed=True) == "NO-OP"


def test_make_delta_lower_is_better():
    d = _make_delta("step_ms", baseline=100.0, candidate=80.0, lower_is_better=True)
    assert d.delta == -20.0
    assert d.delta_pct == -20.0
    assert d.improved is True


def test_make_delta_higher_is_better():
    d = _make_delta("util", baseline=50.0, candidate=70.0, lower_is_better=False)
    assert d.improved is True
    assert d.delta_pct == 40.0


def test_make_delta_zero_baseline_safe():
    d = _make_delta("x", 0.0, 5.0, lower_is_better=False)
    assert d.delta_pct == 0.0  # no division by zero


def test_correctness_no_losses_passes_with_caveat():
    v = _check_correctness([], [1.0], rtol=1e-2, atol=1e-4)
    assert v.passed
    assert "not verified" in v.notes


def test_correctness_matching_curves_pass():
    v = _check_correctness([1.0, 0.5, 0.25], [1.0, 0.5, 0.25], rtol=1e-2, atol=1e-4)
    assert v.passed
    assert v.max_loss_diff == 0.0


def test_correctness_diverging_curves_fail():
    v = _check_correctness([1.0, 0.5, 0.25], [1.0, 0.5, 1.0], rtol=1e-2, atol=1e-4)
    assert not v.passed
    assert "diverged at step" in v.notes


def test_correctness_within_relative_tolerance_passes():
    # 0.5% drift on values around 1.0 — within rtol=1e-2
    v = _check_correctness([1.0, 1.0, 1.0], [1.005, 0.995, 1.003], rtol=1e-2, atol=1e-4)
    assert v.passed


def test_correctness_overlapping_window_only():
    # candidate is shorter — only overlapping steps are checked, not padded
    v = _check_correctness([1.0, 0.5, 0.25, 0.1], [1.0, 0.5], rtol=1e-2, atol=1e-4)
    assert v.passed


def test_compare_payloads_full_pipeline_pass():
    baseline = {
        "step_times_ms": [10.0] * 10,
        "memory_peak_bytes": 1_000_000_000,
        "gpu_utilization_samples": [60.0] * 10,
        "loss_values": [1.0, 0.9, 0.8],
    }
    candidate = {
        "step_times_ms": [7.0] * 10,  # 30% faster
        "memory_peak_bytes": 800_000_000,
        "gpu_utilization_samples": [80.0] * 10,
        "loss_values": [1.0, 0.9, 0.8],
    }
    cmp = compare_payloads(baseline, candidate, rtol=1e-2, atol=1e-4)
    assert cmp.verdict == "PASS"
    assert cmp.speedup_pct > 0
    assert cmp.correctness.passed


def test_compare_payloads_correctness_fail_flags_correctness():
    baseline = {"step_times_ms": [10.0] * 5, "loss_values": [1.0]}
    candidate = {"step_times_ms": [5.0] * 5, "loss_values": [99.0]}  # diverged
    cmp = compare_payloads(baseline, candidate)
    assert cmp.verdict == "FAIL (correctness; speedup measured but loss diverged)"
    assert not cmp.correctness.passed
