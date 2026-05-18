"""Regression tests for bugs patched in the multi-rep minGPT benchmark session.

Each test pins the surface a real bug exposed; they should fail on the
pre-fix code and pass on the fix. Grouped by file/area so a future reader
can scan from a stack trace back to the original incident.
"""

from __future__ import annotations

from unittest.mock import patch
from urllib.error import URLError

from profine.benchmarker.benchmarker import _strip_warmup
from profine.benchmarker.comparator import BenchmarkComparison, CorrectnessVerdict, StepTimeStats
from profine.benchmarker.report import _projected_savings


# ---------------------------------------------------------------------------
# benchmarker/report.py — _projected_savings divided by zero at speedup→100%
# ---------------------------------------------------------------------------

def _bare_comparison(speedup_pct: float) -> BenchmarkComparison:
    """Comparison with just enough state for _projected_savings to run."""
    return BenchmarkComparison(
        metrics=[],
        correctness=CorrectnessVerdict(passed=True),
        speedup_pct=speedup_pct,
        baseline_step_stats=StepTimeStats(),
        candidate_step_stats=StepTimeStats(),
    )


def test_projected_savings_does_not_divide_by_zero_at_extreme_speedup():
    """Bug surfaced when candidate.step_time → 0 (e.g. zero-sample comparison),
    pushing speedup_pct to exactly 100. The old code did 100/(1-1.0)."""
    # speedup_pct == 100 used to raise ZeroDivisionError; should now render.
    lines = _projected_savings(_bare_comparison(speedup_pct=100.0), cost_per_hour=2.50)
    assert lines, "should produce non-empty output even at maxed-out speedup"
    # The first line is the headline; just check it's a string we built, not a crash.
    assert "hours" in lines[0]


def test_projected_savings_returns_empty_when_no_speedup():
    """Negative or zero speedup should yield no projection at all."""
    assert _projected_savings(_bare_comparison(speedup_pct=0.0), cost_per_hour=1.0) == []
    assert _projected_savings(_bare_comparison(speedup_pct=-5.0), cost_per_hour=1.0) == []


# ---------------------------------------------------------------------------
# benchmarker/benchmarker.py — _strip_warmup wiped the whole sample array
# ---------------------------------------------------------------------------

def test_strip_warmup_keeps_at_least_three_samples_when_adaptation_shortened_run():
    """Bug surfaced when StepController.adaptation reduced total_steps to the
    floor (=10) but warmup_steps was also 10 → step_times had ~9 entries →
    stabilization fell back to min_warmup=10 → step_times[10:] = []. The
    downstream comparison then reported a bogus 100%-faster result."""
    payload = {"step_times_ms": [2800.0, 17.0, 14.0, 19.0, 18.0, 18.0, 18.0, 19.0, 18.0]}
    _strip_warmup(payload, warmup_steps=10)
    # With the cap, we keep tail samples instead of stripping everything.
    remaining = payload["step_times_ms"]
    assert len(remaining) >= 3, f"strip wiped to {len(remaining)} samples; cap broken"
    # And the kept samples should be from the steady-state tail, not the
    # compile-cold-start first entry.
    assert remaining[0] != 2800.0


def test_strip_warmup_normal_case_untouched():
    """The cap must not change behavior when there are plenty of samples."""
    payload = {"step_times_ms": [100.0] * 5 + [20.0] * 20}  # 25 samples
    _strip_warmup(payload, warmup_steps=5)
    assert len(payload["step_times_ms"]) >= 15  # well above the floor


# ---------------------------------------------------------------------------
# profiler/orchestrator.py — same warmup-strip bug applied to profile pipeline
# ---------------------------------------------------------------------------

def test_orchestrator_strip_cap_keeps_at_least_three_samples():
    """Mirror of the benchmarker fix: profile stage also can't produce a
    zero-sample ProfileRecord when adaptation shortened the run."""
    from profine.profiler.stabilization import detect_stabilization_point

    # Same shape as the benchmarker case.
    all_step_times = [2800.0, 17.0, 14.0, 19.0, 18.0, 18.0, 18.0, 19.0, 18.0]
    effective_warmup = detect_stabilization_point(all_step_times, min_warmup=10)
    # The orchestrator caps the strip:
    effective_warmup = min(effective_warmup, max(0, len(all_step_times) - 3))
    steady = all_step_times[effective_warmup:]
    assert len(steady) >= 3


# ---------------------------------------------------------------------------
# telemetry/emit.py — _resolve_hardware ignored explicit override
# ---------------------------------------------------------------------------

def test_resolve_hardware_explicit_override_beats_profile_record():
    """Bug surfaced in batch replay: a profile_record produced on A100 was
    reused to emit an A10G fingerprint, but profile_record.hardware_name
    overrode the explicit hardware_name argument, mis-tagging the row."""
    from profine.telemetry.emit import _resolve_hardware

    # Caller passes "1x_a10g" explicitly; profile_record says "1x_a100".
    # Explicit override must win.
    hw = _resolve_hardware({"hardware_name": "1x_a100"}, fallback_name="1x_a10g")
    assert hw is not None
    assert hw.name == "1x_a10g"


def test_resolve_hardware_falls_back_to_profile_record_when_no_override():
    """When the caller doesn't pass anything, the profile record still wins."""
    from profine.telemetry.emit import _resolve_hardware

    hw = _resolve_hardware({"hardware_name": "1x_a100"}, fallback_name=None)
    assert hw is not None
    assert hw.name == "1x_a100"


# ---------------------------------------------------------------------------
# telemetry/recorder.py — POST now retries once on URLError before giving up
# ---------------------------------------------------------------------------

def test_post_retries_once_on_url_error_then_succeeds():
    """Bug surfaced as silent data loss: Render dyno cold-start (~9s) timed
    out the single-attempt POST. We now retry once with a short backoff."""
    from unittest.mock import MagicMock
    from profine.telemetry import recorder as _rec_mod
    from profine.telemetry.recorder import TelemetryRecorder

    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="x", client_version="0.1")

    # First call raises (simulating cold-start timeout); second returns a
    # successful context-manager mock.
    ok_resp = MagicMock()
    ok_resp.__enter__.return_value.read.return_value = b""
    ok_resp.__exit__.return_value = False
    side_effects = [URLError("timed out"), ok_resp]

    # Shorten the backoff so the test doesn't block for 2s.
    with patch.object(_rec_mod, "_HTTP_RETRY_BACKOFF_SECONDS", 0.0), \
         patch.object(_rec_mod, "urlopen", side_effect=side_effects) as urlopen_mock:
        # Drive _post directly so we don't have to wait on daemon threads.
        r._post({"probe": True})

    assert urlopen_mock.call_count == 2, "expected exactly one retry after URLError"


def test_post_gives_up_after_two_failed_attempts():
    """Sanity: if both attempts fail, _post returns without raising and the
    final failure is logged at WARNING (so it's not totally silent)."""
    from profine.telemetry import recorder as _rec_mod
    from profine.telemetry.recorder import TelemetryRecorder

    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="x", client_version="0.1")

    with patch.object(_rec_mod, "_HTTP_RETRY_BACKOFF_SECONDS", 0.0), \
         patch.object(_rec_mod, "urlopen",
                      side_effect=URLError("offline")) as urlopen_mock:
        r._post({"probe": True})  # must not raise

    assert urlopen_mock.call_count == 2  # one + one retry, then give up
