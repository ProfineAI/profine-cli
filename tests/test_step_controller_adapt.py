"""Probe-and-adapt logic in StepController.

Tests the data-driven step-count reduction: when observed step time
implies the requested total won't fit the wall-clock budget, the
controller lowers `total_steps` so the run ends cleanly instead of
being cut by the wall-clock cap mid-step.
"""

from __future__ import annotations

from profine.profiler.hooks import StepController


def _drive(controller: StepController, step_times_s: list[float]) -> None:
    """Manually advance the controller `len(step_times_s)` times.

    We bypass `install()` (which monkey-patches torch.optim) and instead
    poke the internal counters directly, so the test doesn't need a GPU.
    """
    t = 0.0
    controller._wall_start = 0.0
    for dt in step_times_s:
        t += dt
        controller.steps_completed += 1
        controller._maybe_adapt(elapsed=t)


def test_adapts_down_when_step_time_overshoots_budget():
    # 60 requested, 60s budget, 5s/step → projected 300s, only ~12 fit.
    sc = StepController(total_steps=60, wall_clock_limit_s=60.0)
    sc._probe_steps = 3
    sc._adapt_floor = 5
    _drive(sc, [5.0, 5.0, 5.0])
    # After 3 probe steps + observed 5s avg + 45s budget left → fits=9.
    # new_total = 3 + 9 = 12, which is below the original 60.
    assert sc.total_steps == 12
    assert sc._adapted is True


def test_does_not_adapt_when_run_fits_budget():
    # 60 requested, 600s budget, 1s/step → 600s = exactly fits.
    sc = StepController(total_steps=60, wall_clock_limit_s=600.0)
    sc._probe_steps = 3
    _drive(sc, [1.0, 1.0, 1.0])
    # 60 × 1s = 60s << 600s budget, no reduction.
    assert sc.total_steps == 60
    assert sc._adapted is True


def test_floor_prevents_reducing_too_aggressively():
    # Crazy slow steps would suggest 1-2 step total; floor must hold.
    sc = StepController(total_steps=60, wall_clock_limit_s=30.0)
    sc._probe_steps = 3
    sc._adapt_floor = 10
    _drive(sc, [10.0, 10.0, 10.0])
    # 3 steps × 10s = 30s already exhausted budget. fits=0 → would
    # propose new_total=3, but floor=10 keeps it at 10.
    assert sc.total_steps == 10


def test_adaptation_runs_only_once():
    sc = StepController(total_steps=60, wall_clock_limit_s=60.0)
    sc._probe_steps = 3
    sc._adapt_floor = 5
    _drive(sc, [5.0, 5.0, 5.0])  # first adapt fires here → total becomes 12
    first_total = sc.total_steps
    _drive(sc, [5.0, 5.0])  # subsequent steps must not re-lower
    assert sc.total_steps == first_total
