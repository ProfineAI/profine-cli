"""Tests for event_collector helpers (no real torch.profiler)."""

from __future__ import annotations

from profine.profiler.event_collector import (
    PROFILER_ACTIVE_STEPS_CAP,
    compute_profiler_schedule,
    compute_scale_factor,
    parse_events_from_payload,
)


def test_scale_factor_below_cap_is_one():
    assert compute_scale_factor(PROFILER_ACTIVE_STEPS_CAP - 1) == 1.0
    assert compute_scale_factor(PROFILER_ACTIVE_STEPS_CAP) == 1.0


def test_scale_factor_above_cap_proportional():
    factor = compute_scale_factor(PROFILER_ACTIVE_STEPS_CAP * 3)
    assert factor == 3.0


def test_schedule_capped_at_active_steps():
    sched = compute_profiler_schedule(warmup_steps=5, active_steps=PROFILER_ACTIVE_STEPS_CAP * 5)
    assert sched["active"] == PROFILER_ACTIVE_STEPS_CAP
    assert sched["warmup"] == 5
    assert sched["repeat"] == 1


def test_schedule_min_active_is_one():
    sched = compute_profiler_schedule(warmup_steps=0, active_steps=0)
    assert sched["active"] == 1


def test_parse_events_from_payload_basic():
    raw = [
        {"name": "op_a", "self_cuda_time_total_us": 100.0, "flops": 5.0},
        {"name": "op_b", "category": "matmul", "count": 3},
    ]
    events = parse_events_from_payload(raw)
    assert len(events) == 2
    assert events[0].name == "op_a"
    assert events[0].self_cuda_time_total_us == 100.0
    assert events[0].flops == 5.0
    assert events[1].category == "matmul"
    assert events[1].count == 3


def test_parse_events_from_payload_empty():
    assert parse_events_from_payload([]) == []


def test_parse_events_handles_missing_fields_with_defaults():
    events = parse_events_from_payload([{"name": "x"}])
    assert events[0].name == "x"
    assert events[0].self_cuda_time_total_us == 0.0
    assert events[0].count == 1
    assert events[0].input_dtypes == []
