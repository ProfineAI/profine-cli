"""Round-trip tests for every schema dataclass.

The CLI persists ProfileRecord/BottleneckReport/OptimizationCandidate as
JSON and reads them back via _dict_to_* helpers in cli/commands.py. If
those round-trips silently lose fields, downstream stages get partial
data and produce confusing output without an error. Lock the contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from profine.cli.commands import (
    _dict_to_bottleneck_report,
    _dict_to_candidate,
    _dict_to_profile_record,
)
from profine.schema.bottleneck_report import BottleneckEntry, BottleneckReport
from profine.schema.optimization_candidate import OptimizationCandidate
from profine.schema.profile_record import (
    KernelCategoryBreakdown,
    KernelSummary,
    PhaseBreakdown,
    ProfileRecord,
    ProfilerEvent,
)


def test_profile_record_roundtrip_preserves_core_fields():
    record = ProfileRecord(
        status="ok",
        script_path="train.py",
        hardware_name="1x_a100",
        steps_requested=60,
        steps_completed=60,
        warmup_steps_requested=30,
        warmup_steps_effective=12,
        runtime_seconds=42.5,
        step_times_ms=[10.0, 9.5, 9.7],
        loss_values=[1.0, 0.9, 0.85],
        gpu_util_samples=[80.0, 82.0],
        gpu_util_mean=81.0,
        gpu_util_pattern="sustained",
        memory_peak_bytes=8 * 1024**3,
    )
    d = json.loads(json.dumps(asdict(record), default=str))
    rebuilt = _dict_to_profile_record(d)
    assert rebuilt.status == record.status
    assert rebuilt.script_path == record.script_path
    assert rebuilt.steps_completed == record.steps_completed
    assert rebuilt.step_times_ms == record.step_times_ms
    assert rebuilt.loss_values == record.loss_values
    assert rebuilt.gpu_util_pattern == record.gpu_util_pattern
    assert rebuilt.memory_peak_bytes == record.memory_peak_bytes


def test_profile_record_roundtrip_with_nested_dataclasses():
    record = ProfileRecord(
        script_path="train.py",
        hardware_name="1x_h100",
        profiler_events=[ProfilerEvent(name="aten::matmul", self_cuda_time_total_us=100.0)],
        top_kernels=[KernelSummary(name="ampere_sgemm", category="matmul", cuda_time_us=100.0, pct_of_total=50.0)],
        kernel_breakdown=KernelCategoryBreakdown(matmul_pct=50.0, other_pct=50.0),
        phase_breakdown=PhaseBreakdown(forward_pct=40.0, backward_pct=50.0, optimizer_pct=10.0),
    )
    d = json.loads(json.dumps(asdict(record), default=str))
    rebuilt = _dict_to_profile_record(d)
    assert len(rebuilt.profiler_events) == 1
    assert rebuilt.profiler_events[0].name == "aten::matmul"
    assert len(rebuilt.top_kernels) == 1
    assert rebuilt.top_kernels[0].category == "matmul"
    assert rebuilt.kernel_breakdown.matmul_pct == 50.0
    assert rebuilt.phase_breakdown.optimizer_pct == 10.0


def test_profile_record_roundtrip_handles_missing_optional_fields():
    # Saved records from older runs may lack newer fields — should fill defaults
    minimal = {"script_path": "x.py", "hardware_name": "1x_t4"}
    rebuilt = _dict_to_profile_record(minimal)
    assert rebuilt.status == "ok"
    assert rebuilt.steps_completed == 0
    assert rebuilt.kernel_breakdown is None


def test_bottleneck_report_roundtrip():
    report = BottleneckReport(
        bottlenecks=[
            BottleneckEntry(
                category="attention",
                location="flash_fwd_kernel",
                time_share_pct=42.0,
                est_headroom_pct=20.0,
                confidence="observed",
            ),
            BottleneckEntry(
                category="dataloader",
                location="DataLoader stall",
                time_share_pct=15.0,
                est_headroom_pct=12.0,
            ),
        ],
        compute_bound=True,
        summary="GPU is mostly busy with attention.",
    )
    d = json.loads(json.dumps(asdict(report)))
    rebuilt = _dict_to_bottleneck_report(d)
    assert len(rebuilt.bottlenecks) == 2
    assert rebuilt.bottlenecks[0].category == "attention"
    assert rebuilt.compute_bound is True
    assert rebuilt.bottlenecks[1].confidence == "observed"  # schema default


def test_optimization_candidate_roundtrip():
    cand = OptimizationCandidate(
        entry_id="flash_attention_2",
        category="attention",
        name="FlashAttention-2",
        description="Fused attention.",
        rank=1,
        priority="high",
        est_speedup_low_pct=20.0,
        est_speedup_high_pct=40.0,
        rationale="attention is 42% of step",
        bottlenecks_addressed=["attention"],
        risks=["may need flash-attn binary"],
        code_pattern="attn_implementation='flash_attention_2'",
        estimated_effort="trivial",
    )
    d = json.loads(json.dumps(asdict(cand)))
    rebuilt = _dict_to_candidate(d)
    assert rebuilt.entry_id == cand.entry_id
    assert rebuilt.priority == cand.priority
    assert rebuilt.bottlenecks_addressed == ["attention"]
    assert rebuilt.risks == ["may need flash-attn binary"]


def test_unknown_extra_keys_in_dict_are_ignored():
    # Forward compat: extra keys from a newer producer must not break older readers
    d = {
        "script_path": "x.py",
        "hardware_name": "1x_a100",
        "future_field_added_later": "ignore me",
    }
    rebuilt = _dict_to_profile_record(d)
    assert rebuilt.script_path == "x.py"
