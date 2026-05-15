"""Tests for the run_profile_stats write path.

Three layers:
  1. emit._gather_profile_stats — extracts aggregated stats from
     on-disk profile_record.json + bottleneck_report.json.
  2. TelemetryRecorder.record_profile_stats — accepts a stats dict,
     ships it in the batch payload via the allowlist.
  3. Backend POST /api/telemetry/run accepts the new `profile_stats`
     block and inserts a row into run_profile_stats.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from profine.schema.hardware import HardwareConfig
from profine.telemetry.emit import _gather_profile_stats
from profine.telemetry.fields import (
    ALLOWED_PROFILE_STATS_FIELDS,
    filter_profile_stats,
)
from profine.telemetry.fingerprint import Fingerprint
from profine.telemetry.recorder import TelemetryRecorder


# ===========================================================
# fields.py allowlist
# ===========================================================


def test_allowlist_filter_drops_unknown_profile_keys():
    payload = {
        "step_time_p50_ms": 80.0,
        "memory_peak_gb": 1.4,
        # Unsafe extras a buggy caller might smuggle in:
        "script_path": "/users/foo/secret.py",
        "raw_step_times_ms": [80.1, 80.2, 80.3],
    }
    out = filter_profile_stats(payload)
    assert "script_path" not in out
    assert "raw_step_times_ms" not in out
    assert out["step_time_p50_ms"] == 80.0


def test_allowlist_disjoint_from_other_sets():
    """Same audit guarantee as for fingerprint/outcome: no field
    appears in more than one allowlist."""
    from profine.telemetry.fields import (
        ALLOWED_FINGERPRINT_FIELDS,
        ALLOWED_OUTCOME_FIELDS,
    )
    # runtime_seconds happens to be in both outcomes and profile stats;
    # that's intentional (per-attempt vs per-run total). All other
    # keys must be unique to one allowlist.
    fingerprint_vs_stats = ALLOWED_FINGERPRINT_FIELDS & ALLOWED_PROFILE_STATS_FIELDS
    assert fingerprint_vs_stats == set()
    outcome_vs_stats = ALLOWED_OUTCOME_FIELDS & ALLOWED_PROFILE_STATS_FIELDS
    assert outcome_vs_stats == {"runtime_seconds"}


# ===========================================================
# emit._gather_profile_stats
# ===========================================================


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    for sub in ("read", "profile", "interpret", "edit", "benchmark"):
        (tmp_path / sub).mkdir()
    return tmp_path


def _write(p: Path, payload: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def _hw(vram_gb: float = 80.0, gpu_count: int = 1) -> HardwareConfig:
    return HardwareConfig(
        name="1x_a100", label="A100", modal_gpu="A100-80GB",
        gpu_count=gpu_count, gpu_kind="A100", vram_gb=vram_gb,
        compute_capability="sm80",
    )


def test_extracts_runtime_and_step_distribution(output_dir):
    _write(output_dir / "profile" / "profile_record.json", {
        "runtime_seconds": 79.4,
        "steps_completed": 25,
        "warmup_steps_effective": 10,
        "step_times_ms": [80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 100.0],
        "memory_peak_bytes": 1_533_101_056,
    })
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats is not None
    assert stats["runtime_seconds"] == pytest.approx(79.4)
    assert stats["steps_completed"] == 25
    assert stats["warmup_steps_detected"] == 10
    # p50 of the 10-element series is the midpoint between 88 and 90
    assert stats["step_time_p50_ms"] == pytest.approx(89.0)
    # p95: between 96 and 100 at q=0.95 of n-1=9 → index 8.55 → 96 + 0.55*4 = 98.2
    assert stats["step_time_p95_ms"] == pytest.approx(98.2)
    # CV is small (smooth ramp)
    assert 0 < stats["step_time_cv"] < 0.1


def test_extracts_memory_peak_in_gb_and_pct(output_dir):
    _write(output_dir / "profile" / "profile_record.json", {
        "memory_peak_bytes": 8 * (1024 ** 3),  # 8 GB
    })
    stats = _gather_profile_stats(output_dir, _hw(vram_gb=80.0))
    assert stats["memory_peak_gb"] == pytest.approx(8.0)
    assert stats["memory_peak_pct"] == pytest.approx(10.0)


def test_uses_total_vram_for_multi_gpu(output_dir):
    _write(output_dir / "profile" / "profile_record.json", {
        "memory_peak_bytes": 16 * (1024 ** 3),
    })
    # 4x A100 80GB = 320GB total
    stats = _gather_profile_stats(output_dir, _hw(vram_gb=80.0, gpu_count=4))
    assert stats["memory_peak_gb"] == pytest.approx(16.0)
    assert stats["memory_peak_pct"] == pytest.approx(5.0)


def test_extracts_gpu_utilization_percentiles(output_dir):
    _write(output_dir / "profile" / "profile_record.json", {
        "gpu_util_samples": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0],
    })
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats["gpu_util_p50_pct"] == pytest.approx(55.0)
    assert stats["gpu_util_p95_pct"] == pytest.approx(92.75)


def test_computes_compute_pct_from_overheads(output_dir):
    _write(output_dir / "profile" / "profile_record.json", {
        "step_times_ms": [10.0, 11.0],
        "dataloader_stall_pct": 8.0,
        "communication_overhead_pct": 4.0,
    })
    stats = _gather_profile_stats(output_dir, _hw())
    # 100 - 8 - 4 = 88
    assert stats["compute_pct"] == pytest.approx(88.0)


def test_clamps_compute_pct_at_zero(output_dir):
    """Pathological inputs (sum > 100) must clamp to 0, not go negative."""
    _write(output_dir / "profile" / "profile_record.json", {
        "step_times_ms": [10.0, 11.0],
        "dataloader_stall_pct": 70.0,
        "communication_overhead_pct": 50.0,  # 70 + 50 > 100
    })
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats["compute_pct"] == pytest.approx(0.0)


def test_classifies_dataloader_bottleneck_from_flags(output_dir):
    _write(output_dir / "profile" / "profile_record.json", {
        "step_times_ms": [10.0, 11.0],
    })
    _write(output_dir / "interpret" / "bottleneck_report.json", {
        "data_pipeline_bound": True,
        "compute_bound": False,
    })
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats["primary_bottleneck"] == "dataloader"


def test_classifies_compute_bottleneck_when_flags_say_so(output_dir):
    _write(output_dir / "profile" / "profile_record.json", {
        "step_times_ms": [10.0, 11.0],
    })
    _write(output_dir / "interpret" / "bottleneck_report.json", {
        "compute_bound": True,
        "data_pipeline_bound": False,
    })
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats["primary_bottleneck"] == "compute"


def test_unwraps_nested_bottleneck_report_shape(output_dir):
    """run-all output wraps the report under a 'bottleneck_report' key."""
    _write(output_dir / "profile" / "profile_record.json", {
        "step_times_ms": [10.0, 11.0],
    })
    _write(output_dir / "interpret" / "bottleneck_report.json", {
        "bottleneck_report": {
            "memory_bandwidth_bound": True,
        }
    })
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats["primary_bottleneck"] == "memory_bandwidth"


def test_falls_back_to_pct_heuristic_when_no_report(output_dir):
    """Without bottleneck_report, classify from raw percentages."""
    _write(output_dir / "profile" / "profile_record.json", {
        "step_times_ms": [10.0, 11.0],
        "dataloader_stall_pct": 25.0,
    })
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats["primary_bottleneck"] == "dataloader"


def test_returns_none_when_profile_record_missing(output_dir):
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats is None


def test_ignores_step_times_with_zero_or_nan(output_dir):
    _write(output_dir / "profile" / "profile_record.json", {
        "step_times_ms": [0, -1.0, float("nan"), 10.0, 12.0],
    })
    stats = _gather_profile_stats(output_dir, _hw())
    # Only 10.0 and 12.0 survive — p50 = 11.0
    assert stats["step_time_p50_ms"] == pytest.approx(11.0)


def test_tolerates_corrupt_profile_json(output_dir):
    (output_dir / "profile" / "profile_record.json").write_text("not json{")
    stats = _gather_profile_stats(output_dir, _hw())
    assert stats is None


# ===========================================================
# TelemetryRecorder.record_profile_stats
# ===========================================================


def _fp() -> Fingerprint:
    return Fingerprint(
        arch_class="transformer-decoder", param_bucket="100M-1B",
        hardware_class="1x_a100", precision="mixed_bf16",
        optimizer_class="adam_family", has_compile=True, has_distributed=False,
        fingerprint_hash="c" * 64,
    )


def test_recorder_attaches_profile_stats_to_batch():
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="pf_live_x",
                          client_version="0.1")
    r.begin_run(_fp())
    r.record_optimization("compile_default", catalog_version="v1", applied=True)
    r.record_profile_stats({
        "step_time_p50_ms": 80.0, "memory_peak_gb": 1.4,
        "primary_bottleneck": "compute",
    })
    payload = r._drain_payload()
    assert "profile_stats" in payload
    assert payload["profile_stats"]["step_time_p50_ms"] == 80.0
    assert payload["profile_stats"]["primary_bottleneck"] == "compute"


def test_recorder_profile_stats_alone_is_a_valid_batch():
    """No optimization outcomes; just the fingerprint + profile stats.
    Should still flush (a profile-only run is real telemetry).
    """
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="pf_live_x",
                          client_version="0.1")
    r.begin_run(_fp())
    r.record_profile_stats({"step_time_p50_ms": 80.0})
    payload = r._drain_payload()
    assert payload is not None
    assert payload["outcomes"] == []
    assert payload["profile_stats"]["step_time_p50_ms"] == 80.0


def test_recorder_drains_then_clears_profile_stats():
    """A second drain after the first returns None (nothing pending)."""
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="pf_live_x",
                          client_version="0.1")
    r.begin_run(_fp())
    r.record_profile_stats({"step_time_p50_ms": 80.0})
    assert r._drain_payload() is not None
    assert r._drain_payload() is None


def test_recorder_filters_unknown_profile_keys():
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="pf_live_x",
                          client_version="0.1")
    r.begin_run(_fp())
    r.record_profile_stats({
        "step_time_p50_ms": 80.0,
        "script_path": "/secret",      # unsafe
        "raw_step_times_ms": [80, 81], # unsafe
    })
    payload = r._drain_payload()
    assert payload is not None
    assert "script_path" not in payload["profile_stats"]
    assert "raw_step_times_ms" not in payload["profile_stats"]


def test_recorder_replaces_profile_stats_on_double_call():
    """A pipeline could compute then refine — keep the last submission."""
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="pf_live_x",
                          client_version="0.1")
    r.begin_run(_fp())
    r.record_profile_stats({"step_time_p50_ms": 80.0})
    r.record_profile_stats({"step_time_p50_ms": 75.0, "memory_peak_gb": 2.0})
    payload = r._drain_payload()
    assert payload["profile_stats"] == {"step_time_p50_ms": 75.0, "memory_peak_gb": 2.0}


def test_recorder_record_profile_stats_is_noop_when_disabled():
    r = TelemetryRecorder(api_url="https://api.profine.ai", enabled=False)
    r.begin_run(_fp())
    r.record_profile_stats({"step_time_p50_ms": 80.0})
    assert r._drain_payload() is None
