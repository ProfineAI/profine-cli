"""Tests for emit_run() — the artifact-reader path.

We write fake JSON artifacts to a tmpdir matching the layout the
pipeline produces, then verify the recorder receives the right
fingerprint and outcome rows. The recorder itself is mocked so
nothing leaves the process.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from profine.catalog import CATALOG_VERSION
from profine.telemetry.emit import emit_run


# ----------------------------- fixtures -----------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """An output_dir/{read,profile,edit,benchmark}/ skeleton."""
    for sub in ("read", "profile", "edit", "benchmark"):
        (tmp_path / sub).mkdir()
    return tmp_path


def _write(p: Path, payload: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def _arch_payload() -> dict:
    """A minimal architecture_record.json compatible with the loader."""
    return {
        "framework": {"value": "raw_pytorch"},
        "model_family": {"value": "GPT"},
        "model_class": {"value": "GPT"},
        "attention_type": {"value": "causal_mha"},
        "attention_impl": {"value": "manual"},
        "num_layers": {"value": 12},
        "hidden_size": {"value": 768},
        "vocab_size": {"value": 50257},
        "num_heads": {"value": 12},
        "compile_mode": {"value": "default"},
        "optimizer": {"name": {"value": "AdamW"}},
        "precision": {
            "training_dtype": {"value": "bf16"},
            "autocast_enabled": {"value": True},
        },
        "distributed": {"strategy": {"value": "none"}},
    }


def _profile_payload(hardware_name: str = "1x_a100") -> dict:
    return {"hardware_name": hardware_name, "status": "ok"}


def _manifest_payload(applied=("compile_default",), skipped=()):
    return {
        "optimization_id": applied[0] if applied else None,
        "applied_ids": list(applied),
        "skipped": [{"entry_id": s, "reason": "test"} for s in skipped],
    }


def _bench_payload(speedup_pct: float = 35.0, correctness_pass: bool = True) -> dict:
    # Mirrors the real benchmark_comparison.json shape: `correctness.passed`
    # is the bool we read, and the top-level `verdict` is a string label.
    return {
        "speedup_pct": speedup_pct,
        "correctness": {
            "passed": correctness_pass,
            "loss_match": correctness_pass,
            "max_loss_diff": 0.01 if correctness_pass else 0.5,
            "rtol": 0.05,
            "atol": 0.01,
            "notes": "",
        },
        "verdict": "PASS" if correctness_pass else "FAIL (correctness; speedup measured but loss diverged)",
    }


def _enabled_recorder() -> MagicMock:
    """A recorder mock that reports enabled=True and captures calls."""
    rec = MagicMock()
    rec.enabled = True
    return rec


# ----------------------------- happy path ---------------------------------


def test_emit_run_with_full_artifacts(output_dir):
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    _write(output_dir / "profile" / "profile_record.json", _profile_payload())
    _write(output_dir / "edit" / "change_manifest.json",
           _manifest_payload(applied=("compile_default", "amp_bf16"),
                             skipped=("grad_accum",)))
    _write(output_dir / "benchmark" / "benchmark_comparison.json", _bench_payload())

    rec = _enabled_recorder()
    sent = emit_run(output_dir, rec, hardware_name="1x_a100")
    assert sent is True

    # Fingerprint was set on the recorder
    rec.begin_run.assert_called_once()
    fp = rec.begin_run.call_args.args[0]
    assert fp.arch_class == "transformer-decoder"
    assert fp.optimizer_class == "adam_family"
    assert fp.precision == "mixed_bf16"

    # Three outcome rows: primary (with speedup), secondary applied, skipped
    optimization_ids = [call.kwargs["optimization_id"]
                        for call in rec.record_optimization.call_args_list]
    assert optimization_ids == ["compile_default", "amp_bf16", "grad_accum"]

    # Primary carries the bench results
    primary_call = rec.record_optimization.call_args_list[0]
    assert primary_call.kwargs["applied"] is True
    assert primary_call.kwargs["speedup_factor"] == pytest.approx(1.35)
    assert primary_call.kwargs["loss_ok"] is True
    assert primary_call.kwargs["catalog_version"] == CATALOG_VERSION

    # Secondary applied — null outcomes (we can't attribute speedup individually)
    secondary = rec.record_optimization.call_args_list[1]
    assert secondary.kwargs["applied"] is True
    assert secondary.kwargs["speedup_factor"] is None

    # Skipped — applied=False
    skipped = rec.record_optimization.call_args_list[2]
    assert skipped.kwargs["applied"] is False


def test_loss_ok_reads_correctness_passed_not_verdict():
    """Regression: _loss_ok_from_bench used to read `correctness.verdict`,
    which never existed on the correctness sub-dict. The real key is
    `correctness.passed` (bool). The bug emitted loss_ok=None for every
    row, which made the optimization_priors materialized view's
    success_rate column NULL across the board — silently breaking the
    suggester's priors-based failure-avoidance filter."""
    from profine.telemetry.emit import _loss_ok_from_bench

    # Real shape: correctness.passed is the bool we read.
    assert _loss_ok_from_bench({"correctness": {"passed": True}}) is True
    assert _loss_ok_from_bench({"correctness": {"passed": False}}) is False

    # Old buggy shape: correctness.verdict — must NOT be silently treated
    # as truthy. Returning None is correct; emit.py will then record
    # loss_ok=None and the priors view will filter that row out.
    assert _loss_ok_from_bench({"correctness": {"verdict": "pass"}}) is None
    assert _loss_ok_from_bench({"correctness": {"verdict": "fail"}}) is None

    # Defensive cases.
    assert _loss_ok_from_bench({}) is None
    assert _loss_ok_from_bench({"correctness": None}) is None
    assert _loss_ok_from_bench({"correctness": "not a dict"}) is None


def test_emit_run_records_correctness_fail_as_loss_ok_false(output_dir):
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    _write(output_dir / "profile" / "profile_record.json", _profile_payload())
    _write(output_dir / "edit" / "change_manifest.json", _manifest_payload())
    _write(output_dir / "benchmark" / "benchmark_comparison.json",
           _bench_payload(speedup_pct=10.0, correctness_pass=False))

    rec = _enabled_recorder()
    emit_run(output_dir, rec, hardware_name="1x_a100")
    primary = rec.record_optimization.call_args_list[0]
    assert primary.kwargs["loss_ok"] is False


def test_emit_run_handles_negative_speedup(output_dir):
    """A regression (negative speedup_pct) maps to speedup_factor < 1.0."""
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    _write(output_dir / "profile" / "profile_record.json", _profile_payload())
    _write(output_dir / "edit" / "change_manifest.json", _manifest_payload())
    _write(output_dir / "benchmark" / "benchmark_comparison.json",
           _bench_payload(speedup_pct=-20.0))

    rec = _enabled_recorder()
    emit_run(output_dir, rec, hardware_name="1x_a100")
    primary = rec.record_optimization.call_args_list[0]
    assert primary.kwargs["speedup_factor"] == pytest.approx(0.80)


# ----------------------------- missing files ------------------------------


def test_emit_run_without_arch_record_skips(output_dir):
    """No architecture file → can't build a fingerprint → return False."""
    rec = _enabled_recorder()
    sent = emit_run(output_dir, rec, hardware_name="1x_a100")
    assert sent is False
    rec.begin_run.assert_not_called()
    rec.record_optimization.assert_not_called()


def test_emit_run_uses_hardware_fallback(output_dir):
    """No profile record but hardware_name fallback supplied → still works."""
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    rec = _enabled_recorder()
    sent = emit_run(output_dir, rec, hardware_name="1x_a100")
    assert sent is True
    rec.begin_run.assert_called_once()


def test_emit_run_without_any_hardware_signal_skips(output_dir):
    """Missing profile record AND no hardware fallback → skip."""
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    rec = _enabled_recorder()
    sent = emit_run(output_dir, rec, hardware_name=None)
    assert sent is False


def test_emit_run_with_no_edit_or_benchmark_still_emits_fingerprint(output_dir):
    """Fingerprint without any outcome rows is recorded but the
    backend will drop it (no outcomes to insert)."""
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    _write(output_dir / "profile" / "profile_record.json", _profile_payload())
    rec = _enabled_recorder()
    sent = emit_run(output_dir, rec, hardware_name="1x_a100")
    assert sent is True
    rec.begin_run.assert_called_once()
    rec.record_optimization.assert_not_called()


def test_emit_run_skips_when_recorder_disabled(output_dir):
    """Disabled recorder is a hard short-circuit — no file IO at all."""
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    _write(output_dir / "profile" / "profile_record.json", _profile_payload())
    rec = MagicMock()
    rec.enabled = False
    sent = emit_run(output_dir, rec, hardware_name="1x_a100")
    assert sent is False
    rec.begin_run.assert_not_called()


# ----------------------------- malformed inputs ---------------------------


def test_emit_run_tolerates_corrupt_architecture_json(output_dir):
    (output_dir / "read" / "architecture_record.json").write_text("not json{")
    rec = _enabled_recorder()
    sent = emit_run(output_dir, rec, hardware_name="1x_a100")
    assert sent is False  # gracefully skips, no exception


def test_emit_run_tolerates_corrupt_manifest(output_dir):
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    _write(output_dir / "profile" / "profile_record.json", _profile_payload())
    (output_dir / "edit" / "change_manifest.json").write_text("garbage")
    rec = _enabled_recorder()
    sent = emit_run(output_dir, rec, hardware_name="1x_a100")
    # Fingerprint should still go out; outcomes are skipped.
    assert sent is True
    rec.begin_run.assert_called_once()
    rec.record_optimization.assert_not_called()


def test_emit_run_unknown_hardware_name_falls_back_to_stub(output_dir):
    """A hardware name we don't have in the YAML still produces a stub
    HardwareConfig so we don't lose the fingerprint."""
    _write(output_dir / "read" / "architecture_record.json", _arch_payload())
    rec = _enabled_recorder()
    sent = emit_run(output_dir, rec, hardware_name="custom_unknown_gpu")
    assert sent is True
    fp = rec.begin_run.call_args.args[0]
    # hardware_class is normalised from the name we passed
    assert fp.hardware_class == "custom_unknown_gpu"
