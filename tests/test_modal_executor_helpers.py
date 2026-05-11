"""Tests for executor / orchestrator pure-function helpers.

`_remote_execute` runs inside Modal and uses signal/filesystem, so we
exercise the helpers around it instead.
"""

from __future__ import annotations

import json

from profine.modal.executor import (
    _needs_flash_attention,
    _parse_output,
)
from profine.profiler.hooks import RESULTS_SENTINEL
from profine.profiler.orchestrator import (
    _drop_bad_deps,
    _extract_bad_pip_names,
    _extract_missing_dep_names,
    _looks_like_cpu_run,
    _same_error_signature,
)


def test_needs_flash_attention_dash_form():
    assert _needs_flash_attention(["flash-attn==2.7.4", "torch"]) is True


def test_needs_flash_attention_underscore_form():
    assert _needs_flash_attention(["flash_attn", "torch"]) is True


def test_needs_flash_attention_absent():
    assert _needs_flash_attention(["torch", "transformers"]) is False


def test_needs_flash_attention_empty():
    assert _needs_flash_attention([]) is False


def test_parse_output_finds_sentinel_payload():
    payload = {"step_times_ms": [1.0, 2.0]}
    raw = f"setup ok\nepoch 0\n{RESULTS_SENTINEL}{json.dumps(payload)}\n"
    result = _parse_output(raw, runtime_seconds=12.5)
    assert result.success is True
    assert result.payload["step_times_ms"] == [1.0, 2.0]
    assert result.payload["runtime_seconds"] == 12.5


def test_parse_output_uses_last_sentinel():
    # First sentinel is partial; second is the real payload
    partial = json.dumps({"status": "partial"})
    final = json.dumps({"status": "ok", "step_times_ms": [1.0]})
    raw = f"{RESULTS_SENTINEL}{partial}\nsome later log\n{RESULTS_SENTINEL}{final}\n"
    result = _parse_output(raw, runtime_seconds=1.0)
    assert result.payload["status"] == "ok"


def test_parse_output_no_sentinel_is_failure():
    result = _parse_output("just normal stdout, no sentinel\n", runtime_seconds=1.0)
    assert result.success is False
    assert "No results sentinel" in result.error


def test_extract_bad_pip_names_basic():
    err = "ERROR: Could not find a version that satisfies the requirement totally-fake-pkg"
    assert "totally-fake-pkg" in _extract_bad_pip_names(err)


def test_extract_bad_pip_names_alternate_phrasing():
    err = "No matching distribution found for nonexistent_pkg"
    assert "nonexistent_pkg" in _extract_bad_pip_names(err)


def test_extract_bad_pip_names_empty():
    assert _extract_bad_pip_names("") == set()


def test_drop_bad_deps_filters_named_packages():
    deps = ["torch>=2.0", "bad-pkg==1.0", "transformers", "another-bad[extra]"]
    bad = {"bad-pkg", "another-bad"}
    kept, dropped = _drop_bad_deps(deps, bad)
    assert "torch>=2.0" in kept
    assert "transformers" in kept
    assert "bad-pkg==1.0" in dropped
    assert "another-bad[extra]" in dropped


def test_drop_bad_deps_no_match_keeps_all():
    deps = ["torch", "numpy"]
    kept, dropped = _drop_bad_deps(deps, {"unrelated"})
    assert kept == deps
    assert dropped == []


def test_extract_missing_dep_names_uses_pip_mapping():
    # `yaml` is in known_pypi_toplevel and import_to_pip maps it to PyYAML
    err = "ModuleNotFoundError: No module named 'yaml'"
    assert _extract_missing_dep_names(err) == {"PyYAML"}


def test_same_error_signature_collapses_repeats():
    a = "ModuleNotFoundError: No module named 'pkgX'"
    b = "ModuleNotFoundError: No module named 'pkgX'\nat line 42"
    assert _same_error_signature(a, b) is True


def test_same_error_signature_distinguishes_different_modules():
    a = "ModuleNotFoundError: No module named 'pkgA'"
    b = "ModuleNotFoundError: No module named 'pkgB'"
    assert _same_error_signature(a, b) is False


def test_same_error_signature_empty_inputs():
    assert _same_error_signature("", "") is False
    assert _same_error_signature("anything", "") is False


def test_looks_like_cpu_run_low_util_low_mem():
    payload = {
        "gpu_utilization_samples": [0.0, 1.0, 2.0],
        "memory_peak_bytes": 1024,  # 1 KB — clearly never touched the GPU
    }
    assert _looks_like_cpu_run(payload, "1x_a100") is True


def test_looks_like_cpu_run_normal_run_returns_false():
    payload = {
        "gpu_utilization_samples": [80.0] * 10,
        "memory_peak_bytes": 8 * 1024**3,
    }
    assert _looks_like_cpu_run(payload, "1x_a100") is False
