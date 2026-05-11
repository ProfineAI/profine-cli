"""Tests for centralised defaults and YAML-backed config helpers.

Covers the cleanup that moved hardcoded constants out of source:
  * AppConfig.default_modal_timeout (settings.py)
  * known_pypi_toplevel (patterns.yaml)
  * category_tolerances (catalog.yaml)
  * orchestrator/_classify_error + _extract_missing_dep_names YAML lookup
  * benchmarker/_resolve_tolerance YAML lookup
  * CLI argparse defaults flow from DEFAULTS
"""

from __future__ import annotations

import os

import pytest

from profine.config.settings import DEFAULTS, AppConfig, load_config
from profine.config.yaml_loader import (
    get_category_tolerances,
    get_known_pypi_toplevel,
)


def test_app_config_has_modal_timeout_default():
    cfg = AppConfig()
    assert cfg.default_modal_timeout == 900
    assert cfg.default_steps == 60
    assert cfg.default_warmup_steps == 30
    assert cfg.default_hardware == "1x_a100"


def test_defaults_singleton_matches_app_config():
    fresh = AppConfig()
    assert DEFAULTS.default_modal_timeout == fresh.default_modal_timeout
    assert DEFAULTS.default_steps == fresh.default_steps
    assert DEFAULTS.default_warmup_steps == fresh.default_warmup_steps


def test_load_config_env_override_for_modal_timeout(monkeypatch):
    monkeypatch.setenv("PROFINE_MODAL_TIMEOUT", "1800")
    cfg = load_config()
    assert cfg.default_modal_timeout == 1800


def test_load_config_env_override_invalid_modal_timeout_raises(monkeypatch):
    monkeypatch.setenv("PROFINE_MODAL_TIMEOUT", "not-an-int")
    with pytest.raises(ValueError):
        load_config()


def test_known_pypi_toplevel_yaml_loaded():
    names = get_known_pypi_toplevel()
    # spot-check a representative sample; full list lives in patterns.yaml
    for expected in ("numpy", "transformers", "flash_attn", "triton"):
        assert expected in names, f"missing {expected} in known_pypi_toplevel"
    # values must be lowercased so error-message lookups are case-insensitive
    assert all(n == n.lower() for n in names)


def test_known_pypi_toplevel_is_cached():
    # lru_cache on the underlying loader should give identity-stable results
    assert get_known_pypi_toplevel() == get_known_pypi_toplevel()


def test_category_tolerances_yaml_loaded():
    tols = get_category_tolerances()
    assert "precision" in tols
    assert "mixed_precision" in tols
    assert "quantization" in tols
    rtol, atol = tols["precision"]
    # bf16/fp16 widening must be looser than the algebraic-equivalence default
    assert rtol > 1e-2
    assert atol > 1e-4
    # quantization should be at least as loose as precision
    q_rtol, q_atol = tols["quantization"]
    assert q_rtol >= rtol
    assert q_atol >= atol


def test_classify_error_routes_known_dep_to_missing_dep():
    from profine.profiler.orchestrator import _classify_error
    err = "ModuleNotFoundError: No module named 'transformers'"
    assert _classify_error(err) == "missing_dep"


def test_classify_error_routes_unknown_module_to_script():
    from profine.profiler.orchestrator import _classify_error
    err = "ModuleNotFoundError: No module named 'definitely_not_real_pkg_xyz'"
    assert _classify_error(err) == "script"


def test_classify_error_routes_image_build_failure_to_deps():
    from profine.profiler.orchestrator import _classify_error
    assert _classify_error("Image build failed: bad pkg") == "deps"
    assert _classify_error("No matching distribution found for foo") == "deps"


def test_classify_error_routes_network_to_infra():
    from profine.profiler.orchestrator import _classify_error
    assert _classify_error("ConnectionReset by peer") == "infra"
    assert _classify_error("Network is unreachable") == "infra"


def test_extract_missing_dep_names_uses_import_to_pip_mapping():
    from profine.profiler.orchestrator import _extract_missing_dep_names
    # `yaml` is in known_pypi_toplevel and patterns.yaml maps it to PyYAML
    err = "ModuleNotFoundError: No module named 'yaml'"
    assert _extract_missing_dep_names(err) == {"PyYAML"}


def test_extract_missing_dep_names_skips_unknown():
    from profine.profiler.orchestrator import _extract_missing_dep_names
    err = "ModuleNotFoundError: No module named 'nope_unknown_pkg'"
    assert _extract_missing_dep_names(err) == set()


def test_resolve_tolerance_widens_for_precision_category():
    from profine.benchmarker.benchmarker import _resolve_tolerance
    DEFAULT_RTOL, DEFAULT_ATOL = 1e-2, 1e-4
    rtol, atol = _resolve_tolerance("mixed_precision", DEFAULT_RTOL, DEFAULT_ATOL)
    assert rtol > DEFAULT_RTOL
    assert atol > DEFAULT_ATOL


def test_resolve_tolerance_widens_for_quantization_category():
    from profine.benchmarker.benchmarker import _resolve_tolerance
    DEFAULT_RTOL, DEFAULT_ATOL = 1e-2, 1e-4
    rtol, atol = _resolve_tolerance("int8_quantization", DEFAULT_RTOL, DEFAULT_ATOL)
    # quantization gets the loosest table entry
    assert rtol >= 1e-1
    assert atol >= 5e-2


def test_resolve_tolerance_passes_through_for_unknown_category():
    from profine.benchmarker.benchmarker import _resolve_tolerance
    DEFAULT_RTOL, DEFAULT_ATOL = 1e-2, 1e-4
    rtol, atol = _resolve_tolerance("kernel_fusion", DEFAULT_RTOL, DEFAULT_ATOL)
    assert (rtol, atol) == (DEFAULT_RTOL, DEFAULT_ATOL)


def test_resolve_tolerance_respects_user_override():
    from profine.benchmarker.benchmarker import _resolve_tolerance
    # When the user passes non-default tolerances we must not overwrite them
    # even for a category that would otherwise be widened.
    rtol, atol = _resolve_tolerance("mixed_precision", 3e-3, 7e-5)
    assert (rtol, atol) == (3e-3, 7e-5)


def test_cli_argparse_defaults_flow_from_settings():
    from profine.cli.main import build_parser

    parser = build_parser()
    profile_args = parser.parse_args(["profile", "train.py"])
    assert profile_args.timeout == DEFAULTS.default_modal_timeout
    assert profile_args.steps == DEFAULTS.default_steps
    assert profile_args.warmup == DEFAULTS.default_warmup_steps
    assert profile_args.hardware == DEFAULTS.default_hardware

    bench_args = parser.parse_args(
        ["benchmark", "train.py", "--optimized", "edited.py"]
    )
    assert bench_args.timeout == DEFAULTS.default_modal_timeout
    assert bench_args.steps == DEFAULTS.default_steps
    assert bench_args.warmup == DEFAULTS.default_warmup_steps
