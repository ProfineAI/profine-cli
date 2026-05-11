"""Tests that every yaml_loader function returns the expected shape.

These guard against typos in YAML keys and ensure the lru_cache behaves.
"""

from __future__ import annotations

from profine.config.yaml_loader import (
    get_attention_impl_map,
    get_categories,
    get_category_tolerances,
    get_dataloader_patterns,
    get_distributed_patterns,
    get_exclude_patterns,
    get_hardware_aliases,
    get_hardware_presets,
    get_import_to_pip,
    get_known_pypi_toplevel,
    get_loss_names,
    get_model_loaders,
    get_optimizer_names,
    get_precision_map,
    get_scheduler_names,
    get_transient_error_patterns,
    load_catalog,
    load_hardware,
    load_kernel_patterns,
    load_patterns,
)


def test_kernel_patterns_yaml_loads():
    data = load_kernel_patterns()
    assert isinstance(data, dict)
    assert "categories" in data


def test_patterns_yaml_loads():
    data = load_patterns()
    assert isinstance(data, dict)
    assert "model_loaders" in data


def test_hardware_yaml_loads():
    data = load_hardware()
    assert isinstance(data, dict)
    assert "presets" in data


def test_catalog_yaml_loads():
    entries = load_catalog()
    assert isinstance(entries, list)
    assert len(entries) > 0
    assert all("id" in e for e in entries)


def test_exclude_patterns_is_list():
    assert isinstance(get_exclude_patterns(), list)


def test_categories_keys_are_strings_values_are_lists():
    cats = get_categories()
    assert all(isinstance(k, str) for k in cats)
    assert all(isinstance(v, list) for v in cats.values())


def test_precision_map_returns_dict():
    assert isinstance(get_precision_map(), dict)


def test_attention_impl_map_returns_dict():
    assert isinstance(get_attention_impl_map(), dict)


def test_pattern_sets_return_sets():
    for fn in (
        get_model_loaders,
        get_optimizer_names,
        get_scheduler_names,
        get_loss_names,
        get_distributed_patterns,
        get_dataloader_patterns,
    ):
        result = fn()
        assert isinstance(result, set), f"{fn.__name__} must return a set"
        assert len(result) > 0, f"{fn.__name__} returned empty set"


def test_optimizer_names_includes_common_ones():
    optimizers = get_optimizer_names()
    for expected in ("Adam", "AdamW", "SGD"):
        assert expected in optimizers


def test_loss_names_includes_common_ones():
    losses = get_loss_names()
    for expected in ("CrossEntropyLoss", "MSELoss"):
        assert expected in losses


def test_import_to_pip_mapping():
    mapping = get_import_to_pip()
    assert mapping["cv2"] == "opencv-python"
    assert mapping["yaml"] == "PyYAML"


def test_transient_error_patterns_returns_tuple():
    patterns = get_transient_error_patterns()
    assert isinstance(patterns, tuple)
    assert any("connection" in p for p in patterns)


def test_hardware_presets_includes_a100():
    presets = get_hardware_presets()
    assert "1x_a100" in presets
    a100 = presets["1x_a100"]
    assert a100["modal_gpu"] == "A100-80GB"
    assert a100["bf16_supported"] is True


def test_hardware_aliases_resolve():
    aliases = get_hardware_aliases()
    assert aliases["1x_a100_80gb"] == "1x_a100"


def test_known_pypi_toplevel_is_lowercased_set():
    names = get_known_pypi_toplevel()
    assert isinstance(names, set)
    assert all(n == n.lower() for n in names)


def test_category_tolerances_have_two_floats():
    for key, val in get_category_tolerances().items():
        assert isinstance(val, tuple) and len(val) == 2
        assert all(isinstance(x, float) for x in val)


def test_yaml_loaders_are_cached():
    # Same call returns same dict identity (lru_cache hit) for the
    # underlying load_* functions.
    assert load_patterns() is load_patterns()
    assert load_hardware() is load_hardware()
    assert load_kernel_patterns() is load_kernel_patterns()
