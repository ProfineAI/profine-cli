"""Load and cache pattern data from YAML config files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent


# kernel_patterns.yaml

@lru_cache(maxsize=1)
def load_kernel_patterns() -> dict[str, Any]:
    return yaml.safe_load((_CONFIG_DIR / "kernel_patterns.yaml").read_text(encoding="utf-8"))


def get_exclude_patterns() -> list[str]:
    return load_kernel_patterns().get("exclude_patterns", [])


def get_categories() -> dict[str, list[str]]:
    return load_kernel_patterns().get("categories", {})


def get_precision_map() -> dict[str, list[str]]:
    return load_kernel_patterns().get("precision_from_kernel", {})


def get_attention_impl_map() -> dict[str, list[str]]:
    return load_kernel_patterns().get("attention_impl", {})


# patterns.yaml

@lru_cache(maxsize=1)
def load_patterns() -> dict[str, Any]:
    return yaml.safe_load((_CONFIG_DIR / "patterns.yaml").read_text(encoding="utf-8"))


def get_model_loaders() -> set[str]:
    return set(load_patterns().get("model_loaders", []))


def get_optimizer_names() -> set[str]:
    return set(load_patterns().get("optimizers", []))


def get_scheduler_names() -> set[str]:
    return set(load_patterns().get("schedulers", []))


def get_loss_names() -> set[str]:
    return set(load_patterns().get("loss_functions", []))


def get_distributed_patterns() -> set[str]:
    return set(load_patterns().get("distributed", []))


def get_dataloader_patterns() -> set[str]:
    return set(load_patterns().get("dataloaders", []))


def get_import_to_pip() -> dict[str, str]:
    return load_patterns().get("import_to_pip", {})


def get_transient_error_patterns() -> tuple[str, ...]:
    return tuple(load_patterns().get("transient_error_patterns", []))


def get_known_pypi_toplevel() -> set[str]:
    return {name.lower() for name in load_patterns().get("known_pypi_toplevel", [])}


# hardware.yaml

@lru_cache(maxsize=1)
def load_hardware() -> dict[str, Any]:
    return yaml.safe_load((_CONFIG_DIR / "hardware.yaml").read_text(encoding="utf-8"))


def get_hardware_presets() -> dict[str, dict[str, Any]]:
    return load_hardware().get("presets", {})


def get_hardware_aliases() -> dict[str, str]:
    return load_hardware().get("aliases", {})


# catalog.yaml

@lru_cache(maxsize=1)
def _load_catalog_raw() -> dict[str, Any]:
    return yaml.safe_load((_CONFIG_DIR / "catalog.yaml").read_text(encoding="utf-8"))


def load_catalog() -> list[dict[str, Any]]:
    return _load_catalog_raw().get("entries", [])


def get_category_tolerances() -> dict[str, tuple[float, float]]:
    """Map of category-fragment -> (rtol, atol) overrides for benchmark
    correctness checks. See `category_tolerances` in catalog.yaml."""
    out: dict[str, tuple[float, float]] = {}
    for key, val in (_load_catalog_raw().get("category_tolerances") or {}).items():
        out[key] = (float(val["rtol"]), float(val["atol"]))
    return out
