"""Application configuration.

Loads settings from defaults, optional TOML file, and environment overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profine.schema.hardware import (
    HardwareConfig,
    ModalRuntimeConfig,
    ThresholdConfig,
    get_hardware,
)


@dataclass(slots=True)
class Credentials:
    """API keys and tokens."""
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    modal_token_id: str | None = None
    modal_token_secret: str | None = None
    hf_token: str | None = None

    @staticmethod
    def from_env() -> Credentials:
        return Credentials(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            modal_token_id=os.environ.get("MODAL_TOKEN_ID"),
            modal_token_secret=os.environ.get("MODAL_TOKEN_SECRET"),
            hf_token=os.environ.get("HF_TOKEN"),
        )


@dataclass(slots=True)
class AppConfig:
    """Top-level application config."""
    default_steps: int = 60
    default_warmup_steps: int = 30
    default_hardware: str = "1x_a100"
    default_provider: str = "openai"
    default_modal_timeout: int = 900
    modal: ModalRuntimeConfig = field(default_factory=ModalRuntimeConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    hardware: dict[str, HardwareConfig] = field(default_factory=dict)


DEFAULTS = AppConfig()


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load config from an optional TOML file, with env overrides.

    If no config_path is given, uses defaults.
    """
    config = AppConfig()

    if config_path and config_path.exists():
        config = _load_toml(config_path, config)

    # Environment overrides
    if os.environ.get("PROFINE_STEPS"):
        config.default_steps = int(os.environ["PROFINE_STEPS"])
    if os.environ.get("PROFINE_WARMUP_STEPS"):
        config.default_warmup_steps = int(os.environ["PROFINE_WARMUP_STEPS"])
    if os.environ.get("PROFINE_HARDWARE"):
        config.default_hardware = os.environ["PROFINE_HARDWARE"]
    if os.environ.get("PROFINE_PROVIDER"):
        config.default_provider = os.environ["PROFINE_PROVIDER"]
    if os.environ.get("PROFINE_MODAL_TIMEOUT"):
        config.default_modal_timeout = int(os.environ["PROFINE_MODAL_TIMEOUT"])

    return config


def _load_toml(path: Path, config: AppConfig) -> AppConfig:
    """Merge TOML config into the defaults."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return config

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    # Top-level settings
    config.default_steps = data.get("steps", config.default_steps)
    config.default_warmup_steps = data.get("warmup_steps", config.default_warmup_steps)
    config.default_hardware = data.get("hardware", config.default_hardware)
    config.default_provider = data.get("provider", config.default_provider)

    # Modal settings
    modal_data = data.get("modal", {})
    if modal_data:
        for key, val in modal_data.items():
            if hasattr(config.modal, key):
                setattr(config.modal, key, val)

    # Thresholds
    threshold_data = data.get("thresholds", {})
    if threshold_data:
        for key, val in threshold_data.items():
            if hasattr(config.thresholds, key):
                setattr(config.thresholds, key, val)

    return config
