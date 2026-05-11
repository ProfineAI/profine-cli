"""Hardware and infrastructure configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HardwareConfig:
    """GPU hardware specifications."""
    name: str
    label: str
    modal_gpu: str
    gpu_count: int
    gpu_kind: str
    vram_gb: float
    interconnect: str = "pcie"
    cost_per_hour: float = 0.0
    bf16_supported: bool = False
    flash_attention_supported: bool = False
    fp8_supported: bool = False
    peak_tflops_fp32: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp8: float = 0.0
    peak_memory_bandwidth_gb_s: float = 0.0
    l2_cache_mb: float = 0.0
    sm_count: int = 0
    tensor_core_gen: str = ""
    compute_capability: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_vram_gb(self) -> float:
        return self.vram_gb * self.gpu_count

    @property
    def arithmetic_intensity_threshold(self) -> float | None:
        """Ops/byte: below this the workload is memory-bandwidth-bound."""
        if self.peak_tflops_fp16 > 0 and self.peak_memory_bandwidth_gb_s > 0:
            return (self.peak_tflops_fp16 * 1e12) / (self.peak_memory_bandwidth_gb_s * 1e9)
        return None


@dataclass(slots=True)
class ModalRuntimeConfig:
    """Modal execution parameters."""
    timeout_seconds: int = 900
    cpu: float = 8.0
    memory_mb: int = 40000
    ephemeral_disk_mb: int = 524288
    python_version: str = "3.13"
    enable_memory_snapshot: bool = True
    enable_warmstart: bool = False
    warmstart_scaledown_window_seconds: int = 300
    hf_cache_volume_name: str = "profine-hf-cache"
    pip_cache_volume_name: str = "profine-pip-cache"
    base_packages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ThresholdConfig:
    """Bottleneck detection thresholds."""
    dataloader_stall_pct: float = 10.0
    communication_overhead_pct: float = 10.0
    memory_headroom_pct: float = 10.0
    low_gpu_utilization_pct: float = 50.0


def get_hardware(name: str) -> HardwareConfig:
    """Look up a hardware preset by name from config/hardware.yaml."""
    from profine.config.yaml_loader import get_hardware_aliases, get_hardware_presets

    aliases = get_hardware_aliases()
    name = aliases.get(name, name)

    presets = get_hardware_presets()
    if name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown hardware '{name}'. Available: {available}")

    raw = presets[name]
    return HardwareConfig(name=name, **{
        k: v for k, v in raw.items()
        if k in HardwareConfig.__dataclass_fields__
    })
