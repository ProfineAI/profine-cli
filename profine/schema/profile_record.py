"""Profile record schema — canonical output of the Profiler tool (plan 4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProfilerEvent:
    """A single kernel/op event from torch.profiler."""
    name: str
    category: str = ""
    self_cpu_time_total_us: float = 0.0
    self_cuda_time_total_us: float = 0.0
    flops: float = 0.0
    bytes_moved: float = 0.0
    count: int = 1
    input_dtypes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class KernelSummary:
    """Aggregated kernel by name."""
    name: str
    category: str
    cuda_time_us: float
    pct_of_total: float
    count: int = 1
    flops: float = 0.0
    bytes_moved: float = 0.0


@dataclass(slots=True)
class KernelCategoryBreakdown:
    """Time distribution across kernel categories."""
    matmul_pct: float = 0.0
    attention_pct: float = 0.0
    elementwise_pct: float = 0.0
    normalization_pct: float = 0.0
    optimizer_pct: float = 0.0
    communication_pct: float = 0.0
    memory_pct: float = 0.0
    dataloader_pct: float = 0.0
    other_pct: float = 0.0


@dataclass(slots=True)
class PhaseBreakdown:
    """Time distribution across training phases."""
    forward_pct: float = 0.0
    backward_pct: float = 0.0
    optimizer_pct: float = 0.0
    dataloader_pct: float = 0.0
    other_pct: float = 0.0


@dataclass(slots=True)
class ProfileRecord:
    """Canonical output of the Profiler tool.

    Combines raw measurements + computed heuristics into a single object
    that downstream tools consume.
    """
    # Status
    status: str = "ok"  # ok | oom | crash | timeout
    error: str | None = None

    # Identity
    script_path: str = ""
    hardware_name: str = ""

    # Step configuration
    steps_requested: int = 0
    steps_completed: int = 0
    warmup_steps_requested: int = 0
    warmup_steps_effective: int = 0

    # Timing
    runtime_seconds: float = 0.0
    step_times_ms: list[float] = field(default_factory=list)
    warmup_step_times_ms: list[float] = field(default_factory=list)

    # Losses
    loss_values: list[float] = field(default_factory=list)

    # GPU utilization
    gpu_util_samples: list[float] = field(default_factory=list)
    gpu_util_mean: float = 0.0
    gpu_util_pattern: str = "unknown"

    # Memory
    memory_samples_bytes: list[int] = field(default_factory=list)
    memory_peak_bytes: int = 0

    # Profiler events (raw)
    profiler_events: list[ProfilerEvent] = field(default_factory=list)

    # Computed heuristics
    top_kernels: list[KernelSummary] = field(default_factory=list)
    kernel_breakdown: KernelCategoryBreakdown | None = None
    phase_breakdown: PhaseBreakdown | None = None
    dataloader_stall_pct: float = 0.0
    arithmetic_intensity: float | None = None
    communication_overhead_pct: float = 0.0
    communication_overlapped: bool = False
    attention_impl: str = "unknown"
    precision: str = "unknown"
    memory_headroom_pct: float = 0.0
    # Metadata (runtime-captured: batch_size, grad_accum, dataloader config, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_time_median_ms(self) -> float | None:
        if not self.step_times_ms:
            return None
        sorted_times = sorted(self.step_times_ms)
        n = len(sorted_times)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_times[mid - 1] + sorted_times[mid]) / 2
        return sorted_times[mid]

    @property
    def memory_peak_gb(self) -> float:
        return self.memory_peak_bytes / (1024 ** 3)

    @property
    def final_loss(self) -> float | None:
        return self.loss_values[-1] if self.loss_values else None


PROFILE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ProfileRecord",
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "oom", "crash", "timeout"]},
        "error": {"type": ["string", "null"]},
        "script_path": {"type": "string"},
        "hardware_name": {"type": "string"},
        "steps_requested": {"type": "integer"},
        "steps_completed": {"type": "integer"},
        "warmup_steps_requested": {"type": "integer"},
        "warmup_steps_effective": {"type": "integer"},
        "runtime_seconds": {"type": "number"},
        "step_times_ms": {"type": "array", "items": {"type": "number"}},
        "warmup_step_times_ms": {"type": "array", "items": {"type": "number"}},
        "loss_values": {"type": "array", "items": {"type": "number"}},
        "gpu_util_samples": {"type": "array", "items": {"type": "number"}},
        "gpu_util_mean": {"type": "number"},
        "gpu_util_pattern": {"type": "string"},
        "memory_samples_bytes": {"type": "array", "items": {"type": "integer"}},
        "memory_peak_bytes": {"type": "integer"},
        "dataloader_stall_pct": {"type": "number"},
        "arithmetic_intensity": {"type": ["number", "null"]},
        "communication_overhead_pct": {"type": "number"},
        "communication_overlapped": {"type": "boolean"},
        "attention_impl": {"type": "string"},
        "precision": {"type": "string"},
        "memory_headroom_pct": {"type": "number"},
    },
    "required": ["status", "script_path"],
}
