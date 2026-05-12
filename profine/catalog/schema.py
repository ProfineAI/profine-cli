"""Optimization catalog entry schema (plan 5.1).

Each catalog entry describes a known optimization with applicability
conditions, expected speedup, risks, and evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ApplicabilityCondition:
    """A single required or forbidden condition."""
    field: str          # architecture record field path, e.g. "attention_type.value"
    operator: str       # "in", "not_in", "eq", "gte", "lte", "exists", "contains"
    value: Any = None   # comparison value


@dataclass(slots=True)
class SpeedupEstimate:
    """Expected speedup range."""
    kernel_low: float = 0.0     # e.g. 1.5 = 1.5x kernel speedup
    kernel_high: float = 0.0
    end_to_end_low_pct: float = 0.0   # e.g. 10 = 10% end-to-end improvement
    end_to_end_high_pct: float = 0.0
    depends_on: str = ""        # e.g. "attention's share of step time"


@dataclass(slots=True)
class EvidenceEntry:
    """A citation backing the optimization."""
    kind: str           # "paper", "blog", "docs", "run", "benchmark"
    ref: str            # human-readable reference
    url: str = ""
    outcome: str = ""   # e.g. "1.6x attn speedup on llama-7b/A100"


@dataclass(slots=True)
class CatalogEntry:
    """A single optimization in the catalog."""
    id: str
    category: str       # attention, compiler, kernel_fusion, data_pipeline, distributed, memory, optimizer, precision, memory_layout
    name: str
    description: str

    # Applicability
    required: list[ApplicabilityCondition] = field(default_factory=list)
    forbidden: list[ApplicabilityCondition] = field(default_factory=list)

    # Impact
    expected_speedup: SpeedupEstimate | None = None
    risks: list[str] = field(default_factory=list)

    # Evidence
    evidence: list[EvidenceEntry] = field(default_factory=list)

    # Implementation hints
    code_pattern: str = ""      # brief description of the code change
    reference_impl: str = ""    # snippet or reference

    # LLM-appended observations from past runs
    unstructured_notes: list[str] = field(default_factory=list)

    # Addressable bottleneck categories
    addresses_bottlenecks: list[str] = field(default_factory=list)

    # Mutual exclusivity — entries sharing the same non-zero group
    # cannot both be applied in a single --top N stacked edit.
    exclusive_group: int = 0
