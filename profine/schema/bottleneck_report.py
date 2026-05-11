"""Bottleneck report schema — output of the Profile Interpreter (plan 4.3).

The interpreter diagnoses where time is going and quantifies headroom.
It does NOT propose fixes — that's the suggester's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BottleneckEntry:
    """A single diagnosed bottleneck with quantified headroom."""
    category: str           # e.g. "attention", "data_pipeline", "precision", "memory_bandwidth"
    location: str           # e.g. "flash_fwd_kernel at model.py:142" or "DataLoader stall"
    time_share_pct: float   # % of total step time this bottleneck accounts for
    est_headroom_pct: float # estimated % speedup if fully addressed
    confidence: str = "observed"  # observed | inferred | guessed
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class BottleneckReport:
    """Ranked list of bottlenecks produced by the Profile Interpreter."""
    # Ranked bottlenecks (highest impact first)
    bottlenecks: list[BottleneckEntry] = field(default_factory=list)

    # High-level classification
    compute_bound: bool = False
    memory_bandwidth_bound: bool = False
    memory_capacity_bound: bool = False
    data_pipeline_bound: bool = False
    communication_bound: bool = False

    # Narrative
    summary: str = ""         # 2-3 sentence executive summary
    time_breakdown_narrative: str = ""  # "where the time is going" paragraph

    # Unstructured LLM observations
    unstructured_notes: list[str] = field(default_factory=list)


BOTTLENECK_REPORT_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BottleneckReport",
    "type": "object",
    "properties": {
        "bottlenecks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "location": {"type": "string"},
                    "time_share_pct": {"type": "number"},
                    "est_headroom_pct": {"type": "number"},
                    "confidence": {"type": "string", "enum": ["observed", "inferred", "guessed"]},
                    "supporting_evidence": {"type": "array"},
                    "notes": {"type": "string"},
                },
                "required": ["category", "location", "time_share_pct", "est_headroom_pct", "confidence"],
            },
        },
        "compute_bound": {"type": "boolean"},
        "memory_bandwidth_bound": {"type": "boolean"},
        "memory_capacity_bound": {"type": "boolean"},
        "data_pipeline_bound": {"type": "boolean"},
        "communication_bound": {"type": "boolean"},
        "summary": {"type": "string"},
        "time_breakdown_narrative": {"type": "string"},
        "unstructured_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["bottlenecks", "summary"],
}
