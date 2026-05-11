"""Optimization candidate schema — output of the Suggester (plan 4.4).

Each candidate is a catalog entry that passed applicability checks,
enriched with LLM reasoning about priority and expected ROI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OptimizationCandidate:
    """A single optimization recommendation."""
    entry_id: str                   # catalog entry ID
    category: str                   # e.g. "precision", "attention", "compiler"
    name: str                       # human-readable name
    description: str                # why this optimization helps

    # Ranking
    rank: int = 0                   # 1 = highest priority
    priority: str = "medium"        # "critical" | "high" | "medium" | "low"

    # Impact estimate (from catalog + LLM adjustment)
    est_speedup_low_pct: float = 0.0
    est_speedup_high_pct: float = 0.0
    confidence: str = "inferred"    # "high" | "medium" | "low"

    # LLM reasoning
    rationale: str = ""             # why this was ranked here
    bottlenecks_addressed: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    # Implementation
    code_pattern: str = ""          # brief code change description
    estimated_effort: str = ""      # "trivial" | "small" | "medium" | "large"

    # Evidence
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SuggestionReport:
    """Full output of the Suggest Optimizations tool."""
    candidates: list[OptimizationCandidate] = field(default_factory=list)
    summary: str = ""               # executive summary
    total_est_speedup_low_pct: float = 0.0
    total_est_speedup_high_pct: float = 0.0
    warnings: list[str] = field(default_factory=list)
    unstructured_notes: list[str] = field(default_factory=list)


OPTIMIZATION_CANDIDATE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SuggestionReport",
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "string"},
                    "category": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "rank": {"type": "integer"},
                    "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "est_speedup_low_pct": {"type": "number"},
                    "est_speedup_high_pct": {"type": "number"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "rationale": {"type": "string"},
                    "bottlenecks_addressed": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "code_pattern": {"type": "string"},
                    "estimated_effort": {"type": "string"},
                    "evidence": {"type": "array"},
                },
                "required": ["entry_id", "category", "name", "rank", "priority", "rationale"],
            },
        },
        "summary": {"type": "string"},
        "total_est_speedup_low_pct": {"type": "number"},
        "total_est_speedup_high_pct": {"type": "number"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "unstructured_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["candidates", "summary"],
}
