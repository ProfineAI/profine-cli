"""Run record schema — one entry per optimization pipeline run.

Captures what was tried, what happened, and the outcome so the system
learns across runs rather than re-deriving everything from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class OptimizationAttempt:
    """A single optimization attempted during a run."""
    optimization_id: str = ""
    optimization_name: str = ""
    applied: bool = False
    speedup_pct: float = 0.0
    correctness_passed: bool = True
    failure_reason: str = ""
    notes: str = ""


@dataclass(slots=True)
class RunRecord:
    """Complete record of a single pipeline run."""
    run_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    script_path: str = ""
    hardware: str = ""

    # Inputs
    architecture_summary: dict[str, Any] = field(default_factory=dict)
    profile_summary: dict[str, Any] = field(default_factory=dict)
    bottleneck_summary: dict[str, Any] = field(default_factory=dict)

    # What was tried
    attempts: list[OptimizationAttempt] = field(default_factory=list)

    # Outcome
    total_speedup_pct: float = 0.0
    final_correctness_passed: bool = True

    # User prefs used
    user_preferences_hash: str = ""

    # Free-form
    notes: list[str] = field(default_factory=list)
