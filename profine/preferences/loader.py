"""User Preferences loader (plan 6).

Loads a markdown file and exposes it as a raw string for LLM tools,
plus optional structured extraction of common fields.

The plan says: "Do not assume or enforce a structure." So we pass the
raw markdown to every LLM call, but also try to extract known fields
for deterministic filtering (do_not_touch, tolerance, etc).

Usage:
    from profine.preferences import load_preferences

    prefs = load_preferences("prefs.md")
    prefs.raw           # full markdown string for LLM calls
    prefs.do_not_touch  # ["optimizer_choice", "learning_rate"]
    prefs.rtol          # 1e-3
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class UserPreferences:
    """Parsed user preferences."""
    raw: str = ""                               # full markdown, passed to every LLM

    # Extracted fields (best-effort, may be empty)
    hardware: str = ""                          # e.g. "H100 x 8"
    goal: str = ""                              # e.g. "reduce step time"
    risk_tolerance: str = "medium"              # conservative | medium | experimental
    rtol: float = 1e-2
    atol: float = 1e-4
    do_not_touch: list[str] = field(default_factory=list)
    must_stay_in: str = ""                      # e.g. "pure pytorch"
    allow_microbatch_change: bool = False
    max_iterations: int = 8
    max_wall_clock_minutes: int = 120


def load_preferences(path: str | Path | None = None, raw: str | None = None) -> UserPreferences:
    """Load user preferences from a file or raw string.

    Args:
        path: Path to a markdown preferences file.
        raw: Raw markdown string (alternative to path).

    Returns:
        UserPreferences with raw text + extracted fields.
    """
    if path:
        text = Path(path).read_text(encoding="utf-8")
    elif raw:
        text = raw
    else:
        return UserPreferences()

    prefs = UserPreferences(raw=text)
    _extract_fields(prefs, text)
    return prefs


def _extract_fields(prefs: UserPreferences, text: str) -> None:
    """Best-effort extraction of known fields from markdown."""
    lines = text.lower().splitlines()

    for line in lines:
        stripped = line.strip().lstrip("- ").strip()

        # Hardware
        if stripped.startswith("gpu:"):
            prefs.hardware = _value_after(stripped, "gpu:")

        # Goal
        if stripped.startswith("primary:"):
            prefs.goal = _value_after(stripped, "primary:")

        # Risk tolerance
        if stripped.startswith("level:"):
            val = _value_after(stripped, "level:")
            if val in ("conservative", "medium", "experimental"):
                prefs.risk_tolerance = val

        # Numerical tolerance
        if "rtol" in stripped:
            match = re.search(r"rtol\s*[=:]\s*([\d.eE\-+]+)", stripped)
            if match:
                prefs.rtol = float(match.group(1))
        if "atol" in stripped:
            match = re.search(r"atol\s*[=:]\s*([\d.eE\-+]+)", stripped)
            if match:
                prefs.atol = float(match.group(1))

        # Do not touch
        if stripped.startswith("do_not_touch:"):
            val = _value_after(stripped, "do_not_touch:")
            prefs.do_not_touch = _parse_list(val)

        # Must stay in
        if stripped.startswith("must_stay_in:"):
            prefs.must_stay_in = _value_after(stripped, "must_stay_in:")

        # Microbatch
        if "allow_microbatch" in stripped:
            prefs.allow_microbatch_change = "true" in stripped or "yes" in stripped

        # Iterations
        if stripped.startswith("max_iterations:"):
            try:
                prefs.max_iterations = int(_value_after(stripped, "max_iterations:"))
            except ValueError:
                pass

        # Wall clock
        if stripped.startswith("max_wall_clock"):
            match = re.search(r"(\d+)", stripped)
            if match:
                prefs.max_wall_clock_minutes = int(match.group(1))


def _value_after(line: str, prefix: str) -> str:
    return line[len(prefix):].strip()


def _parse_list(val: str) -> list[str]:
    """Parse [a, b, c] or a, b, c into a list."""
    val = val.strip("[] ")
    return [item.strip().strip("'\"") for item in val.split(",") if item.strip()]
