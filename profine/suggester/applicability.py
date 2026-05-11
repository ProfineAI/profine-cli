"""Deterministic applicability checker for catalog entries.

Evaluates each CatalogEntry's required/forbidden conditions against
the ArchitectureRecord and BottleneckReport to filter which optimizations
are relevant.
"""

from __future__ import annotations

from typing import Any

from profine.catalog.schema import ApplicabilityCondition, CatalogEntry
from profine.schema.bottleneck_report import BottleneckReport


class ApplicabilityChecker:
    """Check which catalog entries apply to a given architecture + bottleneck profile."""

    def __init__(
        self,
        architecture_record: dict[str, Any],
        bottleneck_report: BottleneckReport | None = None,
    ) -> None:
        self._arch = architecture_record
        self._bottlenecks = bottleneck_report

    def check(self, entry: CatalogEntry) -> tuple[bool, list[str]]:
        """Check if a catalog entry applies.

        Returns:
            (applicable, reasons) — True if all required conditions met and
            no forbidden conditions triggered. reasons lists any failures.
        """
        reasons: list[str] = []

        for req in entry.required:
            val = self._resolve_field(req.field)
            if val is _MISSING:
                # Missing field + not_in = condition passes (field isn't set to a forbidden value)
                # Missing field + other operators = can't evaluate, skip with warning
                if req.operator == "not_in":
                    continue
                reasons.append(f"Field '{req.field}' not found in architecture record")
                continue
            if not _evaluate(req, val):
                reasons.append(f"Required condition failed: {req.field} {req.operator} {req.value} (got {val!r})")

        for forb in entry.forbidden:
            val = self._resolve_field(forb.field)
            if val is _MISSING:
                continue  # field absent = forbidden condition not triggered
            if _evaluate(forb, val):
                reasons.append(f"Forbidden condition triggered: {forb.field} {forb.operator} {forb.value}")

        return len(reasons) == 0, reasons

    def check_bottleneck_relevance(self, entry: CatalogEntry) -> float:
        """ROI = match × catalog impact ceiling. Entries that address
        any diagnosed bottleneck get full match weight; the rest get a
        floor so high-impact universal optimizations stay visible."""
        if not entry.addresses_bottlenecks:
            return 0.0

        impact = 0.5
        if entry.expected_speedup and entry.expected_speedup.end_to_end_high_pct:
            impact = entry.expected_speedup.end_to_end_high_pct / 100.0

        if not self._bottlenecks:
            return impact * 0.5

        active_categories = {b.category for b in self._bottlenecks.bottlenecks}
        for flag in ("compute_bound", "memory_bandwidth_bound",
                     "memory_capacity_bound", "data_pipeline_bound",
                     "communication_bound"):
            if getattr(self._bottlenecks, flag, False):
                active_categories.add(flag)

        any_match = any(b in active_categories for b in entry.addresses_bottlenecks)
        relevance = 1.0 if any_match else 0.4
        return relevance * impact

    def filter_catalog(
        self, entries: list[CatalogEntry],
    ) -> list[tuple[CatalogEntry, float, list[str]]]:
        """Filter and score a full catalog.

        Returns list of (entry, relevance_score, rejection_reasons) for
        entries that passed applicability. Sorted by relevance descending.
        """
        results: list[tuple[CatalogEntry, float, list[str]]] = []
        for entry in entries:
            applicable, reasons = self.check(entry)
            if not applicable:
                continue
            relevance = self.check_bottleneck_relevance(entry)
            results.append((entry, relevance, reasons))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _resolve_field(self, field_path: str) -> Any:
        """Resolve a dotted field path against the architecture record.

        Supports paths like "optimizer.name.value", "precision.training_dtype.value".
        """
        parts = field_path.split(".")
        current: Any = self._arch
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    return _MISSING
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return _MISSING
        return current


class _MissingSentinel:
    """Sentinel for missing field values."""
    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MissingSentinel()


def _evaluate(condition: ApplicabilityCondition, actual: Any) -> bool:
    """Evaluate a single condition against an actual value."""
    op = condition.operator
    expected = condition.value

    if op == "eq":
        return actual == expected
    if op == "in":
        return actual in expected if isinstance(expected, list) else actual == expected
    if op == "not_in":
        return actual not in expected if isinstance(expected, list) else actual != expected
    if op == "gte":
        try:
            return actual >= expected
        except TypeError:
            return False  # incompatible types: treat as "condition not satisfied"
    if op == "lte":
        try:
            return actual <= expected
        except TypeError:
            return False
    if op == "exists":
        return actual is not _MISSING
    if op == "contains":
        return expected in actual if isinstance(actual, (str, list)) else False

    return False
