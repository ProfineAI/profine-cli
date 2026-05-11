"""Optimization catalog — loads entries from config/catalog.yaml."""

from __future__ import annotations

from profine.catalog.schema import (
    ApplicabilityCondition,
    CatalogEntry,
    EvidenceEntry,
    SpeedupEstimate,
)
from profine.config.yaml_loader import load_catalog


def _build_entry(raw: dict) -> CatalogEntry:
    """Convert a raw YAML dict into a CatalogEntry dataclass."""
    required = [
        ApplicabilityCondition(field=r["field"], operator=r["operator"], value=r.get("value"))
        for r in raw.get("required", [])
    ]
    forbidden = [
        ApplicabilityCondition(field=r["field"], operator=r["operator"], value=r.get("value"))
        for r in raw.get("forbidden", [])
    ]

    speedup = None
    if raw.get("expected_speedup"):
        s = raw["expected_speedup"]
        speedup = SpeedupEstimate(
            kernel_low=s.get("kernel_low", 0.0),
            kernel_high=s.get("kernel_high", 0.0),
            end_to_end_low_pct=s.get("end_to_end_low_pct", 0.0),
            end_to_end_high_pct=s.get("end_to_end_high_pct", 0.0),
            depends_on=s.get("depends_on", ""),
        )

    evidence = [
        EvidenceEntry(kind=e["kind"], ref=e["ref"], url=e.get("url", ""))
        for e in raw.get("evidence", [])
    ]

    return CatalogEntry(
        id=raw["id"],
        category=raw["category"],
        name=raw["name"],
        description=raw.get("description", "").strip(),
        required=required,
        forbidden=forbidden,
        expected_speedup=speedup,
        risks=raw.get("risks", []),
        evidence=evidence,
        code_pattern=raw.get("code_pattern", "").strip(),
        addresses_bottlenecks=raw.get("addresses_bottlenecks", []),
    )


def get_catalog() -> list[CatalogEntry]:
    """Return the full optimization catalog from YAML."""
    return [_build_entry(raw) for raw in load_catalog()]


def get_entry(entry_id: str) -> CatalogEntry | None:
    """Look up a catalog entry by ID."""
    for entry in get_catalog():
        if entry.id == entry_id:
            return entry
    return None
