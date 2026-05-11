"""
Three layers of protection, all derived from the catalog YAML so adding
an entry automatically gets covered:

1. Field-path lint: every `field:` referenced by a rule must resolve to
   a known path in the architecture record schema. Catches typos that
   would silently produce dead rules.

2. Per-entry applicability: for each entry, build a record that satisfies
   all rules (assert applicable=True), then mutate to violate one rule
   at a time (assert applicable=False with the expected reason). Catches
   logic flips and operator misuse.

3. Golden integration: a realistic post-fix nanoGPT record yields the
   expected applicable set. Catches regressions in the suggester's
   end-to-end behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from profine.catalog.entries import get_catalog
from profine.catalog.schema import CatalogEntry
from profine.schema.architecture_record import (
    ARCHITECTURE_SCHEMA,
    FIELD_LITERALS,
    validate_record,
)
from profine.suggester.applicability import ApplicabilityChecker

CATALOG = get_catalog()


# 1. Field-path lint

def _schema_paths() -> set[str]:
    """All dotted paths the architecture record schema knows about,
    suffixed with `.value` since that's what catalog rules target."""
    paths: set[str] = set()

    def walk(node: dict, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        for key, sub in node.get("properties", {}).items():
            path = f"{prefix}.{key}" if prefix else key
            paths.add(f"{path}.value")
            if isinstance(sub, dict) and sub.get("type") == "object":
                walk(sub, path)

    walk(ARCHITECTURE_SCHEMA, "")
    return paths


@pytest.mark.parametrize("entry", CATALOG, ids=lambda e: e.id)
def test_catalog_field_paths_resolve(entry: CatalogEntry) -> None:
    known = _schema_paths()
    for cond in (*entry.required, *entry.forbidden):
        assert cond.field in known, (
            f"catalog entry {entry.id!r} references unknown field "
            f"{cond.field!r}; add it to ARCHITECTURE_SCHEMA or fix the typo"
        )


# 2. Per-entry applicability (auto-derived)

def _set_path(record: dict, dotted: str, value: Any) -> None:
    """Set record[a][b][c] = {value: ...}; create intermediate dicts."""
    parts = dotted.split(".")
    cur = record
    for part in parts[:-2]:
        cur = cur.setdefault(part, {})
    cur[parts[-2]] = {parts[-1]: value}


def _satisfying_value(operator: str, expected: Any) -> Any:
    """Pick a value that makes a `required` condition true."""
    if operator == "in":
        return expected[0] if isinstance(expected, list) else expected
    if operator == "eq":
        return expected
    if operator == "not_in":
        return "__sentinel_not_in_list__"
    if operator == "gte":
        return expected
    if operator == "lte":
        return expected
    raise NotImplementedError(f"required operator {operator!r}")


def _violating_value(operator: str, expected: Any) -> Any:
    """Pick a value that makes a condition false (for required) or
    triggered (for forbidden)."""
    if operator == "in":
        return "__sentinel_outside_list__"
    if operator == "eq":
        return expected  # for forbidden: triggers; for required: tests violation differently
    if operator == "not_in":
        return expected[0] if isinstance(expected, list) else expected
    if operator == "gte":
        return expected - 1  # below the threshold
    if operator == "lte":
        return expected + 1  # above the threshold
    raise NotImplementedError(f"operator {operator!r}")


def _build_satisfying_record(entry: CatalogEntry) -> dict:
    """Construct an architecture record that makes `entry` applicable."""
    record: dict = {}
    for cond in entry.required:
        _set_path(record, cond.field, _satisfying_value(cond.operator, cond.value))
    return record


@pytest.mark.parametrize("entry", CATALOG, ids=lambda e: e.id)
def test_entry_satisfying_record_is_applicable(entry: CatalogEntry) -> None:
    record = _build_satisfying_record(entry)
    ok, reasons = ApplicabilityChecker(record).check(entry)
    assert ok, f"{entry.id} should be applicable on its own satisfying record; got {reasons}"


@pytest.mark.parametrize("entry", CATALOG, ids=lambda e: e.id)
def test_entry_each_required_rejects_when_violated(entry: CatalogEntry) -> None:
    """For each `required` rule, mutate the satisfying record to violate
    that one rule and confirm the entry is rejected."""
    for i, cond in enumerate(entry.required):
        record = _build_satisfying_record(entry)
        _set_path(record, cond.field, _violating_value(cond.operator, cond.value))
        ok, reasons = ApplicabilityChecker(record).check(entry)
        assert not ok, (
            f"{entry.id} required[{i}] ({cond.field} {cond.operator} {cond.value}) "
            f"should reject when violated, but applicable=True"
        )


@pytest.mark.parametrize("entry", CATALOG, ids=lambda e: e.id)
def test_entry_each_forbidden_rejects_when_triggered(entry: CatalogEntry) -> None:
    for i, cond in enumerate(entry.forbidden):
        record = _build_satisfying_record(entry)
        # Trigger the forbidden condition.
        if cond.operator == "in":
            trigger = cond.value[0] if isinstance(cond.value, list) else cond.value
        elif cond.operator == "eq":
            trigger = cond.value
        elif cond.operator == "not_in":
            trigger = "__sentinel_outside_list__"
        elif cond.operator == "gte":
            trigger = cond.value
        elif cond.operator == "lte":
            trigger = cond.value
        else:
            pytest.fail(f"unhandled forbidden operator {cond.operator!r}")
        _set_path(record, cond.field, trigger)
        ok, reasons = ApplicabilityChecker(record).check(entry)
        assert not ok, (
            f"{entry.id} forbidden[{i}] ({cond.field} {cond.operator} {cond.value}) "
            f"should reject when triggered, but applicable=True"
        )


# 3. Golden integration: post-fix nanoGPT record

NANOGPT_POST_FIX = {
    "precision": {"training_dtype": {"value": "bfloat16"}},
    "compile_mode": {"value": "default"},
    "optimizer": {
        "name": {"value": "AdamW"},
        "fused": {"value": True},
    },
    "distributed": {
        "strategy": {"value": "ddp"},
        "gradient_accumulation_steps": {"value": 40},
    },
    "model_family": {"value": "GPT"},
    "attention_type": {"value": "causal_mha"},
    "attention_impl": {"value": "flash_attention_2"},
}

# Already-enabled optimizations OR known-broken combinations that must
# never appear in nanoGPT's recommendation set.
NANOGPT_FORBIDDEN_RECOMMENDATIONS = {
    "bf16_mixed_precision",     # already on bf16
    "fp16_mixed_precision",     # already mixed precision
    "torch_compile",            # already compiled
    "fused_adamw",              # already fused
    "foreach_adamw",            # already fused
    "flash_attention_2",        # already on flash
    "sdpa",                     # already on flash, which subsumes sdpa
    "torch_compile_max_autotune",  # default CUDAGraphs break grad-accum
}


def test_nanogpt_post_fix_no_already_enabled_recommendations() -> None:
    """The hard contract: an already-fully-tuned nanoGPT must not be
    told to enable optimizations it already has on. Anything else the
    catalog surfaces is the relevance-ranker's responsibility."""
    checker = ApplicabilityChecker(NANOGPT_POST_FIX)
    applicable = {e.id for e, _, _ in checker.filter_catalog(CATALOG)}
    bad = applicable & NANOGPT_FORBIDDEN_RECOMMENDATIONS
    assert not bad, f"already-enabled optimizations leaked through: {bad}"


# 4. Validator (literal contract)

def test_validator_coerces_known_aliases() -> None:
    record = {
        "precision": {"training_dtype": {"value": "fp32"}},
        "compile_mode": {"value": True},
    }
    errors = validate_record(record)
    assert errors == []
    assert record["precision"]["training_dtype"]["value"] == "float32"
    assert record["compile_mode"]["value"] == "default"


def test_validator_flags_unknown_literal() -> None:
    record = {"compile_mode": {"value": "turbo"}}
    errors = validate_record(record)
    assert any("compile_mode" in e for e in errors)


def test_validator_passes_when_field_absent() -> None:
    assert validate_record({}) == []


def test_field_literals_cover_catalog_eq_rules() -> None:
    """Any catalog rule using `eq` on a constrained-literal target is
    only safe if that target is in FIELD_LITERALS — otherwise the LLM
    can emit anything and silently bypass the rule."""
    eq_targets: set[str] = set()
    for entry in CATALOG:
        for cond in (*entry.required, *entry.forbidden):
            if cond.operator == "eq":
                # Strip trailing ".value" to match FIELD_LITERALS keys.
                target = cond.field.removesuffix(".value")
                eq_targets.add(target)
    missing = eq_targets - set(FIELD_LITERALS.keys())
    assert not missing, (
        f"catalog uses `eq` on fields without strict literal validation: "
        f"{missing}; add them to FIELD_LITERALS"
    )
