"""Edge cases for the applicability checker.

test_catalog already covers the per-entry positive/negative cases; this
file targets the operator semantics (in / not_in / eq / gte / lte /
exists / contains) and the missing-field behaviour.
"""

from __future__ import annotations

from profine.catalog.schema import ApplicabilityCondition, CatalogEntry
from profine.suggester.applicability import ApplicabilityChecker


def _entry(*reqs: ApplicabilityCondition, forbid: list[ApplicabilityCondition] | None = None) -> CatalogEntry:
    return CatalogEntry(
        id="test", category="test", name="test", description="",
        required=list(reqs), forbidden=list(forbid or []),
    )


def _arch(**kwargs) -> dict:
    # Build an architecture-record-shaped dict where each key is a
    # `{value: ...}` field per the schema convention.
    return {k: {"value": v} for k, v in kwargs.items()}


def test_eq_passes_on_match():
    entry = _entry(ApplicabilityCondition(field="dtype.value", operator="eq", value="float32"))
    ok, _ = ApplicabilityChecker(_arch(dtype="float32")).check(entry)
    assert ok


def test_eq_fails_on_mismatch():
    entry = _entry(ApplicabilityCondition(field="dtype.value", operator="eq", value="float32"))
    ok, reasons = ApplicabilityChecker(_arch(dtype="bfloat16")).check(entry)
    assert not ok
    assert reasons


def test_in_operator():
    entry = _entry(ApplicabilityCondition(
        field="attn.value", operator="in", value=["mha", "gqa"]
    ))
    assert ApplicabilityChecker(_arch(attn="mha")).check(entry)[0]
    assert not ApplicabilityChecker(_arch(attn="other")).check(entry)[0]


def test_not_in_operator():
    entry = _entry(ApplicabilityCondition(
        field="impl.value", operator="not_in", value=["flash_attention_2"]
    ))
    # Already using flash → not applicable
    assert not ApplicabilityChecker(_arch(impl="flash_attention_2")).check(entry)[0]
    # Using SDPA → applicable
    assert ApplicabilityChecker(_arch(impl="sdpa")).check(entry)[0]


def test_not_in_with_missing_field_passes():
    # Missing field + not_in = no forbidden value → condition passes
    entry = _entry(ApplicabilityCondition(field="missing.value", operator="not_in", value=["x"]))
    ok, _ = ApplicabilityChecker({}).check(entry)
    assert ok


def test_missing_field_for_other_operators_fails():
    entry = _entry(ApplicabilityCondition(field="missing.value", operator="eq", value="x"))
    ok, reasons = ApplicabilityChecker({}).check(entry)
    assert not ok
    assert any("not found" in r for r in reasons)


def test_gte_operator():
    entry = _entry(ApplicabilityCondition(field="size.value", operator="gte", value=1024))
    assert ApplicabilityChecker(_arch(size=2048)).check(entry)[0]
    assert ApplicabilityChecker(_arch(size=1024)).check(entry)[0]
    assert not ApplicabilityChecker(_arch(size=512)).check(entry)[0]


def test_lte_operator():
    entry = _entry(ApplicabilityCondition(field="size.value", operator="lte", value=1024))
    assert ApplicabilityChecker(_arch(size=512)).check(entry)[0]
    assert not ApplicabilityChecker(_arch(size=2048)).check(entry)[0]


def test_forbidden_blocks_when_triggered():
    entry = _entry(
        ApplicabilityCondition(field="dtype.value", operator="eq", value="float32"),
        forbid=[ApplicabilityCondition(field="impl.value", operator="eq", value="flash")],
    )
    ok, reasons = ApplicabilityChecker(_arch(dtype="float32", impl="flash")).check(entry)
    assert not ok
    assert any("Forbidden" in r for r in reasons)


def test_forbidden_inactive_when_field_missing():
    # Forbidden condition for a missing field → not triggered
    entry = _entry(
        ApplicabilityCondition(field="dtype.value", operator="eq", value="float32"),
        forbid=[ApplicabilityCondition(field="missing.value", operator="eq", value="x")],
    )
    ok, _ = ApplicabilityChecker(_arch(dtype="float32")).check(entry)
    assert ok


def test_all_required_must_pass():
    entry = _entry(
        ApplicabilityCondition(field="a.value", operator="eq", value=1),
        ApplicabilityCondition(field="b.value", operator="eq", value=2),
    )
    assert ApplicabilityChecker(_arch(a=1, b=2)).check(entry)[0]
    assert not ApplicabilityChecker(_arch(a=1, b=3)).check(entry)[0]
    assert not ApplicabilityChecker(_arch(a=2, b=2)).check(entry)[0]
