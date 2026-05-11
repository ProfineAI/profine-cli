"""Tests for editor safety guardrails.

These checks run on every LLM-edited script before it's accepted, so
they're worth full coverage.
"""

from __future__ import annotations

from profine.editor.editor import _check_syntax, _compute_diff, _is_structural_rewrite


def test_check_syntax_passes_valid_code():
    assert _check_syntax("x = 1\n") is None


def test_check_syntax_returns_message_on_invalid():
    msg = _check_syntax("def f(:\n    pass\n")
    assert msg is not None
    assert "SyntaxError" in msg


def test_pure_addition_is_not_structural_rewrite():
    original = "import torch\nmodel = build()\nfor x in loader:\n    train(x)\n"
    edited = "import torch\nmodel = build()\nmodel = torch.compile(model)\nfor x in loader:\n    train(x)\n"
    assert _is_structural_rewrite(original, edited) is False


def test_added_def_trips_guardrail():
    original = "model = build()\nfor x in loader:\n    train(x)\n"
    # LLM invented a brand-new helper — we treat that as a structural rewrite
    edited = "def helper():\n    pass\nmodel = build()\nfor x in loader:\n    train(x)\n"
    assert _is_structural_rewrite(original, edited) is True


def test_added_class_trips_guardrail():
    original = "x = 1\n"
    edited = "class NewThing:\n    pass\nx = 1\n"
    assert _is_structural_rewrite(original, edited) is True


def test_large_deletion_trips_guardrail():
    original = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n"
    edited = "a = 1\n"  # dropped 5 non-blank lines
    assert _is_structural_rewrite(original, edited) is True


def test_renaming_existing_def_does_not_trip():
    # Existing def reused; no NEW defs introduced
    original = "def train():\n    pass\n"
    edited = "def train():\n    return 1\n"
    assert _is_structural_rewrite(original, edited) is False


def test_compute_diff_produces_unified_format():
    original = "a = 1\n"
    edited = "a = 2\n"
    diff = _compute_diff(original, edited)
    assert "-a = 1" in diff
    assert "+a = 2" in diff
    assert "original" in diff
    assert "optimized" in diff


def test_compute_diff_empty_for_identical():
    assert _compute_diff("x = 1\n", "x = 1\n") == ""
