"""Tests for the user preferences markdown loader."""

from __future__ import annotations

from profine.preferences.loader import UserPreferences, load_preferences


def test_empty_returns_default_prefs():
    prefs = load_preferences()
    assert isinstance(prefs, UserPreferences)
    assert prefs.raw == ""
    assert prefs.risk_tolerance == "medium"


def test_raw_is_passed_through_unchanged():
    text = "## My prefs\n- gpu: H100 x 8\n- random unstructured note"
    prefs = load_preferences(raw=text)
    assert prefs.raw == text


def test_extracts_gpu():
    prefs = load_preferences(raw="- gpu: H100 x 8")
    assert "h100" in prefs.hardware


def test_extracts_goal():
    prefs = load_preferences(raw="- primary: reduce step time")
    assert "reduce step time" in prefs.goal


def test_extracts_risk_tolerance_valid():
    for level in ("conservative", "medium", "experimental"):
        prefs = load_preferences(raw=f"- level: {level}")
        assert prefs.risk_tolerance == level


def test_invalid_risk_tolerance_keeps_default():
    prefs = load_preferences(raw="- level: yolo")
    assert prefs.risk_tolerance == "medium"


def test_extracts_rtol_atol():
    prefs = load_preferences(raw="rtol = 1e-3\natol = 5e-5")
    assert prefs.rtol == 1e-3
    assert prefs.atol == 5e-5


def test_extracts_do_not_touch_list():
    prefs = load_preferences(raw="- do_not_touch: [optimizer_choice, learning_rate]")
    assert prefs.do_not_touch == ["optimizer_choice", "learning_rate"]


def test_extracts_max_iterations():
    prefs = load_preferences(raw="- max_iterations: 12")
    assert prefs.max_iterations == 12


def test_invalid_max_iterations_keeps_default():
    prefs = load_preferences(raw="- max_iterations: not-a-number")
    # silent fallback: extraction failed, default preserved
    assert prefs.max_iterations == 8


def test_extracts_max_wall_clock_minutes():
    prefs = load_preferences(raw="- max_wall_clock: 45 min")
    assert prefs.max_wall_clock_minutes == 45


def test_load_from_file(tmp_path):
    p = tmp_path / "prefs.md"
    p.write_text("- level: experimental\n- max_iterations: 5", encoding="utf-8")
    prefs = load_preferences(p)
    assert prefs.risk_tolerance == "experimental"
    assert prefs.max_iterations == 5
