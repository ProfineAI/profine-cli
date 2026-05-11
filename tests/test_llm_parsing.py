"""Tests for profine.llm.utils.parse_json_response.

This is the single hot path that all four LLM-driven tools route through,
so it's worth covering every recovery branch (markdown fence, prose
preamble, trailing text, control chars, malformed input).
"""

from __future__ import annotations

import pytest

from profine.llm.utils import LlmJsonParseError, call_and_parse, parse_json_response


class _StubBackend:
    """Returns a queued sequence of responses; records every (system, user) pair."""

    def __init__(self, *responses: str) -> None:
        self._queue = list(responses)
        self.calls: list[tuple[str, str]] = []

    def call(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._queue.pop(0)


def test_plain_object():
    assert parse_json_response('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_plain_array():
    assert parse_json_response('[1, 2, 3]') == [1, 2, 3]


def test_strips_json_fence():
    raw = '```json\n{"x": 1}\n```'
    assert parse_json_response(raw) == {"x": 1}


def test_strips_unlabeled_fence():
    raw = '```\n{"x": 1}\n```'
    assert parse_json_response(raw) == {"x": 1}


def test_strips_prose_preamble():
    raw = "Here's the JSON you asked for:\n{\"x\": 1}"
    assert parse_json_response(raw) == {"x": 1}


def test_ignores_trailing_text():
    raw = '{"x": 1}\n\nLet me know if you want me to elaborate.'
    assert parse_json_response(raw) == {"x": 1}


def test_handles_control_chars_in_string_value():
    # Real LLM output: code embedded in JSON with literal newlines/tabs
    raw = '{"code": "def f():\n    return 1"}'
    assert parse_json_response(raw) == {"code": "def f():\n    return 1"}


def test_handles_tab_in_string_value():
    raw = '{"code": "x\ty"}'
    assert parse_json_response(raw) == {"code": "x\ty"}


def test_combines_fence_and_prose_and_control_chars():
    raw = (
        "Sure! Here it is:\n"
        "```json\n"
        '{"code": "def f():\n    pass"}\n'
        "```\n"
        "Hope that helps!"
    )
    assert parse_json_response(raw) == {"code": "def f():\n    pass"}


def test_empty_raises():
    with pytest.raises(LlmJsonParseError):
        parse_json_response("")


def test_whitespace_only_raises():
    with pytest.raises(LlmJsonParseError):
        parse_json_response("   \n\t  ")


def test_no_json_value_raises():
    with pytest.raises(LlmJsonParseError):
        parse_json_response("This response has no JSON in it at all.")


def test_truncated_json_raises():
    with pytest.raises(LlmJsonParseError):
        parse_json_response('{"a": 1, "b": ')


def test_carries_raw_text_on_failure():
    raw = "definitely not json"
    try:
        parse_json_response(raw)
    except LlmJsonParseError as exc:
        assert exc.raw == raw
    else:
        pytest.fail("expected LlmJsonParseError")


def test_nested_object_preserved():
    raw = '{"outer": {"inner": [1, 2, {"deep": true}]}}'
    assert parse_json_response(raw) == {"outer": {"inner": [1, 2, {"deep": True}]}}


# call_and_parse

def test_call_and_parse_succeeds_first_try():
    
    backend = _StubBackend('{"a": 1}')
    assert call_and_parse(backend, "sys", "user") == {"a": 1}
    assert len(backend.calls) == 1


def test_call_and_parse_retries_with_hint_on_parse_failure():
    # First response unparseable, second is fine
    backend = _StubBackend("not json", '{"ok": true}')
    result = call_and_parse(backend, "sys", "user")
    assert result == {"ok": True}
    assert len(backend.calls) == 2
    # Retry prompt must include the remediation hint
    assert "previous response was not valid JSON" in backend.calls[1][1]
    assert "escape" in backend.calls[1][1].lower()


def test_call_and_parse_raises_after_all_attempts_fail():
    backend = _StubBackend("garbage one", "garbage two")
    with pytest.raises(LlmJsonParseError) as exc_info:
        call_and_parse(backend, "sys", "user", max_attempts=2)
    assert "after 2 attempts" in str(exc_info.value)
    assert len(backend.calls) == 2


def test_call_and_parse_writes_debug_dump_on_failure(tmp_path):
    raw_bad = "this response is broken in some specific way"
    backend = _StubBackend(raw_bad, raw_bad)
    with pytest.raises(LlmJsonParseError) as exc_info:
        call_and_parse(
            backend, "sys", "user",
            debug_dir=tmp_path, debug_label="testtool",
        )
    # The path is mentioned in the error
    assert "testtool" in str(exc_info.value)
    assert "Raw response saved to" in str(exc_info.value)
    # And the dump file actually exists with the raw response
    dumps = list(tmp_path.glob("testtool_*.txt"))
    assert len(dumps) == 1
    assert dumps[0].read_text(encoding="utf-8") == raw_bad


def test_call_and_parse_does_not_dump_on_success(tmp_path):
    backend = _StubBackend('{"ok": true}')
    call_and_parse(backend, "sys", "user", debug_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_call_and_parse_max_attempts_one_means_no_retry():
    backend = _StubBackend("nope")
    with pytest.raises(LlmJsonParseError):
        call_and_parse(backend, "sys", "user", max_attempts=1)
    assert len(backend.calls) == 1


def test_call_and_parse_handles_unescaped_inner_quotes_via_retry():
    # Realistic failure mode: LLM emits a string value with unescaped " inside,
    # which trips "Unterminated string". Second attempt with hint succeeds.
    bad = '{"snippet": "model.to("cuda")"}'
    good = '{"snippet": "model.to(\\"cuda\\")"}'
    backend = _StubBackend(bad, good)
    assert call_and_parse(backend, "sys", "user") == {"snippet": 'model.to("cuda")'}
