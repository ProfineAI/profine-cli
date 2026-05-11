"""LLM response utilities — JSON extraction.

LLM outputs are messy: markdown fences ```json ... ```, prose preamble
("Here's the analysis:"), trailing commentary after the JSON object,
literal control characters (raw \\n / \\t) inside string values.

`parse_json_response` is the single robust path used by every tool that
asks an LLM for JSON. Centralising it means subtle parser fixes land
once, not four times.

`call_and_parse` adds the orchestration layer on top: one retry with a
remediation hint if the first parse fails (LLMs almost always self-
correct when told what they got wrong), and an optional debug-dump of
the raw response so a human can inspect it after a hard failure.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol


class LlmJsonParseError(ValueError):
    """Raised when an LLM response can't be parsed as JSON even after
    fence stripping and control-char salvage. Carries the raw text so
    callers can log it for debugging."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


def parse_json_response(raw: str) -> Any:
    """Parse a JSON value out of an LLM response.

    Tolerates: ```json/``` fences, prose before the value, trailing text
    after the value, and unescaped control characters inside string
    literals. Raises `LlmJsonParseError` if no valid JSON can be found.
    """
    if not raw or not raw.strip():
        raise LlmJsonParseError("empty response", raw)

    text = _strip_markdown_fence(raw.strip())
    start = _first_json_start(text)
    if start < 0:
        raise LlmJsonParseError("no JSON object or array in response", raw)
    text = text[start:]

    decoder = json.JSONDecoder(strict=False)
    try:
        obj, _end = decoder.raw_decode(text)
        return obj
    except json.JSONDecodeError:
        try:
            salvaged = _escape_unescaped_controls(text)
            obj, _end = decoder.raw_decode(salvaged)
            return obj
        except json.JSONDecodeError as e:
            raise LlmJsonParseError(f"JSON decode failed: {e}", raw) from e


def _strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` (or any ```lang) wrapping, if present."""
    if not text.startswith("```"):
        return text
    try:
        first_nl = text.index("\n")
    except ValueError:
        return text
    inner = text[first_nl + 1:]
    if inner.endswith("```"):
        inner = inner[:-3]
    return inner.strip()


def _first_json_start(text: str) -> int:
    """Index of the first '{' or '[' — i.e. the start of the JSON value.
    Returns -1 if the text contains no JSON value at all."""
    for i, ch in enumerate(text):
        if ch in "{[":
            return i
    return -1


class _CallableBackend(Protocol):
    def call(self, system: str, user: str) -> str: ...


_RETRY_HINT = (
    "\n\n## Your previous response was not valid JSON\n"
    "Specifically: {error}\n\n"
    "Re-emit the full response. Make sure that:\n"
    "- Every literal `\"` inside a string value is escaped as `\\\"`.\n"
    "- Every literal `\\` inside a string value is escaped as `\\\\`.\n"
    "- Newlines inside string values are written as `\\n`, not raw newlines.\n"
    "- The response is a single JSON value with no trailing prose.\n"
)


def call_and_parse(
    backend: _CallableBackend,
    system: str,
    user: str,
    *,
    max_attempts: int = 2,
    debug_dir: str | Path | None = None,
    debug_label: str = "llm_response",
) -> Any:
    """Call the LLM, parse the JSON response, retry once on parse failure.

    LLMs occasionally emit JSON with unescaped quotes or backslashes inside
    string values. They almost always self-correct when told the parse
    failed and shown what to fix; one retry catches the common case
    cheaply without runaway loops.

    On final failure raises `LlmJsonParseError`. If `debug_dir` is set,
    the raw response from the last attempt is written to
    `<debug_dir>/<debug_label>_<timestamp>.txt` and the error message
    points at the file so a human can inspect what came back.
    """
    last_err: str = ""
    last_raw: str = ""
    user_msg = user
    for attempt in range(max_attempts):
        raw = backend.call(system, user_msg)
        last_raw = raw
        try:
            return parse_json_response(raw)
        except LlmJsonParseError as e:
            last_err = str(e)
            user_msg = user + _RETRY_HINT.format(error=last_err)

    debug_path: Path | None = None
    if debug_dir is not None:
        try:
            ddir = Path(debug_dir)
            ddir.mkdir(parents=True, exist_ok=True)
            debug_path = ddir / f"{debug_label}_{int(time.time())}.txt"
            debug_path.write_text(last_raw, encoding="utf-8")
        except OSError:
            debug_path = None  # disk write failed — surface the parse error anyway

    suffix = f". Raw response saved to {debug_path}" if debug_path else ""
    raise LlmJsonParseError(f"{last_err} (after {max_attempts} attempts){suffix}", last_raw)


def _escape_unescaped_controls(text: str) -> str:
    """Re-escape literal control bytes (raw \\n, \\t, etc.) inside JSON
    string literals so json.loads can parse them. Tracks string spans
    honoring backslash escapes; outside strings the input is unchanged.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string and ch < " ":
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(f"\\u{ord(ch):04x}")
            continue
        out.append(ch)
    return "".join(out)
