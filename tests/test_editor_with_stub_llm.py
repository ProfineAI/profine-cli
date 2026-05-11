"""End-to-end Editor tests with a stubbed LLM backend.

Exercises the full edit() path without hitting the network: prompt
building, JSON parsing, syntax healing trigger, structural-rewrite
guardrail, and multi-file edits.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from profine.editor.editor import CodeEditor, EditResult
from profine.llm.backend import LlmBackend
from profine.schema.optimization_candidate import OptimizationCandidate


class StubBackend(LlmBackend):
    """Returns a canned response. Records the last prompt for assertions."""

    def __init__(self, response: str | dict | list[str | dict]) -> None:
        # Allow a list to simulate a multi-step LLM (initial + heal attempts)
        self._queue: list[Any] = response if isinstance(response, list) else [response]
        self.calls: list[tuple[str, str]] = []

    def call(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        nxt = self._queue.pop(0) if self._queue else self._queue
        if isinstance(nxt, dict):
            return json.dumps(nxt)
        return nxt


def _editor_with(backend: StubBackend) -> CodeEditor:
    editor = CodeEditor.__new__(CodeEditor)
    editor._backend = backend
    editor._max_heal_attempts = 1
    return editor


def _candidate(entry_id: str = "torch_compile") -> OptimizationCandidate:
    return OptimizationCandidate(
        entry_id=entry_id,
        category="compile",
        name="torch.compile",
        description="Compile the model",
    )


def test_edit_applies_simple_change():
    src = "import torch\nmodel = build()\n"
    edited = "import torch\nmodel = torch.compile(build())\n"
    backend = StubBackend({
        "applied": True,
        "edited_source": edited,
        "explanation": "Wrapped build() in torch.compile",
        "changes": [{"line_start": 2, "description": "compile"}],
    })
    result = _editor_with(backend).edit(src, _candidate())
    assert result.applied
    assert "torch.compile" in result.edited_source
    assert "Wrapped" in result.explanation
    assert result.optimization_id == "torch_compile"
    assert "+model = torch.compile" in result.diff


def test_edit_not_applicable_returns_unchanged():
    backend = StubBackend({
        "applied": False,
        "edited_source": "",
        "not_applicable_reason": "no model loader found",
    })
    result = _editor_with(backend).edit("x = 1\n", _candidate())
    assert not result.applied
    assert result.not_applicable_reason == "no model loader found"


def test_edit_invalid_syntax_triggers_heal():
    # First response is broken, healer fixes it
    backend = StubBackend([
        {"applied": True, "edited_source": "def f(:\n    pass\n"},
        {"edited_source": "def f():\n    pass\n"},
    ])
    result = _editor_with(backend).edit("x = 1\n", _candidate())
    assert result.applied
    # Heal happened — second LLM call was made
    assert len(backend.calls) == 2


def test_structural_rewrite_with_file_edits_reverts_entry():
    # LLM returns both an entry rewrite AND file_edits — guardrail trips
    src = "from helper import Net\nmodel = Net()\n"
    bad_entry_inlining = (
        "class Net:\n    pass\nclass Trainer:\n    pass\nmodel = Net()\n"
    )
    backend = StubBackend({
        "applied": True,
        "edited_source": bad_entry_inlining,
        "file_edits": [{"path": "helper.py", "edited_source": "class Net:\n    def __init__(self): pass\n"}],
    })
    result = _editor_with(backend).edit(
        src,
        _candidate(),
        local_modules={"helper.py": "class Net:\n    pass\n"},
    )
    # Entry source should be reverted; only file_edit applied
    assert result.edited_source == src
    assert any("structurally rewritten" in w for w in result.warnings)
    assert len(result.extra_file_edits) == 1


def test_file_edits_to_unknown_paths_are_ignored():
    backend = StubBackend({
        "applied": True,
        "edited_source": "x = 1\n",
        "file_edits": [
            {"path": "/etc/passwd", "edited_source": "owned"},
            {"path": "../../../escape.py", "edited_source": "owned"},
        ],
    })
    result = _editor_with(backend).edit(
        "x = 1\n",
        _candidate(),
        local_modules={"good.py": "ok"},
    )
    assert result.extra_file_edits == []
    assert any("Ignored edit to unknown file" in w for w in result.warnings)


def test_file_edit_unchanged_is_skipped():
    # LLM returned a file_edit identical to the original — no-op
    backend = StubBackend({
        "applied": True,
        "edited_source": "x = 1\n",
        "file_edits": [{"path": "m.py", "edited_source": "same\n"}],
    })
    result = _editor_with(backend).edit(
        "x = 1\n",
        _candidate(),
        local_modules={"m.py": "same\n"},
    )
    assert result.extra_file_edits == []


def test_edit_carries_warnings_from_llm():
    backend = StubBackend({
        "applied": True,
        "edited_source": "x = 1\n",
        "warnings": ["heuristic risk noted"],
    })
    result = _editor_with(backend).edit("x = 1\n", _candidate())
    assert "heuristic risk noted" in result.warnings


def test_edit_handles_markdown_fenced_response():
    raw = '```json\n{"applied": true, "edited_source": "x = 2\\n"}\n```'
    backend = StubBackend(raw)
    result = _editor_with(backend).edit("x = 1\n", _candidate())
    assert result.applied
    assert result.edited_source == "x = 2\n"
