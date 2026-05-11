"""Code Editor tool (plan 4.5).

Public API:
    from profine.editor import edit

    result = edit(source, candidate, architecture_record, provider="openai")
"""

from __future__ import annotations

from typing import Any

from profine.schema.optimization_candidate import OptimizationCandidate
from profine.editor.editor import CodeEditor, EditResult


def edit(
    source: str,
    candidate: OptimizationCandidate,
    architecture_record: dict[str, Any] | None = None,
    user_preferences: str | None = None,
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
) -> EditResult:
    """Convenience entry point for the Code Editor tool."""
    e = CodeEditor(provider=provider, api_key=api_key, model=model)
    return e.edit(source, candidate, architecture_record, user_preferences)
