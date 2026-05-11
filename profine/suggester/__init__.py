"""Suggest Optimizations tool (plan 4.4).

Public API:
    from profine.suggester import suggest

    result = suggest(architecture_record, bottleneck_report, provider="openai")
"""

from __future__ import annotations

from typing import Any

from profine.schema.bottleneck_report import BottleneckReport
from profine.suggester.suggester import OptimizationSuggester, SuggestResult


def suggest(
    architecture_record: dict[str, Any],
    bottleneck_report: BottleneckReport | None = None,
    user_preferences: str | None = None,
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
) -> SuggestResult:
    """Convenience entry point for the Suggest Optimizations tool."""
    s = OptimizationSuggester(provider=provider, api_key=api_key, model=model)
    return s.suggest(architecture_record, bottleneck_report, user_preferences)
