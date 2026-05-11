"""LLM-driven script instrumentation for profiling.

Takes a training script and uses an LLM to rewrite it with:
- torch.profiler.profile() wrapping the training loop
- profine hook prelude for step counting, loss capture, GPU sampling
- Step limit enforcement via StepLimitReached
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from profine.llm.backend import LlmBackend
from profine.profiler.event_collector import PROFILER_ACTIVE_STEPS_CAP, compute_scale_factor
from profine.profiler.prompts import build_healing_prompt, build_instrumentation_prompt
from profine.reader.extractor import CodeFacts


@dataclass(slots=True)
class InstrumentedScript:
    """Result of LLM instrumentation."""
    source: str
    original_source: str
    active_steps: int
    scale_factor: float


class ScriptInstrumentor:
    """Uses an LLM to instrument a training script for profiling."""

    def __init__(self, backend: LlmBackend) -> None:
        self._backend = backend

    def instrument(
        self,
        source: str,
        facts: CodeFacts,
        total_steps: int = 60,
        warmup_steps: int = 30,
        benchmark_mode: bool = False,
    ) -> InstrumentedScript:
        """Rewrite the script for profiling or benchmarking.

        Args:
            source: Original script source code.
            facts: Pre-extracted CodeFacts from the extractor.
            total_steps: Total optimizer steps to run.
            warmup_steps: Steps to discard as warmup.
            benchmark_mode: If True, skip torch.profiler (lighter, faster).

        Returns:
            InstrumentedScript with the rewritten source.
        """
        active_steps = total_steps - warmup_steps
        capped_active = min(active_steps, PROFILER_ACTIVE_STEPS_CAP)
        scale_factor = compute_scale_factor(active_steps)

        system, user = build_instrumentation_prompt(
            source=source,
            facts=facts,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            active_steps=capped_active,
            benchmark_mode=benchmark_mode,
        )

        raw = self._backend.call(system, user)
        rewritten = _clean_response(raw)
        _validate(rewritten)

        return InstrumentedScript(
            source=rewritten,
            original_source=source,
            active_steps=capped_active,
            scale_factor=scale_factor,
        )

    def heal(
        self,
        instrumented: InstrumentedScript,
        error_traceback: str,
        hint: str = "",
    ) -> InstrumentedScript:
        traceback = (
            f"## Repair hint from orchestrator\n{hint}\n\n## Error\n{error_traceback}"
            if hint else error_traceback
        )
        system, user = build_healing_prompt(
            instrumented_source=instrumented.source,
            error_traceback=traceback,
            original_source=instrumented.original_source,
        )

        raw = self._backend.call(system, user)
        rewritten = _clean_response(raw)
        _validate(rewritten)

        return InstrumentedScript(
            source=rewritten,
            original_source=instrumented.original_source,
            active_steps=instrumented.active_steps,
            scale_factor=instrumented.scale_factor,
        )


def _clean_response(raw: str) -> str:
    """Strip markdown fences if the LLM wrapped the code."""
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _validate(source: str) -> None:
    """Validate that the instrumented script is syntactically correct."""
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Instrumented script has syntax errors: {e}") from e
