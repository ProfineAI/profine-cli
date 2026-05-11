"""Shared LLM backend for Anthropic and OpenAI.

Both the reader and profiler modules use this for LLM calls.
"""

from __future__ import annotations

import os
from typing import Any

_DEFAULT_TIMEOUT = 120  # seconds
_DEFAULT_MAX_OUTPUT_TOKENS = 32768


class LlmBackend:
    """Base class: turns (system_prompt, user_message) into a response string."""

    def call(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicBackend(LlmBackend):
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6") -> None:
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("No Anthropic API key. Set ANTHROPIC_API_KEY or pass api_key=.")
        self.client = anthropic.Anthropic(api_key=key, timeout=_DEFAULT_TIMEOUT)
        self.model = model

    def call(self, system: str, user: str) -> str:
        chunks: list[str] = []
        with self.client.messages.stream(
            model=self.model,
            max_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
        return "".join(chunks)


class OpenAIBackend(LlmBackend):
    def __init__(self, api_key: str | None = None, model: str = "gpt-5.4-mini") -> None:
        import openai
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OpenAI API key. Set OPENAI_API_KEY or pass api_key=.")
        self.client = openai.OpenAI(api_key=key, timeout=_DEFAULT_TIMEOUT)
        self.model = model

    def call(self, system: str, user: str) -> str:
        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        chunks: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
        return "".join(chunks)


def create_backend(provider: str = "openai", **kwargs: Any) -> LlmBackend:
    """Factory for LLM backends."""
    if provider == "openai":
        return OpenAIBackend(**kwargs)
    return AnthropicBackend(**kwargs)
