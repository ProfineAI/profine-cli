"""Shared LLM backend for Anthropic, OpenAI, and OpenAI-compatible local servers."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, TypeVar

_DEFAULT_TIMEOUT = 120
_DEFAULT_MAX_OUTPUT_TOKENS = 32768
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_SEED: int | None = None

_RETRY_ATTEMPTS = int(os.environ.get("PROFINE_LLM_RETRY_ATTEMPTS", "3"))
_RETRY_BASE_DELAY_SECONDS = float(os.environ.get("PROFINE_LLM_RETRY_BASE_DELAY", "1.5"))

_TRANSIENT_EXC_NAMES = frozenset({
    "RemoteProtocolError", "ReadError", "ReadTimeout",
    "ConnectError", "ConnectTimeout", "WriteError", "PoolTimeout",
    "APIConnectionError", "APITimeoutError",
    "InternalServerError", "RateLimitError",
})

_log = logging.getLogger(__name__)
_T = TypeVar("_T")


def _is_transient(exc: BaseException) -> bool:
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _with_retry(fn: Callable[[], _T], *, label: str) -> _T:
    last_exc: BaseException | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except BaseException as exc:
            if not _is_transient(exc) or attempt == _RETRY_ATTEMPTS:
                raise
            delay = _RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            _log.warning(
                "%s: transient %s on attempt %d/%d (%s); retrying in %.1fs",
                label, type(exc).__name__, attempt, _RETRY_ATTEMPTS, exc, delay,
            )
            last_exc = exc
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


class LlmBackend:
    """Base class: turns (system_prompt, user_message) into a response string."""

    def call(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicBackend(LlmBackend):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = _DEFAULT_TEMPERATURE,
        seed: int | None = _DEFAULT_SEED,
    ) -> None:
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("No Anthropic API key. Set ANTHROPIC_API_KEY or pass api_key=.")
        self.client = anthropic.Anthropic(api_key=key, timeout=_DEFAULT_TIMEOUT)
        self.model = model
        self.temperature = temperature
        # Anthropic's API doesn't accept a seed parameter; we keep the field for
        # parity but ignore it during the call. Temperature=0 is the closest
        # we can get to deterministic sampling.
        self.seed = seed

    def call(self, system: str, user: str) -> str:
        def _once() -> str:
            chunks: list[str] = []
            with self.client.messages.stream(
                model=self.model,
                max_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
            return "".join(chunks)
        return _with_retry(_once, label="anthropic.call")


class OpenAIBackend(LlmBackend):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.4-mini",
        base_url: str | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        seed: int | None = _DEFAULT_SEED,
    ) -> None:
        import openai
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OpenAI API key. Set OPENAI_API_KEY or pass api_key=.")
        client_kwargs: dict[str, Any] = {"api_key": key, "timeout": _DEFAULT_TIMEOUT}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def call(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": _DEFAULT_MAX_OUTPUT_TOKENS,
            "stream": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if self.seed is not None:
            # OpenAI's API treats `seed` as best-effort: same (model, seed,
            # prompt) → same response when system_fingerprint is stable.
            kwargs["seed"] = self.seed
        def _once() -> str:
            stream = self.client.chat.completions.create(**kwargs)
            chunks: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    chunks.append(delta)
            return "".join(chunks)
        return _with_retry(_once, label="openai.call")


class LocalBackend(OpenAIBackend):
    """OpenAI-compatible local backend (Ollama, vLLM, LM Studio, llama.cpp server, LiteLLM).

    Defaults to Ollama at http://localhost:11434/v1. Override via --base-url or
    the PROFINE_LOCAL_BASE_URL / OPENAI_BASE_URL environment variables.
    """

    _DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        seed: int | None = _DEFAULT_SEED,
    ) -> None:
        if not model:
            raise RuntimeError(
                "Local provider requires --model. Example: --model llama3.1:8b (Ollama) "
                "or --model meta-llama/Llama-3.1-8B-Instruct (vLLM)."
            )
        resolved_base = (
            base_url
            or os.environ.get("PROFINE_LOCAL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self._DEFAULT_BASE_URL
        )
        # Local servers typically ignore the key; default to a placeholder so the
        # OpenAI SDK accepts the construction.
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "local"
        super().__init__(
            api_key=resolved_key, model=model, base_url=resolved_base,
            temperature=temperature, seed=seed,
        )


def create_backend(provider: str = "openai", **kwargs: Any) -> LlmBackend:
    """Factory for LLM backends.

    Providers:
      - "openai"    — OpenAI API
      - "anthropic" — Anthropic API
      - "local"     — any OpenAI-compatible local server (Ollama, vLLM, LM Studio, etc.)

    All backends accept `temperature` (default 0.0) and `seed` (best-effort).
    Anthropic ignores `seed` — its API doesn't expose one — but honors temperature=0.
    """
    if provider == "openai":
        return OpenAIBackend(**kwargs)
    if provider == "local":
        return LocalBackend(**kwargs)
    if provider == "anthropic":
        # The Anthropic backend doesn't accept base_url; drop it so older callers don't crash.
        kwargs.pop("base_url", None)
        return AnthropicBackend(**kwargs)
    raise ValueError(f"Unknown provider: {provider!r}. Use 'openai', 'anthropic', or 'local'.")
