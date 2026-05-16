"""Single registry of profine's environment variables.

Two purposes:

  1. Documentation. `profine env` reads this file to show every
     PROFINE_* var with its description, default, source location,
     and currently-resolved value. One audit point for "what knobs
     does this thing expose."

  2. Avoid drift. New env vars added in code without an entry here
     are invisible to the user; reviewers can require both touches
     in the same PR.

Categories are presentation-only. Resolved value is computed lazily
so importing this module has no side effects.

Note: any var listed in profine.auth.MANAGED_KEYS can also be saved
to ~/.profine/auth.json via `profine auth login`. The CLI fills
unset env vars from that file at startup; an explicit env var always
wins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class EnvVar:
    """A documented environment variable."""
    name: str
    category: str       # 'telemetry' | 'profiler' | 'llm' | 'compute' | 'paths'
    description: str
    default: str | None  # None means "no default, must be set if used"
    resolved_by: Callable[[], str | None] = lambda: None
    referenced_in: tuple[str, ...] = ()  # files that read this var


def _env_str(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val is not None else default


# Each entry is the canonical home for one env var. Adding a new one
# here is the convention; code that reads the var should also be
# listed in `referenced_in` so future contributors can find usages.
REGISTRY: tuple[EnvVar, ...] = (
    # ---------- telemetry ----------
    EnvVar(
        name="PROFINE_API_KEY",
        category="telemetry",
        description="Paid customer's bearer key (pf_live_…). Routes telemetry "
                    "to /api/telemetry/run with auth; bypasses the OSS consent file.",
        default=None,
        resolved_by=lambda: _env_str("PROFINE_API_KEY"),
        referenced_in=("profine/telemetry/builder.py", "profine/cli/commands.py"),
    ),
    EnvVar(
        name="PROFINE_API_URL",
        category="telemetry",
        description="Override the telemetry backend URL. Used by the recorder "
                    "and PriorsClient. Default is the production endpoint.",
        default="https://api.profine.ai",
        resolved_by=lambda: _env_str("PROFINE_API_URL", "https://api.profine.ai"),
        referenced_in=("profine/telemetry/builder.py", "profine/telemetry/priors.py",
                       "profine/cli/commands.py"),
    ),
    EnvVar(
        name="PROFINE_NO_TELEMETRY",
        category="telemetry",
        description="Disable telemetry entirely for this invocation (truthy: "
                    "1/true/yes/on). Equivalent to --no-telemetry; env wins on conflict.",
        default=None,
        resolved_by=lambda: _env_str("PROFINE_NO_TELEMETRY"),
        referenced_in=("profine/telemetry/consent.py", "profine/cli/commands.py"),
    ),
    EnvVar(
        name="PROFINE_HOME",
        category="telemetry",
        description="Override the per-user state directory. Defaults to ~/.profine. "
                    "Used for the OSS consent file.",
        default=None,
        resolved_by=lambda: _env_str("PROFINE_HOME"),
        referenced_in=("profine/telemetry/consent.py",),
    ),

    # ---------- profiler (adaptive warmup) ----------
    EnvVar(
        name="PROFINE_ADAPTIVE_WARMUP",
        category="profiler",
        description="Enable online adaptive warmup detection inside the "
                    "instrumented training loop. Truthy values turn it on.",
        default=None,
        resolved_by=lambda: _env_str("PROFINE_ADAPTIVE_WARMUP"),
        referenced_in=("profine/profiler/hooks.py", "profine/cli/commands.py"),
    ),
    EnvVar(
        name="PROFINE_ADAPTIVE_ACTIVE_STEPS",
        category="profiler",
        description="Number of steady-state steps to capture after the adaptive "
                    "warmup detector triggers. Default 20.",
        default="20",
        resolved_by=lambda: _env_str("PROFINE_ADAPTIVE_ACTIVE_STEPS", "20"),
        referenced_in=("profine/profiler/hooks.py",),
    ),
    EnvVar(
        name="PROFINE_TOTAL_STEPS",
        category="profiler",
        description="Authoritative override of total step budget, injected by "
                    "the executor into the remote profiling container. Internal.",
        default=None,
        resolved_by=lambda: _env_str("PROFINE_TOTAL_STEPS"),
        referenced_in=("profine/profiler/hooks.py", "profine/modal/executor.py",
                       "profine/local/executor.py", "profine/skypilot/task_builder.py"),
    ),

    # ---------- LLM ----------
    EnvVar(
        name="OPENAI_API_KEY",
        category="llm",
        description="OpenAI API key, used when --provider=openai.",
        default=None,
        resolved_by=lambda: _redact(_env_str("OPENAI_API_KEY")),
        referenced_in=("profine/llm/backend.py", "backend/core/config.py"),
    ),
    EnvVar(
        name="ANTHROPIC_API_KEY",
        category="llm",
        description="Anthropic API key, used when --provider=anthropic.",
        default=None,
        resolved_by=lambda: _redact(_env_str("ANTHROPIC_API_KEY")),
        referenced_in=("profine/llm/backend.py", "backend/core/config.py"),
    ),
    EnvVar(
        name="PROFINE_LOCAL_BASE_URL",
        category="llm",
        description="Override URL for --provider=local (OpenAI-compatible local "
                    "server, e.g. Ollama). Defaults to http://localhost:11434/v1.",
        default=None,
        resolved_by=lambda: _env_str("PROFINE_LOCAL_BASE_URL"),
        referenced_in=("profine/llm/backend.py",),
    ),

    # ---------- compute (Modal etc.) ----------
    EnvVar(
        name="MODAL_TOKEN_ID",
        category="compute",
        description="Modal token id for the user's Modal account. Required to "
                    "run --compute-backend=modal.",
        default=None,
        resolved_by=lambda: _redact(_env_str("MODAL_TOKEN_ID")),
        referenced_in=("profine/modal/executor.py", "backend/core/config.py"),
    ),
    EnvVar(
        name="MODAL_TOKEN_SECRET",
        category="compute",
        description="Modal token secret. Required with MODAL_TOKEN_ID.",
        default=None,
        resolved_by=lambda: _redact(_env_str("MODAL_TOKEN_SECRET")),
        referenced_in=("profine/modal/executor.py", "backend/core/config.py"),
    ),
    EnvVar(
        name="HF_TOKEN",
        category="compute",
        description="HuggingFace token (optional). Needed to pull gated models "
                    "or push to private repos.",
        default=None,
        resolved_by=lambda: _redact(_env_str("HF_TOKEN")),
        referenced_in=("profine/modal/image_builder.py",),
    ),

    # ---------- debugging ----------
    EnvVar(
        name="PROFINE_DEBUG",
        category="debug",
        description="Enable verbose error messages (truthy: 1/true/yes/on). "
                    "Off by default for clean user-facing errors.",
        default=None,
        resolved_by=lambda: _env_str("PROFINE_DEBUG"),
        referenced_in=("profine/cli/errors.py",),
    ),
)


def _redact(value: str | None) -> str | None:
    """Show first/last few chars of secrets, ellipsis in the middle.

    `profine env` calls this for any API-key-ish var so the output
    is safe to paste into a bug report.
    """
    if value is None:
        return None
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def all_vars() -> tuple[EnvVar, ...]:
    return REGISTRY


def by_category() -> dict[str, list[EnvVar]]:
    """Group entries by category, preserving registry order within."""
    out: dict[str, list[EnvVar]] = {}
    for entry in REGISTRY:
        out.setdefault(entry.category, []).append(entry)
    return out


__all__ = ["EnvVar", "REGISTRY", "all_vars", "by_category"]
