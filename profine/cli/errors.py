"""User-facing error formatting for the CLI.

The pipeline can fail in five well-known ways. For each we want a
single-line message that tells the user what to do, instead of dumping
a traceback that buries the cause.

Use `format_user_error` to convert an exception into (message, exit_code).
Unrecognised exceptions return None so the caller can re-raise and let
the traceback through (real bugs should still surface during dev).
"""

from __future__ import annotations

import os
from pathlib import Path


def format_user_error(exc: BaseException) -> tuple[str, int] | None:
    """Map a known exception to a friendly (message, exit_code) tuple.

    Returns None for unrecognised exceptions — the caller should re-raise
    so the traceback isn't swallowed.
    """
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()

    # Missing API keys — RuntimeError raised by AnthropicBackend / OpenAIBackend
    if name == "RuntimeError" and "api key" in low:
        provider = "OpenAI" if "openai" in low else "Anthropic"
        env_var = "OPENAI_API_KEY" if provider == "OpenAI" else "ANTHROPIC_API_KEY"
        return (
            f"No {provider} API key found. Set {env_var}=... in your environment "
            f"or pass --api-key. (For other providers use --provider.)",
            2,
        )

    # Missing third-party SDK — anthropic / openai not installed
    if isinstance(exc, ImportError):
        pkg = (msg.split("'")[1] if "'" in msg else msg).strip()
        return (f"Missing dependency: {pkg}. Install with: pip install {pkg}", 2)

    # Modal auth not configured — Modal raises AuthError or RuntimeError mentioning
    # tokens. We match by message so we don't depend on the modal package being importable.
    if "modal" in low and ("token" in low or "auth" in low or "not authenticated" in low):
        return (
            "Modal is not authenticated. Run `modal token new` to set up credentials, "
            "or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in your environment.",
            2,
        )

    # File not found — script path that doesn't exist.
    # Render exc.filename directly (no Path round-trip) so the displayed path
    # keeps the separators the user passed; this also keeps the cross-platform
    # test assertion (which uses forward slashes) stable on Windows.
    if isinstance(exc, FileNotFoundError):
        target = exc.filename or msg
        hint = _dataset_prep_hint(target)
        if hint:
            return (f"File not found: {target}\n\n{hint}", 2)
        return (f"File not found: {target}", 2)

    # Permission denied — output dir that's not writable
    if isinstance(exc, PermissionError):
        target = exc.filename or "?"
        return (
            f"Permission denied: {target}. Pick a different --output directory or "
            f"check filesystem permissions.",
            2,
        )

    # Unknown hardware preset — KeyError / ValueError from get_hardware()
    if name in ("KeyError", "ValueError") and "hardware" in low:
        return (
            f"{msg} Run `profine list-hardware` for valid presets, or pass a custom "
            f"HardwareConfig via the API.",
            2,
        )

    # OpenAI/httpx connection failure — typically `--provider local` with no
    # server listening. Surfaces as openai.APIConnectionError (which wraps
    # httpx.ConnectError) or a bare httpx.ConnectError.
    if name in ("APIConnectionError", "ConnectError", "ConnectionRefusedError") or (
        "connection error" in low or "connection refused" in low or "all connection attempts failed" in low
    ):
        return (
            "Could not connect to the LLM endpoint. If you're using --provider local, "
            "make sure your OpenAI-compatible server (Ollama, vLLM, LM Studio, ...) is "
            "running and reachable. Default base URL is http://localhost:11434/v1 "
            "(Ollama); override with --base-url or PROFINE_LOCAL_BASE_URL.",
            2,
        )

    # Malformed JSON from the LLM after retries — wraps LlmJsonParseError
    if name == "LlmJsonParseError":
        # Surface the saved debug-dump path if call_and_parse wrote one
        # (the path is appended to the error message in that case).
        return (
            "LLM returned malformed JSON we couldn't recover after retry. "
            "This is usually transient — try re-running, or switch --provider / "
            "--model. If it keeps happening, inspect the saved response for "
            f"clues. Details: {msg}",
            3,
        )

    return None


def print_user_error(exc: BaseException) -> int:
    """Print a friendly error if recognised, else return -1 to signal that
    the caller should re-raise. On match returns the exit code."""
    formatted = format_user_error(exc)
    if formatted is None:
        return -1
    msg, code = formatted
    # stderr so users grepping stdout for output paths don't pick this up
    import sys
    print(f"Error: {msg}", file=sys.stderr)
    return code


def is_debug_mode() -> bool:
    """When PROFINE_DEBUG=1 we always re-raise so devs see the traceback."""
    return os.environ.get("PROFINE_DEBUG", "").strip() not in ("", "0", "false", "False")


_DATASET_BIN_SUFFIXES = (".bin", ".npy", ".arrow", ".parquet")


def _dataset_prep_hint(missing_path: str) -> str | None:
    """Suggest running a sibling `prepare.py` when the missing file looks
    like a tokenized dataset (nanoGPT/minGPT-style data/<name>/ layout)."""
    try:
        path = Path(missing_path)
    except (TypeError, ValueError):
        return None
    if path.suffix not in _DATASET_BIN_SUFFIXES:
        return None
    prep = path.parent / "prepare.py"
    if not prep.exists():
        return None
    return (
        f"This looks like a tokenized dataset that needs to be built first.\n"
        f"Run:  python {prep}"
    )
