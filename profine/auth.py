"""Saved-credentials store for profine.

Lets users paste API keys once (`profine auth login`) instead of
exporting env vars in every shell. Keys live in `~/.profine/auth.json`
(honors PROFINE_HOME) with 0600 perms.

Precedence on read: the process environment always wins. `apply_to_env`
fills in any unset vars from the saved file, so CI and one-off
`KEY=... profine ...` invocations are never silently overridden.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Final

from profine.telemetry.consent import profine_home


_AUTH_FILENAME: Final[str] = "auth.json"

# The set of credentials `profine auth` knows how to manage. Keep in
# sync with env_vars.REGISTRY — anything secret-y that the CLI reads.
MANAGED_KEYS: Final[tuple[str, ...]] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PROFINE_API_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "HF_TOKEN",
)


def auth_path() -> Path:
    return profine_home() / _AUTH_FILENAME


def _load_raw() -> dict[str, str]:
    path = auth_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Corrupt file = treat as empty rather than refusing to run.
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def load() -> dict[str, str]:
    """Return the saved credentials, filtered to MANAGED_KEYS."""
    raw = _load_raw()
    return {k: v for k, v in raw.items() if k in MANAGED_KEYS}


def _write(data: dict[str, str]) -> None:
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    # Best-effort tighten perms — no-op on Windows.
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def save_key(name: str, value: str) -> None:
    """Persist one credential. Unknown names are rejected."""
    if name not in MANAGED_KEYS:
        raise ValueError(f"Unknown credential: {name}. Known: {', '.join(MANAGED_KEYS)}")
    value = value.strip()
    if not value:
        raise ValueError("Empty value")
    data = _load_raw()
    data[name] = value
    _write(data)


def clear_key(name: str) -> bool:
    """Remove one credential. Returns True if removed, False if not present."""
    data = _load_raw()
    if name not in data:
        return False
    del data[name]
    _write(data)
    return True


def clear_all() -> bool:
    """Delete the entire auth file. Returns True if a file existed."""
    path = auth_path()
    if not path.exists():
        return False
    path.unlink()
    return True


def apply_to_env(environ: dict[str, str] | None = None) -> list[str]:
    """Fill in missing env vars from the saved file.

    Returns the list of names that were applied (i.e. were unset and
    are now set). Existing env vars are never overwritten — the
    environment is the source of truth on conflict, so CI and one-off
    `KEY=... profine ...` invocations win.
    """
    env = os.environ if environ is None else environ
    applied: list[str] = []
    for name, value in load().items():
        if not env.get(name):
            env[name] = value
            applied.append(name)
    return applied


def redact(value: str | None) -> str:
    """Show only first/last 4 chars; for safe display in `auth status`."""
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


__all__ = [
    "MANAGED_KEYS",
    "auth_path",
    "load",
    "save_key",
    "clear_key",
    "clear_all",
    "apply_to_env",
    "redact",
]
