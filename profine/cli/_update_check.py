"""Once-a-day nudge when a newer profine is available on PyPI.

Design constraints:
  * Must NEVER block a CLI command for more than ~1s, even if PyPI is slow.
  * Must NEVER raise — a failed update check is silent.
  * Must be cheap: once per 24h, cached on disk.
  * Must respect `PROFINE_NO_UPDATE_CHECK=1` and non-interactive shells (CI).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_PYPI_URL = "https://pypi.org/pypi/profine/json"
_CACHE_FILENAME = "last_update_check.json"
_CHECK_INTERVAL_SECONDS = 24 * 60 * 60  # 1 day
_HTTP_TIMEOUT_SECONDS = 1.0


def _cache_path() -> Path:
    """Cache lives next to auth.json / telemetry_consent.json."""
    home = os.environ.get("PROFINE_HOME")
    base = Path(home) if home else Path.home() / ".profine"
    return base / _CACHE_FILENAME


def _installed_version() -> str | None:
    """Read the running version from package metadata. None if not installed."""
    try:
        from importlib.metadata import version as _v  # py3.8+
        return _v("profine")
    except Exception:
        return None


def _parse(v: str) -> tuple[int, ...]:
    """Loose semver-ish parse: '0.5.0' -> (0, 5, 0). Non-numeric → (0,)."""
    out: list[int] = []
    for part in v.split("."):
        # Strip pre-release suffix ("0.5.0rc1" → "0")
        n = ""
        for ch in part:
            if ch.isdigit():
                n += ch
            else:
                break
        if n:
            out.append(int(n))
    return tuple(out) or (0,)


def _is_newer(remote: str, local: str) -> bool:
    return _parse(remote) > _parse(local)


def _read_cache() -> dict | None:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(latest: str) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"checked_at": time.time(), "latest": latest}),
            encoding="utf-8",
        )
    except Exception:
        pass  # cache write failure is non-fatal


def _fetch_latest() -> str | None:
    """Query PyPI; return latest version string or None on any failure."""
    from urllib.request import urlopen
    try:
        with urlopen(_PYPI_URL, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("info", {}).get("version")
    except Exception:
        return None


def maybe_print_update_nudge(*, stream=sys.stderr) -> None:
    """Run the check (cached, opt-out-aware) and print a nudge if newer.

    Called on CLI startup. Never raises. Adds at most ~1s on the once-a-day
    invocation that actually hits PyPI; subsequent invocations within 24h
    only touch the disk cache.
    """
    try:
        if os.environ.get("PROFINE_NO_UPDATE_CHECK") == "1":
            return
        # CI and piped invocations should never see the nudge.
        if not (stream.isatty() if hasattr(stream, "isatty") else False):
            return

        installed = _installed_version()
        if not installed:
            return  # editable / source install — no PyPI comparison meaningful

        cache = _read_cache() or {}
        last = float(cache.get("checked_at") or 0.0)
        cached_latest = cache.get("latest")

        if time.time() - last < _CHECK_INTERVAL_SECONDS:
            latest = cached_latest
        else:
            latest = _fetch_latest()
            if latest:
                _write_cache(latest)

        if not latest or not _is_newer(latest, installed):
            return

        # One-line nudge, dim-coloured if rich is available.
        msg = (
            f"profine {latest} is available "
            f"(you have {installed}). Upgrade: pip install -U profine"
        )
        try:
            from rich.console import Console
            Console(file=stream, soft_wrap=True).print(f"[dim]{msg}[/dim]")
        except Exception:
            print(msg, file=stream)
    except Exception:
        return  # never crash the CLI over a version check
