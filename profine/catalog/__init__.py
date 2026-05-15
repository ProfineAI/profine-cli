"""Optimization catalog package.

Exposes `CATALOG_VERSION`, a stable 16-char hash of `catalog.yaml`.
Telemetry rows are partitioned by this version so priors computed
against one catalog can never silently apply after a rename or
restructure. Bumping is automatic — any YAML edit changes the hash.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


def _config_path() -> Path:
    # profine/catalog/__init__.py  →  profine/config/catalog.yaml
    return Path(__file__).resolve().parent.parent / "config" / "catalog.yaml"


@lru_cache(maxsize=1)
def catalog_version() -> str:
    """Stable 16-char hex digest of the catalog YAML.

    Cached for the process lifetime; tests that mutate the file
    should clear the lru_cache.
    """
    try:
        blob = _config_path().read_bytes()
    except OSError:
        # Unusual install layouts (zipapps, frozen builds) — return a
        # marker rather than crashing the recorder.
        return "unknown"
    return hashlib.sha256(blob).hexdigest()[:16]


CATALOG_VERSION: str = catalog_version()


__all__ = ["CATALOG_VERSION", "catalog_version"]
