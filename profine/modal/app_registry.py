"""Modal app naming, deployment signatures, and caching.

Deterministic app names and deployment signatures allow warmstart
container reuse — if the signature matches the cached value, we
skip redeployment and reuse the existing Modal app.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from profine.schema.hardware import HardwareConfig

_CACHE_DIR = ".profine"
_CACHE_FILE = "modal_deployments.json"


def resolve_app_name(
    prefix: str,
    project_root: Path,
    script_path: Path,
    hardware: HardwareConfig,
) -> str:
    """Build a deterministic Modal app name.

    Format: {prefix}-{project}-{script}-{hardware}-{hash8}
    """
    project = project_root.name.lower()[:20]
    script = script_path.stem.lower()[:20]
    hw = hardware.name.lower()

    # Hash for uniqueness
    content = f"{project_root}:{script_path}:{hardware.name}"
    sig = hashlib.sha256(content.encode()).hexdigest()[:8]

    name = f"{prefix}-{project}-{script}-{hw}-{sig}"
    # Modal app names: alphanumeric + hyphens, max 64 chars
    name = "".join(c if c.isalnum() or c == "-" else "-" for c in name)
    return name[:64]


def build_deployment_signature(
    *,
    project_root: Path,
    dependencies: list[str],
    python_version: str,
    hardware: HardwareConfig,
    profine_source_hash: str = "",
) -> str:
    """SHA256 signature of all inputs that affect the Modal image.

    If this changes, the app must be redeployed.
    """
    parts = [
        str(project_root),
        json.dumps(sorted(dependencies)),
        python_version,
        hardware.name,
        hardware.modal_gpu,
        profine_source_hash,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def load_deployment_cache(project_root: Path) -> dict[str, str]:
    """Load cached deployment signatures. Returns {app_name: signature}."""
    cache_path = project_root / _CACHE_DIR / _CACHE_FILE
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_deployment_cache(
    project_root: Path,
    app_name: str,
    signature: str,
) -> None:
    """Cache a deployment signature for later reuse checks."""
    cache_dir = project_root / _CACHE_DIR
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / _CACHE_FILE

    cache = load_deployment_cache(project_root)
    cache[app_name] = signature

    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def invalidate_deployment_cache(project_root: Path) -> None:
    """Remove all cached deployment signatures."""
    cache_path = project_root / _CACHE_DIR / _CACHE_FILE
    if cache_path.exists():
        cache_path.unlink()


def signature_matches(
    project_root: Path,
    app_name: str,
    current_signature: str,
) -> bool:
    """Check if the cached signature for an app matches the current one."""
    cache = load_deployment_cache(project_root)
    return cache.get(app_name) == current_signature
