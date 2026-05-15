#!/usr/bin/env python3
"""Regenerate profine/telemetry/_integrity.py's MANIFEST.

Run this whenever any file in profine/telemetry/ changes. Commit the
updated _integrity.py alongside the change. CI's sync check will
fail if you don't.

Usage:
    python3 scripts/rebuild_telemetry_manifest.py             # rewrite in place
    python3 scripts/rebuild_telemetry_manifest.py --check     # exit 1 if stale
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_DIR = REPO_ROOT / "profine" / "telemetry"
INTEGRITY_FILE = TELEMETRY_DIR / "_integrity.py"

BEGIN_MARKER = "# ---- BEGIN GENERATED MANIFEST"
END_MARKER = "# ---- END GENERATED MANIFEST"


def compute_manifest() -> dict[str, str]:
    """sha256 every .py in profine/telemetry except _integrity.py itself."""
    out: dict[str, str] = {}
    for path in sorted(TELEMETRY_DIR.iterdir()):
        if not path.is_file() or path.suffix != ".py":
            continue
        if path.name == "_integrity.py":
            continue
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def render_manifest_block(manifest: dict[str, str]) -> str:
    """Render the MANIFEST literal exactly as it should appear in the file."""
    lines = [
        BEGIN_MARKER + " (DO NOT EDIT BY HAND) ----",
        "# Regenerate with: python3 scripts/rebuild_telemetry_manifest.py",
        "# Each entry: filename (relative to profine/telemetry/) → sha256 of file bytes",
        "MANIFEST: Final[dict[str, str]] = {",
    ]
    for name, sha in manifest.items():
        lines.append(f'    "{name}": "{sha}",')
    lines.append("}")
    lines.append(END_MARKER + " ----")
    return "\n".join(lines)


_BLOCK_RE = re.compile(
    rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}[^\n]*",
    re.DOTALL,
)


def rewrite_integrity_file(manifest: dict[str, str]) -> str:
    """Splice the rendered block into _integrity.py's marker region.
    Returns the new file contents (does not write)."""
    current = INTEGRITY_FILE.read_text(encoding="utf-8")
    new_block = render_manifest_block(manifest)
    if not _BLOCK_RE.search(current):
        raise RuntimeError("BEGIN/END markers not found in _integrity.py")
    return _BLOCK_RE.sub(new_block, current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if the manifest is stale (used in CI). "
                             "Doesn't modify files.")
    args = parser.parse_args()

    manifest = compute_manifest()
    new_text = rewrite_integrity_file(manifest)
    current = INTEGRITY_FILE.read_text(encoding="utf-8")

    if new_text == current:
        print("manifest up to date — no changes")
        return 0

    if args.check:
        print("ERROR: profine/telemetry/_integrity.py is stale.", file=sys.stderr)
        print("       Run: python3 scripts/rebuild_telemetry_manifest.py", file=sys.stderr)
        return 1

    INTEGRITY_FILE.write_text(new_text, encoding="utf-8")
    print(f"rewrote {INTEGRITY_FILE.relative_to(REPO_ROOT)}")
    print(f"  {len(manifest)} files hashed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
