"""Tests for the telemetry-module tamper-evidence layer.

We deliberately don't test "tampering refuses to run" — the system
is intentionally tamper-EVIDENT (warn) rather than tamper-PROOF
(refuse). The contract is:

  * Untampered → INTEGRITY_OK is True, no warning emitted.
  * Tampered  → INTEGRITY_OK is False, one warning block written
    to stderr, but the import succeeds.
  * Empty manifest (unbuilt source) → check is skipped (treat as OK).
  * Env var override → check is skipped.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from profine.telemetry import _integrity


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("PROFINE_DISABLE_INTEGRITY_CHECK", raising=False)
    yield


def test_baked_manifest_is_non_empty():
    """A built distribution must include the manifest. If this fails,
    rebuild it: `python3 scripts/rebuild_telemetry_manifest.py`."""
    assert _integrity.MANIFEST, "manifest empty — regenerate before shipping"


def test_unmodified_install_passes():
    """On the canonical source tree, integrity must be OK."""
    mismatches = _integrity.verify_integrity()
    assert mismatches == [], f"unexpected mismatches: {mismatches}"
    assert _integrity.INTEGRITY_OK is True


def test_empty_manifest_skips_check():
    """Dev builds without a baked manifest produce an empty list."""
    with patch.object(_integrity, "MANIFEST", {}):
        assert _integrity.verify_integrity() == []


def test_env_var_disables_check(monkeypatch):
    monkeypatch.setenv("PROFINE_DISABLE_INTEGRITY_CHECK", "1")
    # Construct a synthetic manifest that would fail; the env var
    # should short-circuit before we hash anything.
    with patch.object(_integrity, "MANIFEST", {"fields.py": "deadbeef"}):
        assert _integrity.verify_integrity() == []


@pytest.mark.parametrize("falsey", ["", "0", "false", "no", "off"])
def test_falsey_env_values_do_not_disable(monkeypatch, falsey):
    monkeypatch.setenv("PROFINE_DISABLE_INTEGRITY_CHECK", falsey)
    with patch.object(_integrity, "MANIFEST", {"fields.py": "deadbeef"}):
        # Falsey value → env override does NOT apply → check runs.
        # We expect a MODIFIED mismatch (the synthetic hash won't match).
        mismatches = _integrity.verify_integrity()
        assert any("MODIFIED" in m for m in mismatches)


def test_detects_modified_file():
    """Hash one of the real files at runtime; manifest expects the real
    hash; we override the manifest entry with a fake one; verifier
    must report MODIFIED."""
    real = dict(_integrity.MANIFEST)
    real["fields.py"] = "0" * 64  # pretend manifest says this
    with patch.object(_integrity, "MANIFEST", real):
        mismatches = _integrity.verify_integrity()
        assert any("MODIFIED" in m and "fields.py" in m for m in mismatches)


def test_detects_missing_file(tmp_path, monkeypatch):
    """If a declared file isn't on disk, verifier reports MISSING."""
    fake_manifest = {"nonexistent_file.py": "0" * 64}
    with patch.object(_integrity, "MANIFEST", fake_manifest):
        mismatches = _integrity.verify_integrity()
        assert any("MISSING" in m and "nonexistent_file.py" in m for m in mismatches)


def test_detects_extra_files(tmp_path, monkeypatch):
    """A file present on disk but not in manifest is flagged as EXTRA."""
    # Use a manifest containing only one real file. The other real files
    # then look like EXTRA additions.
    fake = {"__init__.py": _integrity.MANIFEST["__init__.py"]}
    with patch.object(_integrity, "MANIFEST", fake):
        mismatches = _integrity.verify_integrity()
        assert any("EXTRA" in m for m in mismatches)


def test_warning_format_is_easy_to_grep(capsys):
    """The user-visible warning must identify itself clearly so
    nobody mistakes it for normal log noise."""
    with patch.object(_integrity, "MANIFEST", {"fields.py": "0" * 64}):
        mismatches = _integrity.verify_integrity()
        _integrity._emit_warning(mismatches)
    err = capsys.readouterr().err
    assert "integrity check failed" in err
    assert "modified version of profine.telemetry" in err


def test_manifest_is_in_sync_with_source():
    """If you change a telemetry file without regenerating the manifest,
    this fails. CI gate."""
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/rebuild_telemetry_manifest.py", "--check"],
        cwd=repo_root, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "manifest is stale — run "
        "`python3 scripts/rebuild_telemetry_manifest.py`"
        f"\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
