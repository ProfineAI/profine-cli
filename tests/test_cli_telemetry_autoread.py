"""Tests for the auto-read behavior that fills in an architecture
record when telemetry needs one.

The contract:
  * Telemetry-enabled run, no architecture_record.json on disk →
    cmd_read is invoked to produce one (so emit_run can build a
    fingerprint).
  * Telemetry-enabled run, file already exists → no read attempted.
  * Telemetry-disabled run → no read attempted regardless of whether
    the file exists (we don't want opted-out users paying for the
    LLM call).
  * Reader fails (no API key, network error, …) → the pipeline
    continues; the run just won't contribute telemetry.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from profine.cli.commands import (
    _emit_telemetry_after,
    _ensure_read_output_for_telemetry,
)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def script(tmp_path: Path) -> Path:
    p = tmp_path / "train.py"
    p.write_text("# placeholder\n", encoding="utf-8")
    return p


def _ns(script: Path | None, **overrides) -> Namespace:
    base = dict(
        script=str(script) if script else None,
        provider="openai", api_key=None, model=None, base_url=None,
        seed=None, output="profine_output", prefs=None,
        hardware="1x_a100", no_telemetry=False,
    )
    base.update(overrides)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# _ensure_read_output_for_telemetry behaviour
# ---------------------------------------------------------------------------


class TestEnsureReadOutput:
    def test_noop_when_arch_record_already_exists(self, output_dir, script):
        (output_dir / "read").mkdir()
        (output_dir / "read" / "architecture_record.json").write_text("{}")

        with patch("profine.cli.commands.cmd_read") as read_mock:
            _ensure_read_output_for_telemetry(_ns(script), output_dir)
        read_mock.assert_not_called()

    def test_runs_reader_when_arch_record_missing(self, output_dir, script):
        with patch("profine.cli.commands.cmd_read", return_value=0) as read_mock:
            _ensure_read_output_for_telemetry(_ns(script), output_dir)
        read_mock.assert_called_once()
        # Check that args passed through correctly.
        called_args = read_mock.call_args.args[0]
        assert called_args.script == str(script)
        assert called_args.provider == "openai"

    def test_skips_when_script_missing(self, output_dir):
        """No script attr → emit_run will no-op anyway; don't try to read."""
        with patch("profine.cli.commands.cmd_read") as read_mock:
            _ensure_read_output_for_telemetry(_ns(None), output_dir)
        read_mock.assert_not_called()

    def test_skips_when_script_does_not_exist(self, output_dir, tmp_path):
        """Stale script path on disk → still don't try to read."""
        ghost = tmp_path / "nonexistent.py"
        with patch("profine.cli.commands.cmd_read") as read_mock:
            _ensure_read_output_for_telemetry(_ns(ghost), output_dir)
        read_mock.assert_not_called()

    def test_reader_failure_does_not_raise(self, output_dir, script, capsys):
        """A reader exception must NOT propagate; print one note and continue."""
        with patch("profine.cli.commands.cmd_read",
                   side_effect=RuntimeError("no api key")):
            # Should not raise.
            _ensure_read_output_for_telemetry(_ns(script), output_dir)
        captured = capsys.readouterr()
        assert "reader failed" in captured.out
        assert "RuntimeError" in captured.out


# ---------------------------------------------------------------------------
# Integration: _emit_telemetry_after wires it correctly
# ---------------------------------------------------------------------------


class TestEmitTelemetryAfterAutoRead:
    def test_telemetry_disabled_skips_auto_read(self, output_dir, script,
                                                 monkeypatch):
        """Opted-out user should NEVER incur the LLM cost of cmd_read."""
        monkeypatch.setenv("PROFINE_NO_TELEMETRY", "1")
        called_pipeline = {"count": 0}

        def pipeline():
            called_pipeline["count"] += 1
            return 0

        with patch("profine.cli.commands.cmd_read") as read_mock:
            rc = _emit_telemetry_after(_ns(script), output_dir, pipeline)
        assert rc == 0
        assert called_pipeline["count"] == 1
        read_mock.assert_not_called()

    def test_telemetry_enabled_runs_auto_read(self, output_dir, script,
                                               monkeypatch, tmp_path):
        """OSS consenting user with no arch record → auto-read fires."""
        # Mimic an OSS consent file so the recorder is "enabled".
        monkeypatch.setenv("PROFINE_HOME", str(tmp_path))
        from profine.telemetry.consent import save_consent
        save_consent(True)

        def pipeline():
            return 0

        with patch("profine.cli.commands.cmd_read", return_value=0) as read_mock:
            _emit_telemetry_after(_ns(script), output_dir, pipeline)
        read_mock.assert_called_once()

    def test_pipeline_runs_even_if_auto_read_fails(self, output_dir, script,
                                                    monkeypatch, tmp_path):
        """Telemetry's reader exploding must not block the actual pipeline."""
        monkeypatch.setenv("PROFINE_HOME", str(tmp_path))
        from profine.telemetry.consent import save_consent
        save_consent(True)

        called_pipeline = {"count": 0}

        def pipeline():
            called_pipeline["count"] += 1
            return 7   # non-zero exit to verify it's preserved

        with patch("profine.cli.commands.cmd_read",
                   side_effect=RuntimeError("boom")):
            rc = _emit_telemetry_after(_ns(script), output_dir, pipeline)
        assert called_pipeline["count"] == 1
        assert rc == 7
