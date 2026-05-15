"""Tests for `profine telemetry {status|enable|disable}`."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from profine.cli.commands import cmd_telemetry
from profine.telemetry.consent import consent_path, load_consent, save_consent


@pytest.fixture(autouse=True)
def isolated_consent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PROFINE_HOME", str(tmp_path))
    for var in ("PROFINE_NO_TELEMETRY", "PROFINE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _ns(action: str) -> Namespace:
    return Namespace(action=action)


# ----------------------------- status -------------------------------------


def test_status_shows_undecided_when_no_file(capsys, tmp_path):
    rc = cmd_telemetry(_ns("status"), tmp_path, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert "not yet decided" in out
    assert "interactive" in out


def test_status_shows_granted_state(capsys, tmp_path):
    record = save_consent(True)
    rc = cmd_telemetry(_ns("status"), tmp_path, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert "GRANTED" in out
    assert record.install_id in out


def test_status_shows_declined_state(capsys, tmp_path):
    save_consent(False)
    rc = cmd_telemetry(_ns("status"), tmp_path, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert "DECLINED" in out


def test_status_flags_env_override(capsys, tmp_path, monkeypatch):
    save_consent(True)
    monkeypatch.setenv("PROFINE_NO_TELEMETRY", "1")
    rc = cmd_telemetry(_ns("status"), tmp_path, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert "PROFINE_NO_TELEMETRY" in out


def test_status_flags_paid_api_key(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("PROFINE_API_KEY", "pf_live_xxx")
    rc = cmd_telemetry(_ns("status"), tmp_path, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert "PROFINE_API_KEY" in out


# ----------------------------- enable / disable ---------------------------


def test_enable_writes_consent(capsys, tmp_path):
    rc = cmd_telemetry(_ns("enable"), tmp_path, None)
    out = capsys.readouterr().out
    assert rc == 0
    record = load_consent()
    assert record is not None
    assert record.granted is True
    assert record.install_id is not None
    assert record.install_id in out  # the id is shown so the user can copy it


def test_disable_clears_consent(capsys, tmp_path):
    save_consent(True)
    rc = cmd_telemetry(_ns("disable"), tmp_path, None)
    assert rc == 0
    record = load_consent()
    assert record is not None
    assert record.granted is False
    assert record.install_id is None


def test_enable_then_disable_then_enable_keeps_history(tmp_path):
    """Enabling regenerates install_id only after disable cleared it.

    This documents an intentional UX: disable clears the id; re-enable
    starts fresh. Users who disable for privacy reasons don't want
    their old install_id resurrected behind their back.
    """
    cmd_telemetry(_ns("enable"), tmp_path, None)
    first_id = load_consent().install_id
    cmd_telemetry(_ns("disable"), tmp_path, None)
    cmd_telemetry(_ns("enable"), tmp_path, None)
    second_id = load_consent().install_id
    assert first_id is not None and second_id is not None
    assert first_id != second_id


def test_status_does_not_change_state(tmp_path):
    save_consent(True)
    cmd_telemetry(_ns("status"), tmp_path, None)
    assert load_consent().granted is True
