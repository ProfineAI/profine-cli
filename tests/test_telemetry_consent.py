"""Tests for OSS telemetry consent storage and credential resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from profine.telemetry.consent import (
    ConsentRecord,
    consent_path,
    env_opted_out,
    load_consent,
    maybe_prompt_for_consent,
    profine_home,
    resolve_recorder_credentials,
    save_consent,
)


@pytest.fixture(autouse=True)
def isolated_consent_dir(tmp_path: Path, monkeypatch):
    """Redirect ~/.profine to a tempdir for every test in this file."""
    monkeypatch.setenv("PROFINE_HOME", str(tmp_path))
    # And clear any opt-out env vars so tests aren't poisoned.
    for var in ("PROFINE_NO_TELEMETRY",):
        monkeypatch.delenv(var, raising=False)
    yield


# ----------------------------- storage layer ------------------------------


def test_profine_home_respects_env_override(tmp_path):
    assert profine_home() == tmp_path


def test_load_consent_returns_none_when_no_file():
    assert load_consent() is None


def test_load_consent_returns_none_on_corrupt_file(tmp_path):
    consent_path().parent.mkdir(parents=True, exist_ok=True)
    consent_path().write_text("not json{{", encoding="utf-8")
    assert load_consent() is None


def test_save_consent_granted_writes_install_id():
    record = save_consent(True)
    assert record.granted is True
    assert record.install_id is not None
    # Persisted to disk:
    on_disk = json.loads(consent_path().read_text(encoding="utf-8"))
    assert on_disk["granted"] is True
    assert on_disk["install_id"] == record.install_id


def test_save_consent_declined_clears_install_id():
    save_consent(True)
    record = save_consent(False)
    assert record.granted is False
    assert record.install_id is None
    on_disk = json.loads(consent_path().read_text(encoding="utf-8"))
    assert on_disk["install_id"] is None


def test_save_consent_granting_twice_keeps_same_install_id():
    first = save_consent(True)
    second = save_consent(True)
    assert first.install_id == second.install_id


def test_load_consent_round_trips_grant():
    saved = save_consent(True)
    loaded = load_consent()
    assert loaded == saved


# ----------------------------- env opt-out --------------------------------


def test_env_opted_out_default_false():
    assert env_opted_out() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_env_opted_out_truthy(monkeypatch, val):
    monkeypatch.setenv("PROFINE_NO_TELEMETRY", val)
    assert env_opted_out() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
def test_env_opted_out_falsey(monkeypatch, val):
    monkeypatch.setenv("PROFINE_NO_TELEMETRY", val)
    assert env_opted_out() is False


# ----------------------------- credential resolution ---------------------


class TestResolveCredentials:
    def test_cli_flag_disables_everything(self):
        save_consent(True)  # even with stored consent
        enabled, api_key, install_id = resolve_recorder_credentials(
            cli_disabled=True, api_key="pf_live_x",
        )
        assert (enabled, api_key, install_id) == (False, None, None)

    def test_env_opt_out_disables(self, monkeypatch):
        monkeypatch.setenv("PROFINE_NO_TELEMETRY", "1")
        save_consent(True)
        enabled, api_key, install_id = resolve_recorder_credentials(
            cli_disabled=False, api_key="pf_live_x",
        )
        assert (enabled, api_key, install_id) == (False, None, None)

    def test_paid_api_key_wins_over_oss_consent(self):
        save_consent(True)  # OSS consent exists
        enabled, api_key, install_id = resolve_recorder_credentials(
            cli_disabled=False, api_key="pf_live_x",
        )
        assert enabled is True
        assert api_key == "pf_live_x"
        assert install_id is None  # OSS path not used

    def test_oss_consent_used_when_no_api_key(self):
        record = save_consent(True)
        enabled, api_key, install_id = resolve_recorder_credentials(
            cli_disabled=False, api_key=None,
        )
        assert enabled is True
        assert api_key is None
        assert install_id == record.install_id

    def test_no_decision_means_disabled(self):
        enabled, api_key, install_id = resolve_recorder_credentials(
            cli_disabled=False, api_key=None,
        )
        assert (enabled, api_key, install_id) == (False, None, None)

    def test_decline_consent_means_disabled(self):
        save_consent(False)
        enabled, api_key, install_id = resolve_recorder_credentials(
            cli_disabled=False, api_key=None,
        )
        assert (enabled, api_key, install_id) == (False, None, None)


# ----------------------------- prompt flow --------------------------------


def _mute_output(_msg=""):
    """No-op replacement for print in tests so we don't spam capsys."""


class TestMaybePromptForConsent:
    def test_skips_when_cli_disabled(self):
        assert maybe_prompt_for_consent(
            api_key=None, cli_disabled=True, is_interactive=True,
            prompter=lambda _: pytest.fail("should not prompt"),
            output=_mute_output,
        ) is None

    def test_skips_when_env_opted_out(self, monkeypatch):
        monkeypatch.setenv("PROFINE_NO_TELEMETRY", "1")
        assert maybe_prompt_for_consent(
            api_key=None, cli_disabled=False, is_interactive=True,
            prompter=lambda _: pytest.fail("should not prompt"),
            output=_mute_output,
        ) is None

    def test_skips_paid_users(self):
        assert maybe_prompt_for_consent(
            api_key="pf_live_xxx", cli_disabled=False, is_interactive=True,
            prompter=lambda _: pytest.fail("should not prompt"),
            output=_mute_output,
        ) is None

    def test_skips_when_consent_already_stored(self):
        save_consent(True)
        assert maybe_prompt_for_consent(
            api_key=None, cli_disabled=False, is_interactive=True,
            prompter=lambda _: pytest.fail("should not prompt"),
            output=_mute_output,
        ) is None

    def test_noninteractive_defaults_to_decline(self):
        record = maybe_prompt_for_consent(
            api_key=None, cli_disabled=False, is_interactive=False,
            prompter=lambda _: pytest.fail("should not prompt"),
            output=_mute_output,
        )
        assert record is not None
        assert record.granted is False
        assert record.install_id is None

    @pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES "])
    def test_interactive_accept(self, answer):
        record = maybe_prompt_for_consent(
            api_key=None, cli_disabled=False, is_interactive=True,
            prompter=lambda _: answer,
            output=_mute_output,
        )
        assert record is not None
        assert record.granted is True
        assert record.install_id is not None

    @pytest.mark.parametrize("answer", ["", "n", "no", "garbage"])
    def test_interactive_decline(self, answer):
        record = maybe_prompt_for_consent(
            api_key=None, cli_disabled=False, is_interactive=True,
            prompter=lambda _: answer,
            output=_mute_output,
        )
        assert record is not None
        assert record.granted is False
        assert record.install_id is None
