"""OSS telemetry consent management.

Consent lives in `~/.profine/telemetry_consent` (XDG-friendly: honors
PROFINE_HOME if set). The file has two states:

  * Missing or `granted: false` → no anonymous telemetry.
  * `granted: true` with an `install_id` → recorder runs in OSS mode.

The first time the OSS profine CLI is invoked without an opted-out
flag, we should prompt the user. The prompt itself is in the CLI
entrypoint; this module is the storage layer plus the policy decision
("should I prompt right now?").

Flag precedence (highest first):
  1. --no-telemetry on the command line  → always off, never prompt.
  2. PROFINE_NO_TELEMETRY=1 in the env    → same as above.
  3. Stored consent file                  → use whatever is in there.
  4. No file yet                          → caller should prompt.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final


_CONSENT_FILENAME: Final[str] = "telemetry_consent.json"
_OPT_OUT_ENV_VARS: Final[tuple[str, ...]] = ("PROFINE_NO_TELEMETRY",)


@dataclass(slots=True, frozen=True)
class ConsentRecord:
    """Persisted user decision."""
    granted: bool
    install_id: str | None  # only set when granted=True


def profine_home() -> Path:
    """Return the directory where profine stores per-user state.

    Override via PROFINE_HOME for tests and unusual setups; otherwise
    defaults to ~/.profine. The directory is created lazily.
    """
    override = os.environ.get("PROFINE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".profine"


def consent_path() -> Path:
    return profine_home() / _CONSENT_FILENAME


def env_opted_out() -> bool:
    """True if any PROFINE_NO_TELEMETRY-style env var is truthy."""
    for var in _OPT_OUT_ENV_VARS:
        val = os.environ.get(var, "").strip().lower()
        if val and val not in ("0", "false", "no", "off"):
            return True
    return False


def load_consent() -> ConsentRecord | None:
    """Return the persisted record or None if the file does not exist
    or is corrupt. The caller treats None as 'should prompt'.
    """
    path = consent_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt file is treated as "no decision yet" rather than
        # erroring out — better UX than refusing to run.
        return None
    granted = bool(data.get("granted"))
    install_id = data.get("install_id") if granted else None
    return ConsentRecord(granted=granted, install_id=install_id)


def save_consent(granted: bool) -> ConsentRecord:
    """Persist the user's choice and return the resulting record.

    Idempotent: granting twice does not regenerate the install_id;
    declining after granting clears it.
    """
    existing = load_consent()
    if granted:
        install_id = (existing.install_id if existing and existing.install_id
                      else str(uuid.uuid4()))
    else:
        install_id = None

    record = ConsentRecord(granted=granted, install_id=install_id)
    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "granted": record.granted,
        "install_id": record.install_id,
    }, indent=2), encoding="utf-8")
    return record


def maybe_prompt_for_consent(
    *,
    api_key: str | None,
    cli_disabled: bool,
    is_interactive: bool,
    prompter=input,
    output=print,
) -> ConsentRecord | None:
    """Ask the user about telemetry on first run, persist the answer.

    Returns the saved record, or None if no prompt was needed (paid
    user, opted out via flag/env, already decided, or non-interactive
    session). The CLI calls this once before recorder construction.

    `prompter` and `output` are injectable for tests so we don't need
    to touch stdin/stdout. Production passes the builtins.
    """
    # Don't prompt when the user has already decided or explicitly
    # disabled, and never prompt paid users (their toggle is the
    # dashboard, not a CLI prompt).
    if cli_disabled or env_opted_out() or api_key is not None:
        return None
    if load_consent() is not None:
        return None
    if not is_interactive:
        # CI, scripts piped via stdin, etc. Default to no-consent so
        # we don't silently turn on data collection.
        return save_consent(False)

    output("")
    output("profine collects anonymous telemetry to learn which optimizations")
    output("work on which architectures. No code, paths, or identifiers — only")
    output("bucketed signals (arch class, hardware class, optimizer family).")
    output("See https://profine.ai/telemetry for the full schema.")
    output("")
    answer = prompter("Send anonymous telemetry? [y/N]: ").strip().lower()
    granted = answer in ("y", "yes")
    record = save_consent(granted)
    if granted:
        output("Thanks! You can change this anytime by deleting "
               f"{consent_path()}.")
    else:
        output("OK, telemetry is off. To enable later: "
               "`profine telemetry enable` or grant manually in the consent file.")
    return record


def resolve_recorder_credentials(
    *,
    cli_disabled: bool,
    api_key: str | None,
) -> tuple[bool, str | None, str | None]:
    """Decide what to pass to TelemetryRecorder, given all inputs.

    Returns (enabled, api_key_for_recorder, install_id_for_recorder).

    Precedence:
      * --no-telemetry / env opt-out wins everywhere.
      * If a paid API key is supplied, use that and ignore the OSS
        consent file (paid users have their own opt-out toggle in the
        dashboard, enforced server-side).
      * Otherwise fall back to the OSS consent record.

    This function never prompts — it only reads existing state. The
    CLI entrypoint is responsible for prompting when this returns
    (False, None, None) because no decision is on file yet.
    """
    if cli_disabled or env_opted_out():
        return (False, None, None)

    if api_key:
        return (True, api_key, None)

    record = load_consent()
    if record is None:
        return (False, None, None)
    if not record.granted or not record.install_id:
        return (False, None, None)
    return (True, None, record.install_id)
