"""Factory that turns CLI args + env + consent into a TelemetryRecorder.

One place, callable from any subcommand handler, so we don't repeat
the credential-resolution dance every time we instrument a new
command. Returns a recorder that's either fully wired (paid / OSS
consenting) or silently disabled.

Endpoint URL is sourced from `PROFINE_API_URL` (default
`https://api.profine.ai`). Paid bearer key is sourced from
`PROFINE_API_KEY` — the customer's own profine API key.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace

from profine.telemetry.consent import (
    maybe_prompt_for_consent,
    resolve_recorder_credentials,
)
from profine.telemetry.recorder import TelemetryRecorder


_DEFAULT_API_URL = "https://api.profine.ai"


def build_recorder(args: Namespace, *, client_version: str) -> TelemetryRecorder:
    """Construct (or no-op) a TelemetryRecorder for this invocation.

    Side effect: on first OSS run we prompt for consent if the
    session is interactive. The prompt is printed to stdout/stderr
    and reads from stdin; tests can stub by setting
    PROFINE_NO_TELEMETRY=1.
    """
    api_url = os.environ.get("PROFINE_API_URL", _DEFAULT_API_URL)
    customer_api_key = os.environ.get("PROFINE_API_KEY")
    cli_disabled = bool(getattr(args, "no_telemetry", False))

    # Prompt only on first OSS run with no decision on file. Safe in
    # all other cases (cli-off, env-off, paid, already decided).
    maybe_prompt_for_consent(
        api_key=customer_api_key,
        cli_disabled=cli_disabled,
        is_interactive=sys.stdin.isatty() and sys.stdout.isatty(),
    )

    enabled, api_key, install_id = resolve_recorder_credentials(
        cli_disabled=cli_disabled,
        api_key=customer_api_key,
    )

    return TelemetryRecorder(
        api_url=api_url,
        api_key=api_key,
        install_id=install_id,
        enabled=enabled,
        client_version=client_version,
    )


__all__ = ["build_recorder"]
