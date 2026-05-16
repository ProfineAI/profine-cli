"""Client-side telemetry recorder.

Collects the fingerprint and per-optimization outcomes during a run,
then ships the batch to the backend via a best-effort async POST.

Design rules (these are why the surface is small):

  * Never block the caller. The actual HTTP call runs on a daemon
    thread; flush() returns immediately.
  * Never raise from a recording method. A telemetry bug must never
    take down a customer's run.
  * Never send a field not on the allowlist (see fields.py). The
    recorder filters the payload before serialization.
  * Disabled mode is a no-op. `enabled=False` makes every method a
    no-op so callers don't need to branch.
  * Idempotent close(). Safe to call multiple times.

Two modes are supported via the constructor:

  * Paid: `api_key="pf_live_..."` — POST to /api/telemetry/run with
    bearer auth. Backend joins to accounts via the key.
  * OSS:  `install_id="<uuid>"`   — POST to /api/telemetry/anon with
    no auth. The install_id is the stable per-machine consent token
    written to ~/.profine/telemetry_consent on first opt-in.

The two modes are exclusive; specifying both is a configuration
error and we raise at construction time (the only place we raise).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from profine.telemetry.fields import (
    filter_fingerprint,
    filter_outcome,
    filter_profile_stats,
)
from profine.telemetry.fingerprint import Fingerprint


# 5 seconds is generous for a small JSON POST. If the backend is
# overloaded, the recorder dies quietly and the run continues.
_HTTP_TIMEOUT_SECONDS: float = 5.0


log = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingOutcome:
    """Outcome row queued for the next flush."""
    optimization_id: str
    catalog_version: str
    applied: bool
    speedup_factor: float | None = None
    loss_ok: bool | None = None
    crashed: bool = False
    crash_class: str | None = None
    runtime_seconds: float | None = None


class TelemetryRecorder:
    """Stateful collector for one run's telemetry.

    Usage:
        recorder = TelemetryRecorder(api_url="https://api.profine.ai",
                                     api_key="pf_live_...",
                                     client_version="0.3.5")
        recorder.begin_run(fingerprint)
        recorder.record_optimization("compile_default", "v1", applied=True,
                                     speedup_factor=1.42, loss_ok=True)
        recorder.close()  # sends in background; never blocks
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        install_id: str | None = None,
        enabled: bool = True,
        client_version: str = "",
    ) -> None:
        if api_key and install_id:
            raise ValueError(
                "TelemetryRecorder: provide api_key (paid) OR install_id (OSS), not both"
            )
        # When the user supplies neither credential, recording silently
        # disables. This is the common case for `profine` invoked with
        # no telemetry config (e.g. CI smoke tests, local debugging).
        if api_key is None and install_id is None:
            enabled = False

        self._api_url = (api_url or "").rstrip("/")
        self._api_key = api_key
        self._install_id = install_id
        self._enabled = enabled and bool(self._api_url)
        self._client_version = client_version

        self._fingerprint: Fingerprint | None = None
        self._outcomes: list[_PendingOutcome] = []
        self._profile_stats: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._closed = False
        # Track in-flight flush threads so close() can join them before
        # the process exits. Daemon threads die on interpreter shutdown,
        # which raced the HTTP POST in earlier behavior.
        self._flush_threads: list[threading.Thread] = []

    # ----- public API ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def begin_run(self, fingerprint: Fingerprint) -> None:
        """Record the fingerprint for the current run.

        Idempotent: calling twice replaces the fingerprint, which lets
        a caller update it once the LLM analyzer refines the initial
        AST-only guess.
        """
        if not self._enabled:
            return
        with self._lock:
            self._fingerprint = fingerprint

    def record_profile_stats(self, stats: dict[str, Any]) -> None:
        """Attach per-run profile statistics to the pending batch.

        Pass a dict with any subset of `ALLOWED_PROFILE_STATS_FIELDS`;
        unknown keys are silently dropped at flush time by the allowlist
        filter. Calling twice replaces the prior stats (we ship at
        most one stats block per run).
        """
        if not self._enabled or not stats:
            return
        # Filter at attach-time too so a buggy caller sees an empty
        # store rather than a partially-shipped payload.
        cleaned = filter_profile_stats(stats)
        if not cleaned:
            return
        with self._lock:
            self._profile_stats = cleaned

    def record_optimization(
        self,
        optimization_id: str,
        *,
        catalog_version: str,
        applied: bool,
        speedup_factor: float | None = None,
        loss_ok: bool | None = None,
        crashed: bool = False,
        crash_class: str | None = None,
        runtime_seconds: float | None = None,
    ) -> None:
        """Append one optimization outcome to the pending batch.

        Validates the crash invariant locally (crashed=True iff
        crash_class is non-None) so junk rows don't reach the backend.
        """
        if not self._enabled:
            return
        if crashed and crash_class is None:
            crash_class = "other"
        if not crashed and crash_class is not None:
            # Reject the conflicting state quietly — better to drop
            # than to ship a row the DB constraint will reject.
            log.debug("telemetry: ignoring outcome with crash_class but crashed=False")
            return
        with self._lock:
            self._outcomes.append(_PendingOutcome(
                optimization_id=optimization_id,
                catalog_version=catalog_version,
                applied=applied,
                speedup_factor=speedup_factor,
                loss_ok=loss_ok,
                crashed=crashed,
                crash_class=crash_class,
                runtime_seconds=runtime_seconds,
            ))

    def flush(self) -> None:
        """Send the accumulated batch to the backend in a daemon thread.

        Returns immediately; does not block on the network. The thread
        handle is kept on self._flush_threads so `close()` can wait
        briefly for in-flight POSTs to finish before the main process
        exits (daemon threads die on exit, so without the join the
        POST is racing the interpreter shutdown).
        """
        if not self._enabled:
            return
        payload = self._drain_payload()
        if payload is None:
            return
        thread = threading.Thread(
            target=self._post_with_logging,
            args=(payload,),
            name="profine-telemetry-flush",
            daemon=True,
        )
        thread.start()
        with self._lock:
            self._flush_threads.append(thread)

    def close(self) -> None:
        """Final flush, then wait briefly for in-flight POSTs.

        Idempotent. Joins each flush thread with a per-thread timeout
        so a slow backend can't hang the CLI on exit; defaults to
        `_HTTP_TIMEOUT_SECONDS + 1` per outstanding flush.
        """
        if self._closed:
            return
        self._closed = True
        self.flush()
        # Snapshot so we can release the lock while joining (joins block).
        with self._lock:
            threads = list(self._flush_threads)
        join_timeout = _HTTP_TIMEOUT_SECONDS + 1.0
        for t in threads:
            t.join(timeout=join_timeout)

    # ----- internal ------------------------------------------------------

    def _drain_payload(self) -> dict[str, Any] | None:
        """Take a snapshot of pending state, return a JSON-safe dict.

        Returns None when there is nothing meaningful to send: no
        fingerprint, OR no outcomes AND no profile_stats. The
        fingerprint is always required (the backend uses it to attach
        every row to a run); but a batch with only profile_stats is
        valid, and so is a batch with only outcomes.
        """
        with self._lock:
            if self._fingerprint is None:
                return None
            if not self._outcomes and self._profile_stats is None:
                return None
            fingerprint = self._fingerprint
            outcomes = self._outcomes
            profile_stats = self._profile_stats
            self._outcomes = []
            self._profile_stats = None

        payload: dict[str, Any] = {
            "client_version": self._client_version,
            "install_id": self._install_id,
            "fingerprint": filter_fingerprint(fingerprint.as_dict()),
            "outcomes": [filter_outcome(asdict(o)) for o in outcomes],
        }
        if profile_stats is not None:
            payload["profile_stats"] = filter_profile_stats(profile_stats)
        return payload

    def _endpoint_for_mode(self) -> str:
        """Pick /run for paid, /anon for OSS."""
        path = "/api/telemetry/run" if self._api_key else "/api/telemetry/anon"
        return f"{self._api_url}{path}"

    def _post_with_logging(self, payload: dict[str, Any]) -> None:
        """Wrap _post() so daemon-thread exceptions get logged, not lost."""
        try:
            self._post(payload)
        except Exception:
            log.debug("telemetry POST failed", exc_info=True)

    def _post(self, payload: dict[str, Any]) -> None:
        """Synchronous POST. Always invoked on a background thread."""
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            self._endpoint_for_mode(),
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
                # Drain the response body so the kernel doesn't keep
                # the socket in CLOSE_WAIT.
                resp.read(4096)
        except URLError as e:
            log.debug("telemetry endpoint unreachable: %s", e)
