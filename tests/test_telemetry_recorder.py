"""Tests for the client-side TelemetryRecorder.

We never hit a real network in tests. The recorder's HTTP call is
patched at the urlopen seam so we can inspect what would have been
sent and confirm fail-silent behavior.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from profine.telemetry.fingerprint import Fingerprint
from profine.telemetry.recorder import TelemetryRecorder


# ----------------------------- helpers ------------------------------------


def _fp() -> Fingerprint:
    return Fingerprint(
        arch_class="transformer-decoder",
        param_bucket="100M-1B",
        hardware_class="1x_a100",
        precision="mixed_bf16",
        optimizer_class="adam_family",
        has_compile=True,
        has_distributed=False,
        fingerprint_hash="abc123" * 10,
        compile_mode="default",
        attention_impl="manual",
        framework="raw_pytorch",
    )


def _wait_for_background_threads(timeout: float = 1.0) -> None:
    """Wait for any profine-telemetry-flush daemons to finish.

    The recorder fires-and-forgets; tests inspecting captured calls
    need to wait for the daemon to actually run.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [t for t in threading.enumerate() if t.name == "profine-telemetry-flush" and t.is_alive()]
        if not alive:
            return
        time.sleep(0.01)


# ----------------------------- construction --------------------------------


def test_recorder_disabled_when_no_credentials():
    """No api_key and no install_id → silently disabled."""
    r = TelemetryRecorder(api_url="https://api.profine.ai", client_version="1.0")
    assert r.enabled is False


def test_recorder_disabled_when_no_api_url():
    r = TelemetryRecorder(api_key="pf_live_xxx")
    assert r.enabled is False


def test_recorder_rejects_both_api_key_and_install_id():
    with pytest.raises(ValueError, match="api_key.*install_id"):
        TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="pf_live_xxx", install_id="abc")


def test_recorder_paid_mode_enabled():
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="pf_live_xxx", client_version="0.1")
    assert r.enabled is True


def test_recorder_oss_mode_enabled():
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          install_id="abc", client_version="0.1")
    assert r.enabled is True


def test_recorder_explicit_disable_overrides_credentials():
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="pf_live_xxx", enabled=False)
    assert r.enabled is False


# ----------------------------- recording API -------------------------------


def test_recording_is_noop_when_disabled():
    r = TelemetryRecorder(api_url="https://api.profine.ai", enabled=False)
    r.begin_run(_fp())
    r.record_optimization("compile_default", catalog_version="v1", applied=True)
    # _outcomes stays empty; nothing to drain
    assert r._drain_payload() is None


def test_drain_returns_none_without_fingerprint():
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="x")
    r.record_optimization("compile_default", catalog_version="v1", applied=True)
    assert r._drain_payload() is None  # no fingerprint set → drop


def test_drain_returns_none_without_outcomes():
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="x")
    r.begin_run(_fp())
    assert r._drain_payload() is None  # fingerprint but no outcomes → drop


def test_record_optimization_appends():
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="x")
    r.begin_run(_fp())
    r.record_optimization("compile_default", catalog_version="v1",
                          applied=True, speedup_factor=1.4, loss_ok=True)
    r.record_optimization("amp_bf16", catalog_version="v1",
                          applied=True, speedup_factor=1.1, loss_ok=True)
    payload = r._drain_payload()
    assert payload is not None
    assert len(payload["outcomes"]) == 2
    assert payload["outcomes"][0]["optimization_id"] == "compile_default"


def test_drain_resets_outcome_queue():
    """A second drain returns None because outcomes were taken."""
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="x")
    r.begin_run(_fp())
    r.record_optimization("o1", catalog_version="v1", applied=True)
    assert r._drain_payload() is not None
    assert r._drain_payload() is None


def test_drain_keeps_fingerprint_between_flushes():
    """begin_run is set once per run; subsequent records use the same fp."""
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="x")
    r.begin_run(_fp())
    r.record_optimization("o1", catalog_version="v1", applied=True)
    r._drain_payload()
    r.record_optimization("o2", catalog_version="v1", applied=True)
    p2 = r._drain_payload()
    assert p2 is not None
    assert p2["fingerprint"]["fingerprint_hash"] == _fp().fingerprint_hash


# ----------------------------- crash invariant -----------------------------


def test_record_normalises_crashed_with_missing_class():
    """crashed=True but crash_class=None → fill in 'other'."""
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="x")
    r.begin_run(_fp())
    r.record_optimization("o1", catalog_version="v1", applied=False, crashed=True)
    payload = r._drain_payload()
    assert payload["outcomes"][0]["crash_class"] == "other"


def test_record_drops_inconsistent_crash_class():
    """crashed=False but crash_class set → drop the row (DB constraint
    would reject it; better to discard locally)."""
    r = TelemetryRecorder(api_url="https://api.profine.ai", api_key="x")
    r.begin_run(_fp())
    r.record_optimization("o1", catalog_version="v1", applied=True,
                          crashed=False, crash_class="oom")
    # That row was discarded. With nothing to send, drain returns None.
    assert r._drain_payload() is None


# ----------------------------- HTTP integration ----------------------------


def test_flush_posts_to_paid_endpoint():
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="pf_live_xxx", client_version="0.3.5")
    r.begin_run(_fp())
    r.record_optimization("compile_default", catalog_version="v1",
                          applied=True, speedup_factor=1.4, loss_ok=True)

    with patch("profine.telemetry.recorder.urlopen") as urlopen_mock:
        urlopen_mock.return_value.__enter__.return_value.read.return_value = b""
        r.flush()
        _wait_for_background_threads()

    assert urlopen_mock.call_count == 1
    request = urlopen_mock.call_args.args[0]
    assert request.full_url == "https://api.profine.ai/api/telemetry/run"
    assert request.get_header("Authorization") == "Bearer pf_live_xxx"
    body = json.loads(request.data.decode("utf-8"))
    assert body["fingerprint"]["arch_class"] == "transformer-decoder"
    assert body["outcomes"][0]["optimization_id"] == "compile_default"


def test_flush_posts_to_anon_endpoint_for_oss():
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          install_id="abc-123", client_version="0.3.5")
    r.begin_run(_fp())
    r.record_optimization("compile_default", catalog_version="v1", applied=True)

    with patch("profine.telemetry.recorder.urlopen") as urlopen_mock:
        urlopen_mock.return_value.__enter__.return_value.read.return_value = b""
        r.flush()
        _wait_for_background_threads()

    request = urlopen_mock.call_args.args[0]
    assert request.full_url == "https://api.profine.ai/api/telemetry/anon"
    assert request.get_header("Authorization") is None  # anon, no auth header
    body = json.loads(request.data.decode("utf-8"))
    assert body["install_id"] == "abc-123"


def test_flush_filters_out_unknown_fields():
    """A future caller that adds a non-allowlisted field must not leak it."""
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="x", client_version="0.1")

    # Construct a Fingerprint and tamper with its dict to simulate
    # a future schema drift that adds an unsafe field.
    fp = _fp()
    rogue_dict = {**fp.as_dict(), "leak_me": "/users/foo/secret.py"}

    # Patch as_dict to return the rogue dict — this is what would
    # happen if Fingerprint silently gained a new field that the
    # allowlist doesn't know about.
    with patch.object(Fingerprint, "as_dict", return_value=rogue_dict):
        r.begin_run(fp)
        r.record_optimization("o1", catalog_version="v1", applied=True)
        with patch("profine.telemetry.recorder.urlopen") as urlopen_mock:
            urlopen_mock.return_value.__enter__.return_value.read.return_value = b""
            r.flush()
            _wait_for_background_threads()

    body = json.loads(urlopen_mock.call_args.args[0].data.decode("utf-8"))
    assert "leak_me" not in body["fingerprint"]


def test_flush_never_blocks_caller():
    """A slow backend must not delay the recording call site."""
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="x", client_version="0.1")
    r.begin_run(_fp())
    r.record_optimization("o1", catalog_version="v1", applied=True)

    def slow_urlopen(*_args, **_kwargs):
        time.sleep(5.0)  # would exceed reasonable test budget if it blocked
        m = MagicMock()
        m.__enter__.return_value.read.return_value = b""
        return m

    with patch("profine.telemetry.recorder.urlopen", side_effect=slow_urlopen):
        start = time.monotonic()
        r.flush()
        elapsed = time.monotonic() - start
    assert elapsed < 0.2, f"flush() blocked for {elapsed:.2f}s"


def test_flush_swallows_network_errors():
    """A network failure must not propagate; the run keeps going."""
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="x", client_version="0.1")
    r.begin_run(_fp())
    r.record_optimization("o1", catalog_version="v1", applied=True)

    with patch("profine.telemetry.recorder.urlopen",
               side_effect=URLError("offline")):
        r.flush()
        _wait_for_background_threads()
    # If we got here without raising, the failure was swallowed.


def test_close_is_idempotent():
    r = TelemetryRecorder(api_url="https://api.profine.ai",
                          api_key="x", client_version="0.1")
    r.begin_run(_fp())
    r.record_optimization("o1", catalog_version="v1", applied=True)
    with patch("profine.telemetry.recorder.urlopen") as urlopen_mock:
        urlopen_mock.return_value.__enter__.return_value.read.return_value = b""
        r.close()
        r.close()  # must not double-send or raise
        _wait_for_background_threads()
    assert urlopen_mock.call_count <= 1


def test_disabled_recorder_does_not_post():
    r = TelemetryRecorder(api_url="https://api.profine.ai", enabled=False)
    r.begin_run(_fp())
    r.record_optimization("o1", catalog_version="v1", applied=True)
    with patch("profine.telemetry.recorder.urlopen") as urlopen_mock:
        r.flush()
        _wait_for_background_threads()
    assert urlopen_mock.call_count == 0
