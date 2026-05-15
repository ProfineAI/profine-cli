"""Tests for the client-side priors fetcher.

We mock urlopen so no real HTTP happens. The contract we're verifying:

  * Successful response → typed OptimizationPrior dict.
  * Empty server response → empty dict (cold start, no error).
  * Any HTTP / parse error → empty dict (cold-start fallback).
  * Repeated calls for the same fingerprint → one network roundtrip,
    not many (per-process memoization).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from profine.telemetry.priors import OptimizationPrior, PriorsClient


HASH = "a" * 64
CAT_V = "abc1234567890def"


def _ok_response(rows: list[dict]) -> MagicMock:
    """Build a fake context-manager urlopen response."""
    body = json.dumps({
        "fingerprint_hash": HASH, "catalog_version": CAT_V, "priors": rows,
    }).encode()
    m = MagicMock()
    m.__enter__.return_value.read.return_value = body
    return m


# ----------------------------- happy path ---------------------------------


def test_returns_typed_priors_from_server():
    rows = [
        {"optimization_id": "compile_default", "n_runs": 23,
         "speedup_p25": 1.20, "speedup_p50": 1.34,
         "speedup_p75": 1.50, "speedup_p95": 1.91,
         "success_rate": 0.91, "success_rate_lo": 0.72, "success_rate_hi": 0.97,
         "crash_rate": 0.04, "crash_rate_lo": 0.01, "crash_rate_hi": 0.20},
        {"optimization_id": "amp_bf16", "n_runs": 12,
         "speedup_p25": 1.05, "speedup_p50": 1.10,
         "speedup_p75": 1.15, "speedup_p95": 1.25,
         "success_rate": 1.0, "success_rate_lo": 0.78, "success_rate_hi": 1.0,
         "crash_rate": 0.0, "crash_rate_lo": 0.0, "crash_rate_hi": 0.22},
    ]
    client = PriorsClient(api_url="https://api.profine.ai")
    with patch("profine.telemetry.priors.urlopen", return_value=_ok_response(rows)):
        priors = client.for_fingerprint(HASH, CAT_V)

    assert set(priors.keys()) == {"compile_default", "amp_bf16"}
    cd = priors["compile_default"]
    assert isinstance(cd, OptimizationPrior)
    assert cd.n_runs == 23
    assert cd.speedup_p50 == pytest.approx(1.34)
    assert cd.speedup_iqr == pytest.approx(0.30)
    assert cd.crash_rate_hi == pytest.approx(0.20)
    assert priors["amp_bf16"].crash_rate == 0.0


def test_speedup_iqr_returns_none_when_p25_or_p75_missing():
    """Cold-start view rows (or older versions) may omit p25/p75."""
    rows = [{"optimization_id": "o1", "n_runs": 5,
             "speedup_p50": 1.2, "speedup_p95": 1.3,
             "crash_rate": 0.0}]
    client = PriorsClient(api_url="https://api.profine.ai")
    with patch("profine.telemetry.priors.urlopen", return_value=_ok_response(rows)):
        priors = client.for_fingerprint(HASH, CAT_V)
    assert priors["o1"].speedup_iqr is None


def test_per_process_cache_avoids_repeat_fetch():
    client = PriorsClient(api_url="https://api.profine.ai")
    rows = [{"optimization_id": "o1", "n_runs": 5, "speedup_p50": 1.2,
             "speedup_p95": 1.3, "success_rate": 1.0, "crash_rate": 0.0}]
    with patch("profine.telemetry.priors.urlopen",
               return_value=_ok_response(rows)) as urlopen_mock:
        client.for_fingerprint(HASH, CAT_V)
        client.for_fingerprint(HASH, CAT_V)
        client.for_fingerprint(HASH, CAT_V)
    assert urlopen_mock.call_count == 1


def test_different_fingerprints_each_fetch_once():
    client = PriorsClient(api_url="https://api.profine.ai")
    rows = []
    with patch("profine.telemetry.priors.urlopen",
               return_value=_ok_response(rows)) as urlopen_mock:
        client.for_fingerprint("a" * 64, CAT_V)
        client.for_fingerprint("b" * 64, CAT_V)
        client.for_fingerprint("a" * 64, CAT_V)  # cache hit
    assert urlopen_mock.call_count == 2


# ----------------------------- cold-start / failure -----------------------


def test_empty_response_returns_empty_dict():
    client = PriorsClient(api_url="https://api.profine.ai")
    with patch("profine.telemetry.priors.urlopen",
               return_value=_ok_response([])):
        priors = client.for_fingerprint(HASH, CAT_V)
    assert priors == {}


def test_network_error_returns_empty_dict_without_raising():
    client = PriorsClient(api_url="https://api.profine.ai")
    with patch("profine.telemetry.priors.urlopen",
               side_effect=URLError("DNS")):
        priors = client.for_fingerprint(HASH, CAT_V)
    assert priors == {}


def test_bad_json_returns_empty_dict():
    client = PriorsClient(api_url="https://api.profine.ai")
    m = MagicMock()
    m.__enter__.return_value.read.return_value = b"not json{"
    with patch("profine.telemetry.priors.urlopen", return_value=m):
        priors = client.for_fingerprint(HASH, CAT_V)
    assert priors == {}


def test_response_missing_priors_key_returns_empty():
    client = PriorsClient(api_url="https://api.profine.ai")
    m = MagicMock()
    m.__enter__.return_value.read.return_value = b'{"something_else": []}'
    with patch("profine.telemetry.priors.urlopen", return_value=m):
        priors = client.for_fingerprint(HASH, CAT_V)
    assert priors == {}


def test_malformed_row_is_dropped_others_kept():
    rows = [
        {"optimization_id": "good", "n_runs": 5, "speedup_p50": 1.1,
         "speedup_p95": 1.2, "success_rate": 1.0, "crash_rate": 0.0},
        {"this": "row", "is": "bad"},  # missing required field
    ]
    client = PriorsClient(api_url="https://api.profine.ai")
    with patch("profine.telemetry.priors.urlopen",
               return_value=_ok_response(rows)):
        priors = client.for_fingerprint(HASH, CAT_V)
    assert set(priors.keys()) == {"good"}


# ----------------------------- request shape ------------------------------


def test_fetch_builds_correct_url():
    client = PriorsClient(api_url="https://api.profine.ai")
    with patch("profine.telemetry.priors.urlopen",
               return_value=_ok_response([])) as urlopen_mock:
        client.for_fingerprint(HASH, CAT_V)
    request = urlopen_mock.call_args.args[0]
    assert request.method == "GET"
    expected_prefix = "https://api.profine.ai/api/telemetry/priors?"
    assert request.full_url.startswith(expected_prefix)
    assert f"fingerprint_hash={HASH}" in request.full_url
    assert f"catalog_version={CAT_V}" in request.full_url


def test_trailing_slash_in_api_url_is_normalised():
    client = PriorsClient(api_url="https://api.profine.ai/")
    with patch("profine.telemetry.priors.urlopen",
               return_value=_ok_response([])) as urlopen_mock:
        client.for_fingerprint(HASH, CAT_V)
    url = urlopen_mock.call_args.args[0].full_url
    # No double-slash before /api/...
    assert "//api/telemetry" not in url


def test_speedup_fields_can_be_null():
    rows = [{"optimization_id": "o1", "n_runs": 1, "speedup_p50": None,
             "speedup_p95": None, "success_rate": None, "crash_rate": 0.5}]
    client = PriorsClient(api_url="https://api.profine.ai")
    with patch("profine.telemetry.priors.urlopen",
               return_value=_ok_response(rows)):
        priors = client.for_fingerprint(HASH, CAT_V)
    assert priors["o1"].speedup_p50 is None
    assert priors["o1"].crash_rate == 0.5
