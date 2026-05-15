"""Client-side reader for the priors backend.

Wraps GET /api/telemetry/priors with:

  * Per-fingerprint memoization (one network call per fingerprint
    per process — priors only refresh nightly server-side).
  * Strict timeout (priors are decision-time hints; we never block
    a profile run on a slow priors lookup).
  * Cold-start fallback (any failure → empty dict; callers degrade
    to their pre-existing LLM-only behavior).

No persistent on-disk cache. Each profine invocation starts fresh.
This keeps the read path simple and avoids "stale priors poisoned
the agent" debug stories.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Final
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# Short timeout — priors are a "nice to have." If the backend is slow,
# fall back to LLM-only ranking rather than block the customer.
_HTTP_TIMEOUT_SECONDS: Final[float] = 3.0

# Default endpoint; tests and self-hosted setups can override.
_DEFAULT_API_URL: Final[str] = "https://api.profine.ai"


log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class OptimizationPrior:
    """Per-(fingerprint, optimization) aggregate from the priors view.

    Three signal families:

      * Speedup distribution — non-parametric percentiles. p25/p75
        gives an honest spread; p50 is the central estimate; p95
        marks the tail.
      * Success rate (loss_ok=true / loss measured) with 95% Wilson
        credible interval. Null when no benchmarked run was applied.
      * Crash rate (any crash / all runs) with 95% Wilson credible
        interval. Always populated.

    All `_lo` / `_hi` fields are 95% credible bounds. Use the upper
    bound when refusing to recommend ("we're 95% sure crash rate is
    at least X") and the lower bound when refusing to celebrate
    ("we're 95% sure success rate is at most Y"). The point estimate
    alone is silently overconfident at small n.
    """
    optimization_id: str
    n_runs: int

    # speedup factor distribution (≥1.0 = faster)
    speedup_p25: float | None
    speedup_p50: float | None
    speedup_p75: float | None
    speedup_p95: float | None

    # success rate (0–1)
    success_rate: float | None
    success_rate_lo: float | None
    success_rate_hi: float | None

    # crash rate (0–1)
    crash_rate: float
    crash_rate_lo: float
    crash_rate_hi: float

    @classmethod
    def from_row(cls, row: dict) -> "OptimizationPrior":
        # Required fields raise KeyError if missing — the caller
        # (PriorsClient._fetch) catches and drops malformed rows.
        return cls(
            optimization_id=str(row["optimization_id"]),
            n_runs=int(row.get("n_runs", 0)),
            speedup_p25=_maybe_float(row.get("speedup_p25")),
            speedup_p50=_maybe_float(row.get("speedup_p50")),
            speedup_p75=_maybe_float(row.get("speedup_p75")),
            speedup_p95=_maybe_float(row.get("speedup_p95")),
            success_rate=_maybe_float(row.get("success_rate")),
            success_rate_lo=_maybe_float(row.get("success_rate_lo")),
            success_rate_hi=_maybe_float(row.get("success_rate_hi")),
            crash_rate=float(row.get("crash_rate") or 0.0),
            crash_rate_lo=float(row.get("crash_rate_lo") or 0.0),
            crash_rate_hi=float(row.get("crash_rate_hi") or 0.0),
        )

    @property
    def speedup_iqr(self) -> float | None:
        """Interquartile range of measured speedups.

        Spread metric we use to weight prior trust in the blend —
        narrow IQR means consistent results, wide IQR means the
        prior is too noisy to trust over the LLM. Returns None when
        we don't have both p25 and p75.
        """
        if self.speedup_p25 is None or self.speedup_p75 is None:
            return None
        return self.speedup_p75 - self.speedup_p25


class PriorsClient:
    """Lightweight HTTP client for the priors endpoint.

    Memoizes results per (fingerprint_hash, catalog_version) so the
    suggester and the failure-avoidance filter share a single fetch.

    Instances are not thread-safe by design — telemetry priors are
    queried from the main pipeline thread once per run.
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        timeout_seconds: float = _HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._api_url = (api_url or _DEFAULT_API_URL).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._cache: dict[tuple[str, str], dict[str, OptimizationPrior]] = {}

    def for_fingerprint(
        self,
        fingerprint_hash: str,
        catalog_version: str,
    ) -> dict[str, OptimizationPrior]:
        """Return {optimization_id → prior} for this fingerprint.

        Empty dict on any failure (network error, bad response, server
        rate-limit). Empty dict is also the legitimate cold-start
        result, so callers should treat "empty" as "no prior; use
        LLM-only logic."
        """
        cache_key = (fingerprint_hash, catalog_version)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        priors = self._fetch(fingerprint_hash, catalog_version)
        self._cache[cache_key] = priors
        return priors

    # ----- internal -------------------------------------------------------

    def _fetch(
        self,
        fingerprint_hash: str,
        catalog_version: str,
    ) -> dict[str, OptimizationPrior]:
        params = urlencode({
            "fingerprint_hash": fingerprint_hash,
            "catalog_version": catalog_version,
        })
        url = f"{self._api_url}/api/telemetry/priors?{params}"
        try:
            with urlopen(Request(url, method="GET"), timeout=self._timeout_seconds) as resp:
                body = resp.read()
        except URLError as e:
            log.debug("priors fetch unreachable: %s", e)
            return {}
        except TimeoutError:
            log.debug("priors fetch timed out")
            return {}
        except Exception:  # noqa: BLE001 — never raise on read path
            log.debug("priors fetch failed", exc_info=True)
            return {}

        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            log.debug("priors fetch returned non-JSON")
            return {}

        rows = payload.get("priors")
        if not isinstance(rows, list):
            return {}

        out: dict[str, OptimizationPrior] = {}
        for row in rows:
            try:
                prior = OptimizationPrior.from_row(row)
                out[prior.optimization_id] = prior
            except (KeyError, TypeError, ValueError):
                log.debug("priors fetch dropped malformed row: %r", row)
        return out


def _maybe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = ["OptimizationPrior", "PriorsClient"]
