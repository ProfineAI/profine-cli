"""Tests for the failure-avoidance filter and the suggester blend.

These exercise the "agent uses the priors" half of the read path.
The priors dict is built directly in each test — no HTTP, no mocks.
"""

from __future__ import annotations

import pytest

from profine.catalog.schema import CatalogEntry
from profine.schema.optimization_candidate import OptimizationCandidate
from profine.telemetry.agent_priors import (
    DEFAULT_BLEND_SATURATION_RUNS,
    DEFAULT_CRASH_RATE_LOWER_BOUND,
    DEFAULT_IQR_TRUST_CEILING,
    DEFAULT_MIN_RUNS_FOR_TRUST,
    blend_candidate_ranking,
    filter_high_crash_rate,
)
from profine.telemetry.priors import OptimizationPrior


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Reference Wilson CI for tests — mirrors the SQL implementation."""
    if n <= 0:
        return (0.0, 0.0)
    import math
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (center - margin) / denom),
            min(1.0, (center + margin) / denom))


# ----------------------------- factories ---------------------------------


def _entry(eid: str, name: str = "") -> CatalogEntry:
    return CatalogEntry(id=eid, category="precision", name=name or eid, description="")


def _candidate(eid: str, lo: float, hi: float, rank: int = 1) -> OptimizationCandidate:
    return OptimizationCandidate(
        entry_id=eid, category="precision", name=eid,
        description="", rank=rank,
        est_speedup_low_pct=lo, est_speedup_high_pct=hi,
    )


def _prior(
    eid: str,
    *,
    n_runs: int,
    speedup_p50: float | None = None,
    crash_rate: float = 0.0,
    iqr: float = 0.0,
) -> OptimizationPrior:
    """Reasonable test fixture.

    Wilson CIs are computed from (crash_rate, n_runs) the same way
    the SQL view computes them, so a prior with n=10 and 60% crash
    has the same `crash_rate_lo` here as it would have on the live DB.
    """
    n_crashed = round(crash_rate * n_runs)
    crash_lo, crash_hi = _wilson(n_crashed, n_runs)
    # success counted only when no crash
    n_success_obs = n_runs - n_crashed
    if n_success_obs > 0:
        # Assume all successful runs had loss_ok=True for the simple helper
        success_lo, success_hi = _wilson(n_success_obs, n_success_obs)
        success_rate = 1.0
    else:
        success_lo, success_hi, success_rate = None, None, None

    half_iqr = iqr / 2 if speedup_p50 is not None else 0
    p25 = (speedup_p50 - half_iqr) if speedup_p50 is not None else None
    p75 = (speedup_p50 + half_iqr) if speedup_p50 is not None else None

    return OptimizationPrior(
        optimization_id=eid,
        n_runs=n_runs,
        speedup_p25=p25,
        speedup_p50=speedup_p50,
        speedup_p75=p75,
        speedup_p95=speedup_p50,
        success_rate=success_rate,
        success_rate_lo=success_lo,
        success_rate_hi=success_hi,
        crash_rate=crash_rate,
        crash_rate_lo=crash_lo,
        crash_rate_hi=crash_hi,
    )


# ===========================================================
# filter_high_crash_rate
# ===========================================================


class TestFailureAvoidance:
    """The filter drops based on the 95% LOWER bound of crash_rate.
    A 100% point estimate with n=2 has a wide CI and lower bound below
    the threshold — kept. A 60% point estimate with n=20 has a tight
    CI and lower bound clearing the threshold — dropped.
    """

    def test_empty_priors_keeps_everything(self):
        entries = [_entry("a"), _entry("b"), _entry("c")]
        kept, stats = filter_high_crash_rate(entries, priors={})
        assert [e.id for e in kept] == ["a", "b", "c"]
        assert stats.dropped == []
        assert stats.kept == 3

    def test_drops_high_crash_rate_with_lots_of_evidence(self):
        """20 runs all crashed → Wilson lower bound ≈ 0.84 → drop."""
        entries = [_entry("safe"), _entry("crashy")]
        priors = {
            "safe":   _prior("safe",   n_runs=20, crash_rate=0.0),
            "crashy": _prior("crashy", n_runs=20, crash_rate=1.0),
        }
        kept, stats = filter_high_crash_rate(entries, priors)
        assert [e.id for e in kept] == ["safe"]
        assert len(stats.dropped) == 1
        assert stats.dropped[0][0] == "crashy"

    def test_keeps_high_crash_point_estimate_when_evidence_is_weak(self):
        """5/5 crash gives point=1.0 but Wilson lower ≈ 0.57 — still
        below 0.25 threshold? Actually 5/5 lower ≈ 0.57 > 0.25, so
        it does drop. Use 2 runs (under min_runs) instead."""
        entries = [_entry("maybe_crashy")]
        priors = {"maybe_crashy": _prior("maybe_crashy", n_runs=2, crash_rate=1.0)}
        kept, stats = filter_high_crash_rate(entries, priors)
        # min_runs=5 gate keeps this; we don't even look at the CI.
        assert [e.id for e in kept] == ["maybe_crashy"]
        assert stats.dropped == []

    def test_keeps_cold_start_entries(self):
        """Entries with no prior at all are passed through untouched."""
        entries = [_entry("brand_new")]
        kept, stats = filter_high_crash_rate(entries, priors={})
        assert [e.id for e in kept] == ["brand_new"]

    def test_wide_ci_keeps_entry_despite_high_point_estimate(self):
        """5/5 crashed: point=1.0, Wilson lower≈0.57. With a stricter
        threshold of 0.70, the lower bound doesn't clear → keep.

        This documents the key behaviour: CI-based filter is naturally
        more conservative at small n than the old point-estimate one.
        """
        entries = [_entry("a")]
        priors = {"a": _prior("a", n_runs=5, crash_rate=1.0)}
        kept, _ = filter_high_crash_rate(entries, priors, crash_rate_lower_bound=0.70)
        assert [e.id for e in kept] == ["a"]

    def test_tight_ci_drops_at_aggressive_threshold(self):
        """100/100 crashed: lower bound > 0.96. Even an aggressive
        threshold like 0.90 still trips."""
        entries = [_entry("a")]
        priors = {"a": _prior("a", n_runs=100, crash_rate=1.0)}
        kept, _ = filter_high_crash_rate(entries, priors, crash_rate_lower_bound=0.90)
        assert kept == []

    def test_lower_threshold_drops_smaller_evidence(self):
        """Tunable threshold: at lower_bound=0.10, weak evidence
        (5/5 crashed → lower≈0.57) clears it easily."""
        entries = [_entry("a")]
        priors = {"a": _prior("a", n_runs=5, crash_rate=1.0)}
        kept, _ = filter_high_crash_rate(entries, priors, crash_rate_lower_bound=0.10)
        assert kept == []

    def test_tunable_min_runs(self):
        entries = [_entry("a")]
        priors = {"a": _prior("a", n_runs=3, crash_rate=1.0)}
        # min_runs=2 → respect the CI on 3/3 (lower bound ≈ 0.31) →
        # drops at default threshold of 0.25.
        kept_strict, _ = filter_high_crash_rate(entries, priors, min_runs=2)
        assert kept_strict == []
        # min_runs=5 (default) → 3 runs is below the gate → keep.
        kept_default, _ = filter_high_crash_rate(entries, priors)
        assert len(kept_default) == 1

    def test_stats_reason_text_includes_lower_bound_and_n_runs(self):
        """The audit message in stats.dropped quotes the lower bound,
        not just the point estimate, so a human reading logs can see
        the CI-based decision."""
        entries = [_entry("crashy")]
        priors = {"crashy": _prior("crashy", n_runs=20, crash_rate=1.0)}
        _, stats = filter_high_crash_rate(entries, priors)
        msg = stats.dropped[0][1]
        assert "crash_rate" in msg
        assert "95% lower bound" in msg
        assert "20 runs" in msg


# ===========================================================
# blend_candidate_ranking
# ===========================================================


class TestBlendRanking:
    def test_empty_priors_preserves_llm_order_by_score(self):
        # cands: A(20% midpoint), B(50% midpoint), C(10% midpoint)
        cands = [
            _candidate("A", lo=10, hi=30, rank=1),
            _candidate("B", lo=40, hi=60, rank=2),
            _candidate("C", lo=5,  hi=15, rank=3),
        ]
        ranked, infos = blend_candidate_ranking(cands, priors={})
        assert [c.entry_id for c in ranked] == ["B", "A", "C"]
        # Ranks are 1-based and updated in place.
        assert [c.rank for c in ranked] == [1, 2, 3]
        assert all(info.used_prior is False for info in infos)

    def test_strong_prior_overrides_weak_llm_estimate(self):
        """LLM thinks A is best, but prior says B has the real wins."""
        cands = [
            _candidate("A", lo=40, hi=60, rank=1),   # llm_factor = 1.50
            _candidate("B", lo=5,  hi=15, rank=2),   # llm_factor = 1.10
        ]
        # B has tons of evidence at 1.8x. A has nothing.
        priors = {"B": _prior("B", n_runs=100, speedup_p50=1.8)}
        ranked, infos = blend_candidate_ranking(cands, priors)
        assert ranked[0].entry_id == "B"
        # Find B's info to verify it used the prior with high alpha.
        b_info = next(i for i in infos if i.entry_id == "B")
        assert b_info.used_prior is True
        assert b_info.n_prior_runs == 100

    def test_few_run_prior_does_not_override_strong_llm(self):
        cands = [
            _candidate("A", lo=40, hi=60, rank=1),     # llm_factor = 1.50
            _candidate("B", lo=5,  hi=15, rank=2),     # llm_factor = 1.10
        ]
        # B has just 3 runs — below trust threshold; ignore prior even if huge.
        priors = {"B": _prior("B", n_runs=3, speedup_p50=2.5)}
        ranked, infos = blend_candidate_ranking(cands, priors)
        assert ranked[0].entry_id == "A"  # LLM-only ordering survives
        b_info = next(i for i in infos if i.entry_id == "B")
        assert b_info.used_prior is False

    def test_alpha_saturates_at_threshold_runs(self):
        cands = [_candidate("A", lo=0, hi=0)]  # llm_factor = 1.0
        priors = {"A": _prior("A", n_runs=DEFAULT_BLEND_SATURATION_RUNS,
                              speedup_p50=2.0)}
        _, infos = blend_candidate_ranking(cands, priors)
        # alpha = 50/50 = 1.0 → blended = 2.0 exactly.
        assert infos[0].blended_score == pytest.approx(2.0)

    def test_alpha_below_saturation(self):
        cands = [_candidate("A", lo=0, hi=0)]  # llm_factor = 1.0
        # 25 runs / 50 saturation → alpha = 0.5; blended = 0.5*1 + 0.5*2 = 1.5
        priors = {"A": _prior("A", n_runs=25, speedup_p50=2.0)}
        _, infos = blend_candidate_ranking(cands, priors)
        assert infos[0].blended_score == pytest.approx(1.5)

    def test_prior_with_null_speedup_p50_is_ignored(self):
        cands = [_candidate("A", lo=10, hi=30)]
        # Lots of runs but no benchmarked speedup measurement (all applied=False).
        priors = {"A": _prior("A", n_runs=100, speedup_p50=None)}
        _, infos = blend_candidate_ranking(cands, priors)
        assert infos[0].used_prior is False

    def test_zero_llm_estimate_does_not_crash(self):
        """An LLM that left est_speedup empty must still sort sensibly."""
        cands = [
            _candidate("A", lo=0, hi=0),
            _candidate("B", lo=10, hi=20),
        ]
        ranked, _ = blend_candidate_ranking(cands, priors={})
        # B has the only positive evidence; should sort first.
        assert ranked[0].entry_id == "B"

    def test_candidates_with_priors_mix_with_those_without(self):
        cands = [
            _candidate("LLM_only_high", lo=40, hi=60),  # 1.50
            _candidate("Prior_low",     lo=20, hi=20),  # 1.20 LLM
        ]
        # Prior_low has strong empirical evidence of low speedup → stays below.
        priors = {"Prior_low": _prior("Prior_low", n_runs=50, speedup_p50=1.05)}
        ranked, _ = blend_candidate_ranking(cands, priors)
        assert ranked[0].entry_id == "LLM_only_high"

    def test_returns_new_list_does_not_mutate_input(self):
        cands = [_candidate("A", lo=10, hi=20, rank=1),
                 _candidate("B", lo=30, hi=40, rank=2)]
        ranked, _ = blend_candidate_ranking(cands, priors={})
        # Original ranks unchanged
        assert cands[0].rank == 1
        assert cands[1].rank == 2
        # Output reflects new ranking
        assert ranked[0].entry_id == "B"
        assert ranked[0].rank == 1

    def test_wide_iqr_discounts_the_prior_score(self):
        """Two candidates with identical n_runs and identical point
        estimates of the prior, but different IQRs. The narrower IQR
        gets higher alpha and rides the prior closer to its p50; the
        wider IQR gets less weight on the prior.
        """
        cands = [
            _candidate("narrow_iqr", lo=0, hi=0),    # llm_factor = 1.0
            _candidate("wide_iqr",   lo=0, hi=0),    # llm_factor = 1.0
        ]
        priors = {
            # IQR=0 means perfectly consistent — alpha_consistency = 1.0
            "narrow_iqr": _prior("narrow_iqr", n_runs=50, speedup_p50=2.0, iqr=0.0),
            # IQR is 100% of p50 — well above the trust ceiling → α_consistency = 0
            "wide_iqr":   _prior("wide_iqr",   n_runs=50, speedup_p50=2.0, iqr=2.0),
        }
        ranked, infos = blend_candidate_ranking(cands, priors)
        # Narrow IQR should sort first (its score = 2.0)
        # Wide IQR's score reverts toward LLM-only (1.0)
        assert ranked[0].entry_id == "narrow_iqr"
        narrow = next(i for i in infos if i.entry_id == "narrow_iqr")
        wide = next(i for i in infos if i.entry_id == "wide_iqr")
        assert narrow.alpha == pytest.approx(1.0)
        assert wide.alpha == pytest.approx(0.0)
        assert wide.blended_score == pytest.approx(1.0)
        assert narrow.blended_score == pytest.approx(2.0)

    def test_alpha_is_product_of_sample_and_consistency(self):
        """25 runs of 50 saturation → α_sample = 0.5.
        IQR is 25% of p50, with ceiling 0.5 → α_consistency = 0.5.
        Final α = 0.25; blended = 0.75*1.0 + 0.25*2.0 = 1.25."""
        cands = [_candidate("A", lo=0, hi=0)]
        priors = {"A": _prior("A", n_runs=25, speedup_p50=2.0, iqr=0.5)}
        _, infos = blend_candidate_ranking(cands, priors)
        info = infos[0]
        assert info.alpha_sample == pytest.approx(0.5)
        assert info.alpha_consistency == pytest.approx(0.5)
        assert info.alpha == pytest.approx(0.25)
        assert info.blended_score == pytest.approx(1.25)

    def test_missing_iqr_does_not_penalize_prior(self):
        """Old-schema priors with no p25/p75 (iqr=None) should
        behave as if iqr_consistency=1.0 — no penalty, matches
        legacy blend math."""
        # Build a prior manually with p25/p75 = None to simulate
        # cold-start view shape.
        prior = OptimizationPrior(
            optimization_id="A", n_runs=50,
            speedup_p25=None, speedup_p50=2.0, speedup_p75=None, speedup_p95=2.0,
            success_rate=1.0, success_rate_lo=1.0, success_rate_hi=1.0,
            crash_rate=0.0, crash_rate_lo=0.0, crash_rate_hi=0.0,
        )
        cands = [_candidate("A", lo=0, hi=0)]
        _, infos = blend_candidate_ranking(cands, {"A": prior})
        assert infos[0].alpha == pytest.approx(1.0)
        assert infos[0].blended_score == pytest.approx(2.0)
