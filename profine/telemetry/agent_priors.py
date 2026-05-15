"""Apply priors to the agent's two decision points.

Pure functions that turn priors into pipeline decisions:

  * `filter_high_crash_rate` runs BEFORE the suggester. Drops catalog
    entries whose prior crash_rate clears a (statistically-trusted)
    threshold. Cold-start entries fall through unchanged.

  * `blend_candidate_ranking` runs AFTER the suggester produces its
    LLM-ranked SuggestionReport. Reweights candidates by an empirical
    posterior so optimizations that worked on similar setups float
    upward, ones that disappointed sink.

Both functions are deliberately decoupled from HTTP. They take the
priors dict the PriorsClient produced — easy to test, easy to bypass
when the user opts out (caller passes an empty dict).

Threshold defaults are conservative:
  * crash_threshold = 0.30 (drop only when 30%+ of measured runs
    crashed — clear avoid signal, not a flaky kernel)
  * min_runs = 5 (matches the k-anonymity floor of the priors view —
    no looser, no tighter)
  * weight_saturates_at = 50 (50 runs is plenty; beyond that the
    prior dominates the LLM at α≈1.0)

These are tunable via kwargs so the suggester can be more or less
aggressive without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from profine.catalog.schema import CatalogEntry
    from profine.schema.optimization_candidate import OptimizationCandidate

from profine.telemetry.priors import OptimizationPrior


# ----- module-level defaults (single source of truth) ---------------------

# Filter when the 95% LOWER credible bound on crash_rate clears this
# threshold. Reading: "we're 95% sure the true crash rate is at least
# 25%". Conservative on purpose — we drop only when there's strong
# evidence, not when one bad run inflated the point estimate.
DEFAULT_CRASH_RATE_LOWER_BOUND: float = 0.25

# K-anonymity floor for the priors view also gates trust here.
DEFAULT_MIN_RUNS_FOR_TRUST: int = 5

# Number of runs after which α (blend weight on the prior) saturates
# to 1.0. Below this, α scales linearly with run count, so a brand-new
# optimization with 5 runs still mostly trusts the LLM.
DEFAULT_BLEND_SATURATION_RUNS: int = 50

# Widest acceptable IQR (as a fraction of the median) before we
# DOWN-WEIGHT the prior — past this, the spread is too noisy to trust
# even if we have lots of samples. e.g. p50=1.4, p25/p75=[1.0, 1.8],
# IQR=0.8, IQR/p50=0.57 → discount the prior heavily.
DEFAULT_IQR_TRUST_CEILING: float = 0.50


# ===========================================================
# Failure-avoidance pre-filter
# ===========================================================


@dataclass(slots=True)
class FilterStats:
    """Audit trail for the pre-filter, surfaced in the run log."""
    kept: int = 0
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (entry_id, reason)


def filter_high_crash_rate(
    entries: list["CatalogEntry"],
    priors: dict[str, OptimizationPrior],
    *,
    crash_rate_lower_bound: float = DEFAULT_CRASH_RATE_LOWER_BOUND,
    min_runs: int = DEFAULT_MIN_RUNS_FOR_TRUST,
) -> tuple[list["CatalogEntry"], FilterStats]:
    """Drop entries that historically crash on this fingerprint.

    Decision rule (uses the 95% LOWER CI bound on crash_rate):
      * No prior, or fewer than `min_runs` contributing runs → keep
        (cold-start path; better to let the LLM try than to filter
        on insufficient evidence).
      * Point estimate alone is ignored: a 5/5 crash run has the
        same point estimate as 50/50 but very different confidence.
      * Drop only when `crash_rate_lo >= crash_rate_lower_bound`.
        That means: "with 95% confidence the true crash rate is at
        least [bound]%." Naturally conservative at small n (wide CI
        keeps the lower bound below the threshold).
    """
    kept: list["CatalogEntry"] = []
    stats = FilterStats()
    for entry in entries:
        prior = priors.get(entry.id)
        if prior is None or prior.n_runs < min_runs:
            kept.append(entry)
            continue
        if prior.crash_rate_lo >= crash_rate_lower_bound:
            stats.dropped.append((
                entry.id,
                f"crash_rate {prior.crash_rate:.0%} (95% lower bound "
                f"{prior.crash_rate_lo:.0%}) over {prior.n_runs} runs",
            ))
            continue
        kept.append(entry)
    stats.kept = len(kept)
    return kept, stats


# ===========================================================
# Suggester ranking blend
# ===========================================================


@dataclass(slots=True, frozen=True)
class BlendInfo:
    """Per-candidate annotation of how the new rank was derived.

    Kept separate from OptimizationCandidate so we don't have to touch
    the existing schema; the suggester step appends this to its
    SuggestionReport's `unstructured_notes` instead.

    `alpha` is the blend weight on the prior (0 = pure LLM, 1 = pure
    prior). It's the product of two trust factors: a sample-size
    factor (`n_runs / saturation_runs`, capped at 1) and an IQR
    factor (narrow spread = higher trust). Both are reported so a
    debugging human can tell *why* the prior was weighted as it was.
    """
    entry_id: str
    used_prior: bool
    n_prior_runs: int
    blended_score: float
    llm_score: float
    prior_score: float | None
    alpha: float
    alpha_sample: float        # weight from sample size alone
    alpha_consistency: float   # weight from IQR tightness alone


def blend_candidate_ranking(
    candidates: list["OptimizationCandidate"],
    priors: dict[str, OptimizationPrior],
    *,
    saturation_runs: int = DEFAULT_BLEND_SATURATION_RUNS,
    min_runs_for_trust: int = DEFAULT_MIN_RUNS_FOR_TRUST,
    iqr_trust_ceiling: float = DEFAULT_IQR_TRUST_CEILING,
) -> tuple[list["OptimizationCandidate"], list[BlendInfo]]:
    """Re-sort candidates by blending LLM speedup estimate with priors.

    Two-factor trust model:

      llm_factor       = 1 + (low + high) / 2 / 100  # range midpoint as a factor
      α_sample         = min(n_runs / saturation_runs, 1.0)
      α_consistency    = max(0, 1 - (IQR / p50) / iqr_trust_ceiling)
      α                = α_sample · α_consistency
      blended_score    = (1 − α) · llm_factor  +  α · prior.speedup_p50

    `α_consistency` is 1.0 when the IQR is zero (perfectly consistent
    observed speedups) and ramps to 0 as the IQR widens past
    `iqr_trust_ceiling` × p50. If we don't have p25/p75 we default
    α_consistency to 1.0 (treat as unknown spread rather than as
    "very noisy"), so the behaviour matches the old code path
    when the priors view doesn't yet expose IQR.

    A candidate with no usable prior keeps its LLM-only score.

    Returns (sorted_candidates, blend_info) — sorted_candidates has
    each item's `rank` updated to its new 1-based position; the
    rest of the fields are untouched.
    """
    scored: list[tuple[float, BlendInfo, "OptimizationCandidate"]] = []
    for c in candidates:
        llm_score = _llm_score(c)
        prior = priors.get(c.entry_id)

        if prior is None or prior.n_runs < min_runs_for_trust or prior.speedup_p50 is None:
            scored.append((
                llm_score,
                BlendInfo(
                    entry_id=c.entry_id, used_prior=False,
                    n_prior_runs=(prior.n_runs if prior else 0),
                    blended_score=llm_score, llm_score=llm_score, prior_score=None,
                    alpha=0.0, alpha_sample=0.0, alpha_consistency=1.0,
                ),
                c,
            ))
            continue

        alpha_sample = min(prior.n_runs / saturation_runs, 1.0)
        alpha_consistency = _consistency_factor(prior, iqr_trust_ceiling)
        alpha = alpha_sample * alpha_consistency
        blended = (1.0 - alpha) * llm_score + alpha * prior.speedup_p50

        scored.append((
            blended,
            BlendInfo(
                entry_id=c.entry_id, used_prior=alpha > 0,
                n_prior_runs=prior.n_runs,
                blended_score=blended, llm_score=llm_score,
                prior_score=prior.speedup_p50,
                alpha=alpha, alpha_sample=alpha_sample,
                alpha_consistency=alpha_consistency,
            ),
            c,
        ))

    # Sort descending: higher blended score → better rank → lower index.
    scored.sort(key=lambda triple: triple[0], reverse=True)

    re_ranked: list["OptimizationCandidate"] = []
    blend_infos: list[BlendInfo] = []
    for new_index, (_, info, candidate) in enumerate(scored, start=1):
        re_ranked.append(replace(candidate, rank=new_index))
        blend_infos.append(info)
    return re_ranked, blend_infos


def _consistency_factor(prior: OptimizationPrior, ceiling: float) -> float:
    """Trust multiplier in [0, 1] derived from the prior's IQR.

    Returns 1.0 when the IQR is missing (we can't measure spread,
    so we don't penalise) or zero (perfectly consistent runs).
    Returns 0.0 when the IQR exceeds `ceiling × p50`.
    """
    iqr = prior.speedup_iqr
    p50 = prior.speedup_p50
    if iqr is None or p50 is None or p50 <= 0 or ceiling <= 0:
        return 1.0
    relative_spread = iqr / p50
    return max(0.0, 1.0 - relative_spread / ceiling)


# ----- helpers ------------------------------------------------------------


def _llm_score(c: "OptimizationCandidate") -> float:
    """Convert an OptimizationCandidate's est_speedup range into a
    single factor on the same scale as `prior.speedup_p50`.

    speedup_p50 is a multiplier ≥1.0 (1.4 = 40% faster). The
    candidate stores the range as percentages; we convert via the
    midpoint. If both endpoints are zero (LLM didn't fill in a
    range), we fall back to 1.0 so the entry sorts after anything
    with positive evidence rather than crashing the sort.
    """
    lo = max(c.est_speedup_low_pct, 0.0)
    hi = max(c.est_speedup_high_pct, lo)
    midpoint_pct = (lo + hi) / 2.0
    return 1.0 + midpoint_pct / 100.0


__all__ = [
    "FilterStats",
    "BlendInfo",
    "filter_high_crash_rate",
    "blend_candidate_ranking",
    "DEFAULT_CRASH_RATE_LOWER_BOUND",
    "DEFAULT_MIN_RUNS_FOR_TRUST",
    "DEFAULT_BLEND_SATURATION_RUNS",
    "DEFAULT_IQR_TRUST_CEILING",
]
