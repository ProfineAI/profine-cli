"""Allowlist of fields that may be sent in telemetry payloads.

This module exists to make the privacy claim auditable: nothing leaves
the process unless the field is in `ALLOWED_FINGERPRINT_FIELDS` or
`ALLOWED_OUTCOME_FIELDS`. Code review of any change touching these
sets is the gate.

Anti-goals — fields we never include even if a caller passes them in:
  * script_path, script_source, model checkpoint path
  * dataset path or any user-supplied filename
  * raw exception messages (use crash_class instead)
  * absolute hyperparameter values (only bucketed/binarised forms)
  * any user, account, or org identifier (the backend attaches those
    after the fact via FK; the client never sends them)

The filter() helper drops unknown keys silently. We prefer silent drop
to raising so a faulty caller can't crash a customer's run, but in
tests we assert the dropped set is empty.
"""

from __future__ import annotations

from typing import Any, Mapping


# Anything sent in the `fingerprint` field of a telemetry write.
#
# Split into two conceptual groups (both go on the wire, both safe to
# log, but only the first group feeds fingerprint_hash):
#
#   1. K-anonymity surface  — the seven dims that go into the hash.
#   2. Recorded enrichment  — richer enum dims we collect for future
#      analytics. Not in the hash today; can be promoted later by
#      bumping the catalog_version.
ALLOWED_FINGERPRINT_FIELDS: frozenset[str] = frozenset({
    # ----- group 1: in-hash -----
    "arch_class",
    "param_bucket",
    "hardware_class",
    "precision",
    "optimizer_class",
    "has_compile",
    "has_distributed",
    "fingerprint_hash",
    # ----- group 2: recorded enrichment -----
    "compile_mode",
    "distributed_strategy",
    "attention_impl",
    "framework",
    "gradient_checkpointing",
    "has_grad_scaler",
})

# Anything sent per-optimization-attempt row.
ALLOWED_OUTCOME_FIELDS: frozenset[str] = frozenset({
    "optimization_id",
    "catalog_version",
    "applied",
    "speedup_factor",
    "loss_ok",
    "crashed",
    "crash_class",
    "runtime_seconds",
})

# Per-run profile statistics — written once per run, separately from
# per-optimization outcomes. Aggregated values only (p50/p95/peak);
# never the raw step_times_ms array.
ALLOWED_PROFILE_STATS_FIELDS: frozenset[str] = frozenset({
    # run-level timing
    "runtime_seconds",
    "steps_completed",
    "warmup_steps_detected",
    # step time distribution (steady state only)
    "step_time_p50_ms",
    "step_time_p95_ms",
    "step_time_cv",
    # GPU util (percent 0-100)
    "gpu_util_p50_pct",
    "gpu_util_p95_pct",
    # memory
    "memory_peak_gb",
    "memory_peak_pct",
    # bottleneck classification
    "primary_bottleneck",
    "compute_pct",
    "dataloader_stall_pct",
    "communication_pct",
})


def filter_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the keys in ALLOWED_FINGERPRINT_FIELDS."""
    return {k: payload[k] for k in payload.keys() & ALLOWED_FINGERPRINT_FIELDS}


def filter_outcome(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the keys in ALLOWED_OUTCOME_FIELDS."""
    return {k: payload[k] for k in payload.keys() & ALLOWED_OUTCOME_FIELDS}


def filter_profile_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the keys in ALLOWED_PROFILE_STATS_FIELDS."""
    return {k: payload[k] for k in payload.keys() & ALLOWED_PROFILE_STATS_FIELDS}
