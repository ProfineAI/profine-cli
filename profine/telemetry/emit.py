"""Read profine's on-disk run artifacts and emit telemetry.

The recorder doesn't need to be threaded through every subcommand.
Instead, after `run-all` (or any pipeline step) finishes writing its
JSON artifacts, this module:

  1. Loads what's on disk in `<output_dir>/{read,profile,edit,benchmark}/`.
  2. Builds a Fingerprint from the architecture + hardware facts.
  3. Builds outcome rows from the benchmark comparison and edit manifest.
  4. Pushes everything into a TelemetryRecorder.

Decoupling telemetry from the call chain means a pipeline crash mid-
run still results in whatever artifacts *did* land on disk getting
recorded. It also means we don't have to plumb a recorder object
through six command handlers.

All file reads are tolerant — missing files just produce fewer rows.
Type errors / malformed JSON drop the affected row but never raise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from profine.catalog import CATALOG_VERSION
from profine.telemetry.fingerprint import Fingerprint, fingerprint_run
from profine.telemetry.recorder import TelemetryRecorder

log = logging.getLogger(__name__)


# Subdirectory names — single source of truth. If the pipeline ever
# renames an output dir, update here and emit keeps working.
_READ_DIR = "read"
_PROFILE_DIR = "profile"
_INTERPRET_DIR = "interpret"
_EDIT_DIR = "edit"
_BENCHMARK_DIR = "benchmark"

_ARCH_FILE = "architecture_record.json"
_PROFILE_FILE = "profile_record.json"
_BOTTLENECK_FILE = "bottleneck_report.json"
_EDIT_MANIFEST = "change_manifest.json"
_BENCH_COMPARISON = "benchmark_comparison.json"


def emit_run(
    output_dir: Path,
    recorder: TelemetryRecorder,
    *,
    hardware_name: str | None = None,
) -> bool:
    """Read artifacts under `output_dir`, send telemetry through `recorder`.

    Returns True iff at least the fingerprint was emitted. The caller
    should still call recorder.close() afterward to flush.

    `hardware_name` is a fallback used when the profile record is
    missing/malformed; usually the CLI knows the hardware preset that
    was passed to `--hardware`.
    """
    if not recorder.enabled:
        return False

    arch_record = _safe_load_json(output_dir / _READ_DIR / _ARCH_FILE)
    if arch_record is None:
        log.debug("telemetry: no architecture_record.json under %s; skip", output_dir)
        return False

    profile_record = _safe_load_json(output_dir / _PROFILE_DIR / _PROFILE_FILE)
    resolved_hardware = _resolve_hardware(profile_record, hardware_name)
    if resolved_hardware is None:
        log.debug("telemetry: no usable hardware info; skip")
        return False

    fingerprint = _build_fingerprint(arch_record, resolved_hardware)
    if fingerprint is None:
        log.debug("telemetry: fingerprint build failed; skip")
        return False
    recorder.begin_run(fingerprint)

    # Per-run profile stats are independent of outcome rows — they
    # land even when no optimization was tried (e.g. cmd_profile alone).
    stats = _gather_profile_stats(output_dir, resolved_hardware)
    if stats:
        recorder.record_profile_stats(stats)

    outcomes = _gather_outcomes(output_dir)
    for opt_row in outcomes:
        recorder.record_optimization(**opt_row, catalog_version=CATALOG_VERSION)

    return True


# ===========================================================
# Profile stats extraction
# ===========================================================

# Heuristic thresholds for primary_bottleneck classification. Defaults
# match what the existing bottleneck interpreter uses in BottleneckReport.
_DATALOADER_STALL_BOTTLENECK_PCT: float = 10.0
_COMMUNICATION_OVERHEAD_BOTTLENECK_PCT: float = 10.0


def _gather_profile_stats(output_dir: Path, hardware) -> dict[str, Any] | None:
    """Build the profile-stats payload from on-disk artifacts.

    Composition of single-concern helpers. Each `_extract_*` returns a
    flat dict of fields and `None` values are filtered at the
    composition site so absent fields stay absent.
    """
    profile = _safe_load_json(output_dir / _PROFILE_DIR / _PROFILE_FILE)
    if profile is None:
        return None

    stats: dict[str, Any] = {}
    _update_nonnull(stats, _extract_run_timing(profile))
    _update_nonnull(stats, _extract_step_time_stats(profile))
    _update_nonnull(stats, _extract_gpu_stats(profile))
    _update_nonnull(stats, _extract_memory_stats(profile, hardware))
    _update_nonnull(stats, _extract_breakdown_pcts(profile))

    bottleneck = _safe_load_json(output_dir / _INTERPRET_DIR / _BOTTLENECK_FILE)
    _set_if(stats, "primary_bottleneck",
            _classify_primary_bottleneck(bottleneck, stats))

    return stats or None


def _extract_run_timing(profile: dict[str, Any]) -> dict[str, Any]:
    """Total runtime, completed step count, post-hoc warmup detection."""
    return {
        "runtime_seconds": _maybe_float(profile.get("runtime_seconds")),
        "steps_completed": _maybe_int(profile.get("steps_completed")),
        # The post-hoc stabilization output — the analysis's decision
        # on where warmup ended, which is what telemetry consumers want
        # (not what the user requested as a CLI flag).
        "warmup_steps_detected": _maybe_int(profile.get("warmup_steps_effective")),
    }


def _extract_step_time_stats(profile: dict[str, Any]) -> dict[str, Any]:
    """p50/p95/cv over the steady-state step time samples.

    Returns an empty dict when fewer than 2 valid samples are present —
    a single point doesn't define a distribution.
    """
    raw = profile.get("step_times_ms")
    if not isinstance(raw, list) or len(raw) < 2:
        return {}
    clean = [float(v) for v in raw if _is_positive(v)]
    if len(clean) < 2:
        return {}
    return {
        "step_time_p50_ms": _percentile(clean, 0.50),
        "step_time_p95_ms": _percentile(clean, 0.95),
        "step_time_cv":     _cv(clean),
    }


def _extract_gpu_stats(profile: dict[str, Any]) -> dict[str, Any]:
    """p50/p95 of GPU utilization samples (0–100)."""
    raw = profile.get("gpu_util_samples")
    if not isinstance(raw, list) or not raw:
        return {}
    clean = [float(v) for v in raw if v is not None]
    if not clean:
        return {}
    return {
        "gpu_util_p50_pct": _percentile(clean, 0.50),
        "gpu_util_p95_pct": _percentile(clean, 0.95),
    }


def _extract_memory_stats(profile: dict[str, Any], hardware) -> dict[str, Any]:
    """Peak memory in GB and as a percentage of total VRAM.

    Uses `hardware.vram_gb * hardware.gpu_count` as the denominator so
    multi-GPU runs are normalized correctly. Returns an empty dict
    when memory_peak_bytes is absent or invalid.
    """
    peak_bytes = _maybe_int(profile.get("memory_peak_bytes"))
    if not peak_bytes:
        return {}
    peak_gb = peak_bytes / (1024 ** 3)
    out: dict[str, Any] = {"memory_peak_gb": round(peak_gb, 4)}
    vram_total_gb = getattr(hardware, "vram_gb", 0.0) * getattr(hardware, "gpu_count", 1)
    if vram_total_gb > 0:
        out["memory_peak_pct"] = round(100 * peak_gb / vram_total_gb, 2)
    return out


def _extract_breakdown_pcts(profile: dict[str, Any]) -> dict[str, Any]:
    """Per-category percentages of step time.

    `dataloader_stall_pct` and `communication_pct` come pre-computed
    from the profiler. `compute_pct` is derived as the residual
    (100 − dl − comm), clamped to [0, 100]. We only emit it when at
    least one of the source pcts is present, so a profile with no
    breakdown information stays silent.
    """
    dl  = _maybe_float(profile.get("dataloader_stall_pct"))
    com = _maybe_float(profile.get("communication_overhead_pct"))
    if dl is None and com is None:
        return {}
    out: dict[str, Any] = {}
    if dl is not None:
        out["dataloader_stall_pct"] = dl
    if com is not None:
        out["communication_pct"] = com
    out["compute_pct"] = max(0.0, min(100.0, 100.0 - (dl or 0.0) - (com or 0.0)))
    return out


def _classify_primary_bottleneck(
    bottleneck_report: dict[str, Any] | None,
    stats: dict[str, Any],
) -> str | None:
    """Map the bottleneck report's boolean flags + percentages into one
    of the server-side enum values. Falls back to heuristics on the
    raw percentages when no report is available.
    """
    if bottleneck_report:
        # Schema allows a nested "bottleneck_report" key (from run-all
        # output that wraps it) or the report at the top level.
        report = bottleneck_report.get("bottleneck_report", bottleneck_report)
        for flag, label in (
            ("dataloader_bound",         "dataloader"),
            ("data_pipeline_bound",      "dataloader"),
            ("communication_bound",      "communication"),
            ("memory_bandwidth_bound",   "memory_bandwidth"),
            ("memory_capacity_bound",    "memory_capacity"),
            ("compute_bound",            "compute"),
        ):
            if report.get(flag) is True:
                return label

    # No structured report: derive from percentages we already have.
    dl = stats.get("dataloader_stall_pct") or 0.0
    comm = stats.get("communication_pct") or 0.0
    if dl >= _DATALOADER_STALL_BOTTLENECK_PCT:
        return "dataloader"
    if comm >= _COMMUNICATION_OVERHEAD_BOTTLENECK_PCT:
        return "communication"
    # We have step times but no further signal — call it compute by
    # default rather than "other," since a healthy training step that
    # isn't stalling on data/comm is compute-bound by elimination.
    if "step_time_p50_ms" in stats:
        return "compute"
    return None


# ----- math helpers (deliberately stdlib-only) ---------------------------


def _percentile(sorted_or_unsorted: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 1]. Sorts a copy."""
    if not sorted_or_unsorted:
        return 0.0
    values = sorted(sorted_or_unsorted)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _cv(values: list[float]) -> float:
    """Coefficient of variation. 0 when undefined."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    if mean <= 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (variance ** 0.5) / mean


def _is_positive(v: Any) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _maybe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _set_if(d: dict[str, Any], key: str, value: Any) -> None:
    """Only assign non-None values, so absent fields stay absent."""
    if value is not None:
        d[key] = value


def _update_nonnull(d: dict[str, Any], updates: dict[str, Any]) -> None:
    """Like dict.update but drops keys whose value is None."""
    for k, v in updates.items():
        if v is not None:
            d[k] = v


# ----- internal helpers ---------------------------------------------------


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    """Read JSON, return None on any failure. Never raises."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.debug("telemetry: could not read %s", path, exc_info=True)
        return None


def _resolve_hardware(
    profile_record: dict[str, Any] | None,
    fallback_name: str | None,
):
    """Build a stub HardwareConfig from whatever we know.

    An explicitly passed `fallback_name` wins over `profile_record.hardware_name`:
    in normal CLI use these are identical, but batch / replay callers (re-emitting
    from on-disk artifacts for a *different* GPU than the one that produced the
    profile record) need the override to land in the fingerprint. The parameter
    name predates this — it really is the authoritative hardware when the caller
    bothered to pass it.
    """
    from profine.schema.hardware import HardwareConfig, get_hardware

    name = fallback_name or (profile_record or {}).get("hardware_name")
    if not name:
        return None
    try:
        return get_hardware(name)
    except Exception:  # noqa: BLE001 — preset lookup may fail in tests
        # Caller passed a name we don't know about; we still want to
        # record something. A minimal stub keeps the fingerprint
        # functional, just with empty compute_capability (precision
        # falls back to mixed_fp16, which is safe).
        return HardwareConfig(
            name=name, label=name, modal_gpu=name,
            gpu_count=1, gpu_kind=name, vram_gb=0.0,
        )


def fingerprint_from_dict(
    arch_record: dict[str, Any],
    hardware,
) -> Fingerprint | None:
    """Public entry: build a Fingerprint from an arch_record dict + hardware.

    Useful for callers (e.g. the suggester) that already loaded the
    architecture record from disk and just need the fingerprint
    without re-running the AST extractor.
    """
    return _build_fingerprint(arch_record, hardware)


def _build_fingerprint(
    arch_record: dict[str, Any],
    hardware,
) -> Fingerprint | None:
    """Hydrate ArchitectureRecord from JSON and fingerprint it."""
    from profine.schema.architecture_record import (
        ArchitectureField,
        ArchitectureRecord,
        DistributedInfo,
        OptimizerInfo,
        PrecisionInfo,
    )

    def _wrap(value):
        if value is None:
            return None
        # Schema field shape: {"value": ..., ...} — we only need .value.
        if isinstance(value, dict) and "value" in value:
            return ArchitectureField(value=value["value"])
        return ArchitectureField(value=value)

    try:
        opt = arch_record.get("optimizer") or {}
        prec = arch_record.get("precision") or {}
        dist = arch_record.get("distributed") or {}

        record = ArchitectureRecord(
            framework=_wrap(arch_record.get("framework")),
            model_family=_wrap(arch_record.get("model_family")),
            model_class=_wrap(arch_record.get("model_class")),
            attention_type=_wrap(arch_record.get("attention_type")),
            attention_impl=_wrap(arch_record.get("attention_impl")),
            num_layers=_wrap(arch_record.get("num_layers")),
            hidden_size=_wrap(arch_record.get("hidden_size")),
            vocab_size=_wrap(arch_record.get("vocab_size")),
            num_heads=_wrap(arch_record.get("num_heads")),
            compile_mode=_wrap(arch_record.get("compile_mode")),
            gradient_checkpointing=_wrap(arch_record.get("gradient_checkpointing")),
            optimizer=OptimizerInfo(name=_wrap(opt.get("name"))) if opt else None,
            precision=PrecisionInfo(
                training_dtype=_wrap(prec.get("training_dtype")),
                autocast_enabled=_wrap(prec.get("autocast_enabled")),
                grad_scaler=_wrap(prec.get("grad_scaler")),
            ) if prec else None,
            distributed=DistributedInfo(strategy=_wrap(dist.get("strategy"))) if dist else None,
        )
        return fingerprint_run(record, hardware)
    except Exception:  # noqa: BLE001 — never raise from telemetry
        log.debug("telemetry: fingerprint construction failed", exc_info=True)
        return None


def _gather_outcomes(output_dir: Path) -> list[dict[str, Any]]:
    """Build outcome rows from the edit manifest + benchmark comparison.

    Logic, conservative:
      * Edit manifest tells us which optimizations were applied and
        which were skipped (with a reason).
      * Benchmark comparison tells us the cumulative result of the
        applied stack — we attribute it to the *first* applied entry
        (the primary recommendation). Per-entry attribution would
        require running benchmark per optimization; that's a future
        upgrade.
      * Skipped entries get an `applied=False` row with no speedup.

    Returns a list of dicts suitable for recorder.record_optimization(**row).
    """
    rows: list[dict[str, Any]] = []
    manifest = _safe_load_json(output_dir / _EDIT_DIR / _EDIT_MANIFEST)
    bench = _safe_load_json(output_dir / _BENCHMARK_DIR / _BENCH_COMPARISON)
    # Profile record gives us per-run GPU wall-clock time. We attribute it
    # to the primary outcome only, mirroring the speedup_factor convention
    # (stacked optimizations ran together; the runtime is the stack's, not
    # any single entry's).
    profile = _safe_load_json(output_dir / _PROFILE_DIR / _PROFILE_FILE)
    runtime_seconds = _maybe_float(profile.get("runtime_seconds")) if profile else None

    if manifest is None:
        return rows

    applied_ids = list(manifest.get("applied_ids") or [])
    primary_id = manifest.get("optimization_id") or (applied_ids[0] if applied_ids else None)
    skipped = manifest.get("skipped") or []

    # Bench-derived outcome attached to the primary applied id.
    if bench is not None and primary_id:
        speedup_factor = _speedup_factor_from_bench(bench)
        loss_ok = _loss_ok_from_bench(bench)
        rows.append({
            "optimization_id": primary_id,
            "applied": True,
            "speedup_factor": speedup_factor,
            "loss_ok": loss_ok,
            "crashed": False,
            "runtime_seconds": runtime_seconds,
        })

    # Applied-but-not-primary entries: we know they were applied; we
    # can't attribute speedup to them individually, so leave speedup
    # null and loss_ok null. The "applied=True with null outcome" rows
    # still help the failure-avoidance filter.
    for entry_id in applied_ids:
        if entry_id == primary_id:
            continue
        rows.append({
            "optimization_id": entry_id,
            "applied": True,
            "speedup_factor": None,
            "loss_ok": None,
            "crashed": False,
        })

    # Skipped entries — useful negative signal ("LLM rejected this
    # because it didn't apply to this script class").
    for sk in skipped:
        if not isinstance(sk, dict):
            continue
        entry_id = sk.get("entry_id")
        if not entry_id:
            continue
        rows.append({
            "optimization_id": entry_id,
            "applied": False,
            "speedup_factor": None,
            "loss_ok": None,
            "crashed": False,
        })

    return rows


def _speedup_factor_from_bench(bench: dict[str, Any]) -> float | None:
    """benchmark_comparison.json stores speedup_pct as a percentage
    relative to baseline (e.g. 35.0 means 1.35x). Convert to a factor."""
    raw = bench.get("speedup_pct")
    if raw is None:
        return None
    try:
        return 1.0 + float(raw) / 100.0
    except (TypeError, ValueError):
        return None


def _loss_ok_from_bench(bench: dict[str, Any]) -> bool | None:
    """Use the correctness check's `passed` flag as the loss_ok signal.

    The correctness sub-dict in benchmark_comparison.json has these keys:
    `passed` (bool), `loss_match`, `max_loss_diff`, `rtol`, `atol`, `notes`,
    `tolerance_widened`, `tolerance_widened_for`. Earlier versions of this
    function read `correctness.verdict`, which never existed on the
    correctness sub-dict — `verdict` lives on the top-level BenchmarkComparison.
    That bug emitted `loss_ok=None` for every row, which made the
    optimization_priors materialized view's `success_rate` column NULL for
    every (fingerprint, optimization) pair, silently breaking the suggester's
    priors-based failure-avoidance filter.
    """
    correctness = bench.get("correctness")
    if not isinstance(correctness, dict):
        return None
    passed = correctness.get("passed")
    if isinstance(passed, bool):
        return passed
    return None
