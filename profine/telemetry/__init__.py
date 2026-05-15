"""Telemetry: anonymous, bucketed run data for the optimization prior.

Three layers:

  1. fingerprint.py — turns an ArchitectureRecord + HardwareConfig into a
     small, bucketed feature dict plus a stable sha256 hash.
  2. crash_class.py — maps a raw failure into one of a few enum-like
     classes (oom, compile_fail, ...) so failure rates are aggregatable.
  3. fields.py — the explicit allowlist of what is ever sent. Anything
     not on the allowlist cannot leave the process.

The module deliberately has no side effects on import; everything is
pure data transformation. The write path that calls the backend lives
elsewhere (see profine/telemetry/recorder.py, added in a later phase).
"""

# Tamper-evidence: hashes the telemetry .py files at import time
# against the baked manifest. Mismatch → one stderr warning, never
# raises. Bypassable, but visible. See _integrity.py for the rationale.
from profine.telemetry import _integrity as _integrity  # noqa: F401

from profine.telemetry.crash_class import classify_crash
from profine.telemetry.emit import emit_run, fingerprint_from_dict
from profine.telemetry.fingerprint import (
    Fingerprint,
    arch_class_of,
    fingerprint_run,
    optimizer_class_of,
    param_bucket_of,
    precision_of,
)
from profine.telemetry.priors import OptimizationPrior, PriorsClient
from profine.telemetry.recorder import TelemetryRecorder

__all__ = [
    "Fingerprint",
    "OptimizationPrior",
    "PriorsClient",
    "TelemetryRecorder",
    "fingerprint_run",
    "fingerprint_from_dict",
    "arch_class_of",
    "optimizer_class_of",
    "param_bucket_of",
    "precision_of",
    "classify_crash",
    "emit_run",
]
