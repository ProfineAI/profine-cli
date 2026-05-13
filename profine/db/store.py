"""JSON file-based knowledge database.

Stores run history and appends benchmark evidence back to catalog entries.
All data lives under a .profine/ directory in the project root.

Usage:
    from profine.db.store import KnowledgeDB

    db = KnowledgeDB("path/to/project")
    db.save_run(run_record)
    db.append_evidence("flash_attention_2", evidence_entry)
    history = db.get_run_history()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from profine.catalog.schema import EvidenceEntry
from profine.db.run_record import OptimizationAttempt, RunRecord


class KnowledgeDB:
    """File-based knowledge database under .profine/ in the project root."""

    def __init__(self, project_root: str | Path) -> None:
        self._root = Path(project_root) / ".profine"
        self._runs_dir = self._root / "runs"
        self._evidence_dir = self._root / "evidence"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, record: RunRecord) -> Path:
        """Save a run record to disk. Auto-generates run_id if empty."""
        if not record.run_id:
            record.run_id = _generate_run_id(record)

        path = self._runs_dir / f"{record.run_id}.json"
        path.write_text(
            json.dumps(asdict(record), indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def get_run_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Load recent run records, newest first."""
        files = sorted(self._runs_dir.glob("*.json"), reverse=True)
        records: list[dict[str, Any]] = []
        for f in files[:limit]:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        return records

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Load a single run record by ID."""
        path = self._runs_dir / f"{run_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def append_evidence(self, optimization_id: str, evidence: EvidenceEntry) -> Path:
        """Append a benchmark evidence entry for an optimization.

        This is how the system learns: the benchmarker records "we tried
        optimization X on architecture Y and got Z% speedup (or failed)."
        """
        evidence_file = self._evidence_dir / f"{optimization_id}.json"

        entries: list[dict[str, Any]] = []
        if evidence_file.exists():
            entries = json.loads(evidence_file.read_text(encoding="utf-8"))

        entries.append(asdict(evidence))
        evidence_file.write_text(
            json.dumps(entries, indent=2, default=str),
            encoding="utf-8",
        )
        return evidence_file

    def get_evidence(self, optimization_id: str) -> list[dict[str, Any]]:
        """Get all accumulated evidence for an optimization."""
        evidence_file = self._evidence_dir / f"{optimization_id}.json"
        if evidence_file.exists():
            return json.loads(evidence_file.read_text(encoding="utf-8"))
        return []

    def get_all_evidence(self) -> dict[str, list[dict[str, Any]]]:
        """Get evidence for all optimizations."""
        result: dict[str, list[dict[str, Any]]] = {}
        for f in self._evidence_dir.glob("*.json"):
            opt_id = f.stem
            result[opt_id] = json.loads(f.read_text(encoding="utf-8"))
        return result

    def build_run_record(
        self,
        script_path: str,
        hardware: str,
        architecture_record: dict[str, Any] | None = None,
        profile_summary: dict[str, Any] | None = None,
        bottleneck_summary: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Create a RunRecord pre-filled with pipeline context."""
        return RunRecord(
            script_path=script_path,
            hardware=hardware,
            architecture_summary=_compact(architecture_record) if architecture_record else {},
            profile_summary=profile_summary or {},
            bottleneck_summary=bottleneck_summary or {},
        )

    def record_attempt(
        self,
        run_record: RunRecord,
        optimization_id: str,
        optimization_name: str,
        applied: bool,
        speedup_pct: float = 0.0,
        correctness_passed: bool = True,
        failure_reason: str = "",
    ) -> None:
        """Add an optimization attempt to a run record."""
        run_record.attempts.append(OptimizationAttempt(
            optimization_id=optimization_id,
            optimization_name=optimization_name,
            applied=applied,
            speedup_pct=speedup_pct,
            correctness_passed=correctness_passed,
            failure_reason=failure_reason,
        ))

        outcome = f"{speedup_pct:+.1f}% speedup" if applied else f"not applied: {failure_reason}"
        if not correctness_passed:
            outcome = f"correctness FAILED — {failure_reason}"

        self.append_evidence(optimization_id, EvidenceEntry(
            kind="run",
            ref=run_record.run_id or "current_run",
            outcome=f"{outcome} on {run_record.hardware}",
        ))


def _generate_run_id(record: RunRecord) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha256(f"{record.script_path}{ts}".encode()).hexdigest()[:8]
    return f"run_{ts}_{h}"


def _compact(arch: dict[str, Any]) -> dict[str, Any]:
    """Extract just values from an architecture record for compact storage."""
    compact: dict[str, Any] = {}
    for key, val in arch.items():
        if key in ("dependencies", "unstructured_notes"):
            continue
        if isinstance(val, dict):
            if "value" in val:
                compact[key] = val["value"]
            else:
                inner = {k2: v2["value"] for k2, v2 in val.items()
                         if isinstance(v2, dict) and "value" in v2}
                if inner:
                    compact[key] = inner
        else:
            compact[key] = val
    return compact
