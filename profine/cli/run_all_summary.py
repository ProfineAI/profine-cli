"""Build a consolidated SUMMARY.md after a successful `profine run-all` invocation.

This is the one file a user (or reviewer) should read after the pipeline finishes —
it captures the architecture, bottleneck, optimizations applied, and benchmark
verdict in a single page, with absolute paths to the per-step artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_summary(output_dir: Path, script: str, hardware: str) -> Path | None:
    """Compose SUMMARY.md from per-step JSON artifacts in `output_dir`.

    Returns the written path, or None if no artifacts were found.
    """
    sections: list[str] = []
    sections.append(f"# profine run-all — {script}")
    sections.append("")
    sections.append(f"**Hardware:** `{hardware}`")
    sections.append("")

    arch = _load_json(output_dir / "read" / "architecture_record.json")
    bottleneck = _load_json(output_dir / "interpret" / "bottleneck_report.json")
    suggestion = _load_json(output_dir / "suggest" / "suggestion_report.json")
    change_manifest = _load_json(output_dir / "edit" / "change_manifest.json")
    comparison = _load_json(output_dir / "benchmark" / "benchmark_comparison.json")

    if not any([arch, bottleneck, suggestion, change_manifest, comparison]):
        return None

    # Headline TL;DR — pull directly from the benchmark comparison.
    if comparison:
        sections.append(_headline(comparison))
        sections.append("")

    if arch:
        sections.append(_architecture_section(arch))
        sections.append("")

    if bottleneck:
        sections.append(_bottleneck_section(bottleneck))
        sections.append("")

    if suggestion or change_manifest:
        sections.append(_optimizations_section(suggestion, change_manifest))
        sections.append("")

    if comparison:
        sections.append(_benchmark_section(comparison))
        sections.append("")

    sections.append(_artifacts_index(output_dir))
    sections.append("")

    summary_path = output_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(sections), encoding="utf-8")
    return summary_path


def _load_json(p: Path) -> Any | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _headline(comparison: dict) -> str:
    speedup = comparison.get("speedup_pct", 0.0)
    correctness_passed = (comparison.get("correctness") or {}).get("passed", True)
    if speedup >= 3.0 and correctness_passed:
        mult = 100.0 / (100.0 - speedup) if speedup < 100 else float("inf")
        return f"## ✅ {speedup:.1f}% faster ({mult:.2f}× speedup), correctness preserved."
    if speedup >= 3.0 and not correctness_passed:
        return f"## ⚠️ {speedup:.1f}% faster, but loss curves diverge — review before shipping."
    if speedup <= -2.0:
        return f"## ❌ {abs(speedup):.1f}% regression — do not ship."
    return f"## ➖ No meaningful change ({speedup:+.1f}%)."


def _architecture_section(arch: dict) -> str:
    lines = ["## Architecture (what we found in your code)", ""]
    # Surface the most-asked fields. Fall back gracefully if missing.
    interesting = [
        ("model_family", "Model"),
        ("framework", "Framework"),
        ("precision", "Precision"),
        ("optimizer", "Optimizer"),
        ("distributed_strategy", "Distributed"),
        ("dataloader", "Dataloader"),
        ("attention_implementation", "Attention impl"),
    ]
    rows: list[str] = []
    for key, label in interesting:
        val = arch.get(key)
        if not val:
            continue
        if isinstance(val, dict):
            val = val.get("value", val)
        rows.append(f"- **{label}:** {val}")
    if not rows:
        rows.append("_(no structured architecture fields parsed)_")
    return "\n".join(lines + rows)


def _bottleneck_section(payload: dict) -> str:
    # `payload` can be either the merged {profile_summary, bottleneck_report} envelope
    # or the bottleneck_report directly.
    report = payload.get("bottleneck_report", payload)
    lines = ["## Bottleneck (what's slowing it down)", ""]
    bottlenecks = report.get("bottlenecks") or report.get("primary_bottlenecks") or []
    if bottlenecks and isinstance(bottlenecks, list):
        for b in bottlenecks[:3]:
            if isinstance(b, dict):
                name = b.get("name") or b.get("category") or "bottleneck"
                desc = b.get("description") or b.get("evidence") or ""
                lines.append(f"- **{name}** — {desc}".rstrip(" —"))
            else:
                lines.append(f"- {b}")
    elif report.get("summary"):
        lines.append(report["summary"])
    else:
        lines.append("_(no bottleneck data — see interpret/ for details)_")
    return "\n".join(lines)


def _optimizations_section(suggestion: dict | None, change_manifest: dict | None) -> str:
    lines = ["## Optimizations", ""]

    # Ranked candidates from suggest step
    candidates = (suggestion or {}).get("candidates") or []
    if candidates:
        lines.append("**Ranked by LLM ROI:**")
        lines.append("")
        for i, c in enumerate(candidates[:5], 1):
            entry_id = c.get("entry_id") or c.get("id") or "?"
            reason = c.get("rationale") or c.get("reason") or ""
            short = reason.split(".")[0][:160] if reason else ""
            lines.append(f"{i}. `{entry_id}` — {short}".rstrip(" —"))
        lines.append("")

    # What actually got applied (from change_manifest)
    applied: list[str] = []
    skipped: list[str] = []
    if change_manifest:
        for item in change_manifest.get("applied", []):
            if isinstance(item, dict):
                applied.append(item.get("entry_id") or item.get("id") or "")
            else:
                applied.append(str(item))
        for item in change_manifest.get("skipped", []):
            if isinstance(item, dict):
                reason = item.get("reason", "")
                eid = item.get("entry_id") or item.get("id") or ""
                skipped.append(f"`{eid}` ({reason})" if reason else f"`{eid}`")
            else:
                skipped.append(str(item))

    if applied:
        lines.append(f"**Applied ({len(applied)}):** " + ", ".join(f"`{x}`" for x in applied if x))
    if skipped:
        lines.append(f"**Skipped ({len(skipped)}):** " + ", ".join(skipped))
    if not applied and not skipped and not candidates:
        lines.append("_(no optimization data)_")

    return "\n".join(lines)


def _benchmark_section(comparison: dict) -> str:
    speedup = comparison.get("speedup_pct", 0.0)
    mem_delta = comparison.get("memory_delta_pct", 0.0)
    util_delta = comparison.get("util_delta_pct", 0.0)
    verdict = comparison.get("verdict", "?")
    correctness = comparison.get("correctness") or {}

    lines = ["## Benchmark (measured on-GPU)", ""]
    lines.append("| Metric | Δ |")
    lines.append("|---|---|")
    lines.append(f"| Step time | **{speedup:+.1f}%** ({_speedup_mult(speedup)}) |")
    lines.append(f"| Peak memory | {mem_delta:+.1f}% |")
    lines.append(f"| GPU utilization | {util_delta:+.1f}% |")
    lines.append(f"| Verdict | **{verdict}** |")
    lines.append(f"| Correctness | {'✓ pass' if correctness.get('passed', True) else '✗ fail'} |")
    return "\n".join(lines)


def _speedup_mult(pct: float) -> str:
    if pct <= 0 or pct >= 100:
        return "—"
    return f"{100.0 / (100.0 - pct):.2f}× faster"


def _artifacts_index(output_dir: Path) -> str:
    lines = ["## Artifacts", ""]
    candidates = [
        ("read/architecture_record.json", "Parsed architecture (JSON)"),
        ("read/architecture_brief.md", "Architecture brief (MD)"),
        ("profile/profile_record.json", "Profile data (JSON)"),
        ("profile/profile_report.md", "Profile report (MD)"),
        ("interpret/bottleneck_report.json", "Bottleneck diagnosis (JSON)"),
        ("interpret/bottleneck_brief.md", "Bottleneck brief (MD)"),
        ("suggest/suggestion_report.json", "Ranked optimizations (JSON)"),
        ("suggest/suggestion_brief.md", "Suggestion brief (MD)"),
        ("edit/edited_train.py", "Optimized training script"),
        ("edit/change_manifest.json", "What was changed and why"),
        ("benchmark/benchmark_comparison.json", "Benchmark comparison (JSON)"),
        ("benchmark/benchmark_report.md", "Benchmark report (MD)"),
    ]
    for rel, label in candidates:
        p = output_dir / rel
        if p.exists():
            lines.append(f"- [`{rel}`]({rel}) — {label}")
    return "\n".join(lines)
