"""Profile Interpreter orchestrator (plan 4.3).

Takes a ProfileRecord + ArchitectureRecord and produces a BottleneckReport
with ranked bottlenecks and quantified headroom.

Usage:
    from profine.interpreter.interpreter import ProfileInterpreter

    interpreter = ProfileInterpreter(provider="openai")
    result = interpreter.interpret(profile_record, architecture_record)

    print(result.markdown)
    print(result.report.bottlenecks[0].category)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profine.interpreter.prompts import SYSTEM_PROMPT, build_interpreter_prompt
from profine.llm.backend import LlmBackend, create_backend
from profine.llm.utils import call_and_parse
from profine.schema.bottleneck_report import BottleneckEntry, BottleneckReport
from profine.schema.hardware import get_hardware
from profine.schema.profile_record import ProfileRecord


@dataclass(slots=True)
class InterpretResult:
    """Output of the Profile Interpreter."""
    report: BottleneckReport
    profile_summary: dict[str, Any]
    markdown: str
    warnings: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_path = out / "bottleneck_report.json"
        brief_path = out / "bottleneck_brief.md"

        # Merge deterministic summary + LLM bottleneck report
        full_report = {
            "profile_summary": self.profile_summary,
            "bottleneck_report": _report_to_dict(self.report),
        }
        report_path.write_text(
            json.dumps(full_report, indent=2, default=str),
            encoding="utf-8",
        )
        brief_path.write_text(self.markdown, encoding="utf-8")
        return {"report": report_path, "brief": brief_path}


class ProfileInterpreter:
    """LLM-driven profile interpreter."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        seed: int | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        if seed is not None:
            kwargs["seed"] = seed
        self._backend = create_backend(provider, **kwargs)

    def interpret(
        self,
        profile_record: ProfileRecord,
        architecture_record: dict[str, Any] | None = None,
        user_preferences: str | None = None,
        *,
        debug_dir: str | Path | None = None,
    ) -> InterpretResult:
        """Interpret a profile and produce a bottleneck diagnosis.

        Args:
            profile_record: Output from the Profiler tool.
            architecture_record: Output from the Read Code tool (optional but recommended).
            user_preferences: Free-form user preferences markdown (optional).
            debug_dir: If set, malformed LLM responses dump here for inspection.

        Returns:
            InterpretResult with .report, .profile_summary, and .markdown.
        """
        # Step 1: Deterministic analysis (no LLM, pure math)
        summary = _compute_deterministic(profile_record, architecture_record)

        # Step 2: LLM analysis (judgment, narrative)
        user_msg = build_interpreter_prompt(
            profile_record, architecture_record, user_preferences,
        )
        parsed = call_and_parse(
            self._backend, SYSTEM_PROMPT, user_msg,
            debug_dir=debug_dir, debug_label="interpreter_response",
        )
        report, llm_markdown = _parse_response(parsed)

        # Step 3: Merge into final markdown
        markdown = _build_merged_markdown(summary, report, llm_markdown)

        warnings = []
        for b in report.bottlenecks:
            if b.confidence == "guessed":
                warnings.append(f"Bottleneck '{b.category}' at '{b.location}' is a guess: {b.notes}")

        return InterpretResult(
            report=report,
            profile_summary=summary,
            markdown=markdown,
            warnings=warnings,
        )


def _compute_deterministic(
    record: ProfileRecord,
    architecture_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure-math analysis of profile data. No LLM needed."""
    summary: dict[str, Any] = {}

    # Hardware context
    try:
        hw = get_hardware(record.hardware_name)
        summary["hardware"] = {
            "gpu": hw.gpu_kind,
            "gpu_count": hw.gpu_count,
            "vram_gb": hw.vram_gb,
            "cost_per_hour": hw.cost_per_hour,
        }
    except ValueError:
        hw = None
        summary["hardware"] = {"name": record.hardware_name}

    # Step timing
    median_ms = record.step_time_median_ms
    summary["step_time"] = {
        "median_ms": median_ms,
        "median_s": round(median_ms / 1000, 2) if median_ms else None,
        "steady_state_steps": len(record.step_times_ms),
    }

    # Cost
    if hw and median_ms:
        cost_per_hr = hw.cost_per_hour
        summary["cost"] = {
            "this_run": round(record.runtime_seconds * cost_per_hr / 3600, 2),
            "per_1k_steps": round(median_ms / 1000 * 1000 * cost_per_hr / 3600, 2),
        }

    # Memory
    peak_gb = record.memory_peak_gb
    total_vram = hw.vram_gb if hw else 0
    summary["memory"] = {
        "peak_gb": round(peak_gb, 2),
        "total_vram_gb": total_vram,
        "utilization_pct": round(peak_gb / total_vram * 100, 1) if total_vram > 0 else None,
        "headroom_pct": round(record.memory_headroom_pct, 1),
    }

    # Per-category absolute time (ms per step)
    if record.kernel_breakdown and median_ms:
        bd = record.kernel_breakdown
        summary["kernel_time_ms_per_step"] = {
            "matmul": round(bd.matmul_pct * median_ms / 100),
            "attention": round(bd.attention_pct * median_ms / 100),
            "elementwise": round(bd.elementwise_pct * median_ms / 100),
            "normalization": round(bd.normalization_pct * median_ms / 100),
            "optimizer": round(bd.optimizer_pct * median_ms / 100),
            "communication": round(bd.communication_pct * median_ms / 100),
            "memory": round(bd.memory_pct * median_ms / 100),
            "other": round(bd.other_pct * median_ms / 100),
        }

    # Top kernels (truncated names)
    if record.top_kernels:
        summary["top_kernels"] = [
            {
                "name": k.name[:40],
                "category": k.category,
                "pct": round(k.pct_of_total, 1),
                "time_ms": round(k.cuda_time_us / 1000, 1),
            }
            for k in record.top_kernels[:10]
        ]

    # Profile flags
    summary["profile_flags"] = {
        "precision": record.precision,
        "attention_impl": record.attention_impl,
        "communication_overhead_pct": record.communication_overhead_pct,
        "dataloader_stall_pct": record.dataloader_stall_pct,
    }

    # Throughput (if architecture record has batch_size + context_length)
    if architecture_record and median_ms:
        batch_size = _arch_val(architecture_record, "dataloader", "batch_size")
        seq_len = _arch_val(architecture_record, "context_length")
        if batch_size and median_ms > 0:
            step_s = median_ms / 1000
            summary["throughput"] = {"samples_per_sec": round(batch_size / step_s, 3)}
            if seq_len:
                summary["throughput"]["tokens_per_sec"] = round(batch_size * seq_len / step_s, 1)

    return summary


def _arch_val(arch: dict[str, Any], *keys: str) -> Any:
    """Extract a value from an architecture record, handling the {value, confidence} wrapper."""
    obj = arch
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
        if obj is None:
            return None
    if isinstance(obj, dict) and "value" in obj:
        return obj["value"]
    return obj


def _build_merged_markdown(
    summary: dict[str, Any],
    report: BottleneckReport,
    llm_markdown: str,
) -> str:
    """Combine deterministic analysis + LLM narrative into one markdown document."""
    lines: list[str] = []

    # Deterministic: Hardware & Cost
    hw = summary.get("hardware", {})
    lines.append(f"# Performance Analysis")
    lines.append("")
    if "gpu" in hw:
        lines.append(f"**Hardware**: {hw.get('gpu_count', 1)}x {hw['gpu']} ({hw.get('vram_gb', '?')} GB) @ ${hw.get('cost_per_hour', '?')}/hr")
    cost = summary.get("cost", {})
    if cost:
        lines.append(f"**Cost**: ${cost['this_run']} this run | ${cost['per_1k_steps']}/1K steps")

    # Deterministic: Step Time & Memory
    step = summary.get("step_time", {})
    mem = summary.get("memory", {})
    if step.get("median_s"):
        lines.append(f"**Step time**: {step['median_s']}s (median, {step['steady_state_steps']} steady-state steps)")
    if mem.get("utilization_pct"):
        lines.append(f"**Memory**: {mem['peak_gb']} GB / {mem['total_vram_gb']} GB ({mem['utilization_pct']}% used, {mem['headroom_pct']}% headroom)")
    lines.append("")

    # Deterministic: Kernel time breakdown
    kernel_times = summary.get("kernel_time_ms_per_step", {})
    if kernel_times:
        lines.append("## Time per Step by Category")
        lines.append("")
        lines.append("| Category | ms/step | % |")
        lines.append("|---|---|---|")
        bd = {k: v for k, v in kernel_times.items() if v > 0}
        for cat, ms in sorted(bd.items(), key=lambda x: -x[1]):
            pct = ms / (step.get("median_ms") or 1) * 100
            lines.append(f"| {cat} | {ms:,} | {pct:.1f}% |")
        lines.append("")

    # Deterministic: Top kernels
    top_k = summary.get("top_kernels", [])
    if top_k:
        lines.append("## Top Kernels")
        lines.append("")
        lines.append("| Kernel | Category | % | Time (ms) |")
        lines.append("|---|---|---|---|")
        for k in top_k:
            lines.append(f"| {k['name']} | {k['category']} | {k['pct']}% | {k['time_ms']:,.1f} |")
        lines.append("")

    # Deterministic: Flags
    flags = summary.get("profile_flags", {})
    lines.append("## Profile Flags")
    lines.append("")
    lines.append(f"- **Precision**: {flags.get('precision', 'unknown')}")
    lines.append(f"- **Attention**: {flags.get('attention_impl', 'unknown')}")
    if flags.get("dataloader_stall_pct", 0) > 0:
        lines.append(f"- **DataLoader stall**: {flags['dataloader_stall_pct']:.1f}%")
    if flags.get("communication_overhead_pct", 0) > 0:
        lines.append(f"- **Communication overhead**: {flags['communication_overhead_pct']:.1f}%")

    # Throughput
    tp = summary.get("throughput", {})
    if tp:
        parts = []
        if "tokens_per_sec" in tp:
            parts.append(f"{tp['tokens_per_sec']:,.1f} tokens/sec")
        if "samples_per_sec" in tp:
            parts.append(f"{tp['samples_per_sec']:.3f} samples/sec")
        if parts:
            lines.append(f"- **Throughput**: {' | '.join(parts)}")
    lines.append("")

    # LLM: Bottleneck diagnosis
    lines.append("## Bottleneck Diagnosis (LLM)")
    lines.append("")
    lines.append(llm_markdown)

    return "\n".join(lines)


def _parse_response(parsed: dict[str, Any]) -> tuple[BottleneckReport, str]:
    """Materialise (BottleneckReport, markdown) from a parsed JSON dict."""
    report_dict = parsed.get("bottleneck_report", {})
    markdown = parsed.get("markdown_report", "")

    entries: list[BottleneckEntry] = []
    for b in report_dict.get("bottlenecks", []):
        entries.append(BottleneckEntry(
            category=b.get("category", "unknown"),
            location=b.get("location", ""),
            time_share_pct=b.get("time_share_pct", 0.0),
            est_headroom_pct=b.get("est_headroom_pct", 0.0),
            confidence=b.get("confidence", "inferred"),
            supporting_evidence=b.get("supporting_evidence", []),
            notes=b.get("notes", ""),
        ))

    report = BottleneckReport(
        bottlenecks=entries,
        compute_bound=report_dict.get("compute_bound", False),
        memory_bandwidth_bound=report_dict.get("memory_bandwidth_bound", False),
        memory_capacity_bound=report_dict.get("memory_capacity_bound", False),
        data_pipeline_bound=report_dict.get("data_pipeline_bound", False),
        communication_bound=report_dict.get("communication_bound", False),
        summary=report_dict.get("summary", ""),
        time_breakdown_narrative=report_dict.get("time_breakdown_narrative", ""),
        unstructured_notes=report_dict.get("unstructured_notes", []),
    )

    return report, markdown


def _report_to_dict(report: BottleneckReport) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(report)
