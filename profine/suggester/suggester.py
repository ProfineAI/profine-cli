"""Suggest Optimizations orchestrator (plan 4.4).

Chains: filter catalog by applicability -> LLM ranks by ROI -> return ranked candidates.

Usage:
    from profine.suggester.suggester import OptimizationSuggester

    suggester = OptimizationSuggester(provider="openai")
    result = suggester.suggest(architecture_record, bottleneck_report)

    for c in result.report.candidates:
        print(f"#{c.rank} {c.name} — {c.rationale}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profine.catalog.entries import get_catalog
from profine.catalog.schema import CatalogEntry
from profine.llm.backend import LlmBackend, create_backend
from profine.llm.utils import call_and_parse
from profine.schema.bottleneck_report import BottleneckReport
from profine.schema.optimization_candidate import OptimizationCandidate, SuggestionReport
from profine.suggester.applicability import ApplicabilityChecker
from profine.suggester.prompts import SYSTEM_PROMPT, build_suggestion_prompt


@dataclass(slots=True)
class SuggestResult:
    """Output of the Suggest Optimizations tool."""
    report: SuggestionReport
    markdown: str
    filtered_count: int = 0      # how many entries passed applicability
    total_count: int = 0         # total catalog entries checked
    warnings: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_path = out / "suggestion_report.json"
        brief_path = out / "suggestion_brief.md"

        report_path.write_text(
            json.dumps(_report_to_dict(self.report), indent=2, default=str),
            encoding="utf-8",
        )
        brief_path.write_text(self.markdown, encoding="utf-8")
        return {"report": report_path, "brief": brief_path}


class OptimizationSuggester:
    """LLM-driven optimization suggester."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        self._backend = create_backend(provider, **kwargs)

    def suggest(
        self,
        architecture_record: dict[str, Any],
        bottleneck_report: BottleneckReport | None = None,
        user_preferences: str | None = None,
        profile_summary: dict[str, Any] | None = None,
        *,
        debug_dir: str | Path | None = None,
    ) -> SuggestResult:
        """Suggest optimizations for a training script.

        Args:
            architecture_record: Output from the Read Code tool.
            bottleneck_report: Output from the Profile Interpreter (optional but recommended).
            user_preferences: Free-form user preferences (optional).

        Returns:
            SuggestResult with ranked candidates.
        """
        catalog = get_catalog()
        checker = ApplicabilityChecker(architecture_record, bottleneck_report)
        filtered = checker.filter_catalog(catalog)

        warnings: list[str] = []
        if not filtered:
            warnings.append("No catalog entries matched the architecture record")
            return SuggestResult(
                report=SuggestionReport(summary="No applicable optimizations found."),
                markdown="## No Applicable Optimizations\n\nNo catalog entries matched.",
                filtered_count=0,
                total_count=len(catalog),
                warnings=warnings,
            )

        # Build LLM prompt with (entry, relevance) pairs
        candidates_for_prompt = [(entry, relevance) for entry, relevance, _ in filtered]
        user_msg = build_suggestion_prompt(
            candidates_for_prompt, bottleneck_report, architecture_record,
            user_preferences, profile_summary,
        )

        parsed = call_and_parse(
            self._backend, SYSTEM_PROMPT, user_msg,
            debug_dir=debug_dir, debug_label="suggester_response",
        )
        report = _parse_response(parsed, filtered)

        # Build markdown (include unranked applicable entries)
        ranked_ids = {c.entry_id for c in report.candidates}
        unranked = [
            (entry, relevance) for entry, relevance, _ in filtered
            if entry.id not in ranked_ids
        ]
        markdown = _generate_markdown(report, unranked)

        return SuggestResult(
            report=report,
            markdown=markdown,
            filtered_count=len(filtered),
            total_count=len(catalog),
            warnings=warnings,
        )


def _parse_response(
    parsed: dict[str, Any],
    filtered: list[tuple[CatalogEntry, float, list[str]]],
) -> SuggestionReport:
    """Materialise a SuggestionReport from a parsed JSON dict."""
    # Reject anything not in the deterministic-filter output: the LLM
    # otherwise reintroduces entries it was told to skip.
    entry_map = {entry.id: entry for entry, _, _ in filtered}
    roi_map = {entry.id: score for entry, score, _ in filtered}

    candidates: list[tuple[float, int, OptimizationCandidate]] = []
    rejected: list[str] = []
    for c in parsed.get("candidates", []):
        entry_id = c.get("entry_id", "")
        entry = entry_map.get(entry_id)
        if entry is None:
            rejected.append(entry_id or "<missing entry_id>")
            continue

        candidates.append((
            roi_map.get(entry_id, 0.0),
            c.get("rank", 999),
            OptimizationCandidate(
                entry_id=entry_id,
                category=entry.category,
                name=entry.name,
                description=entry.description,
                rank=0,
                priority=c.get("priority", "medium"),
                est_speedup_low_pct=c.get("est_speedup_low_pct", 0.0),
                est_speedup_high_pct=c.get("est_speedup_high_pct", 0.0),
                confidence=c.get("confidence", "medium"),
                rationale=c.get("rationale", ""),
                bottlenecks_addressed=c.get("bottlenecks_addressed", []),
                risks=c.get("risks", entry.risks),
                code_pattern=entry.code_pattern,
                estimated_effort=c.get("estimated_effort", ""),
                evidence=[{"kind": e.kind, "ref": e.ref, "url": e.url}
                          for e in entry.evidence],
                exclusive_group=entry.exclusive_group,
            ),
        ))

    # Deterministic ROI is the load-bearing signal; LLM rank breaks ties.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    final_candidates = []
    for new_rank, (_, _, cand) in enumerate(candidates, start=1):
        cand.rank = new_rank
        final_candidates.append(cand)

    warnings = list(parsed.get("warnings", []))
    if rejected:
        warnings.append(
            "Dropped LLM-suggested entries that were not in the filtered "
            f"candidate list (likely hallucinated): {sorted(set(rejected))}"
        )

    return SuggestionReport(
        candidates=final_candidates,
        summary=parsed.get("summary", ""),
        total_est_speedup_low_pct=parsed.get("total_est_speedup_low_pct", 0.0),
        total_est_speedup_high_pct=parsed.get("total_est_speedup_high_pct", 0.0),
        warnings=warnings,
        unstructured_notes=parsed.get("unstructured_notes", []),
    )


def _generate_markdown(
    report: SuggestionReport,
    unranked: list[tuple[CatalogEntry, float]] | None = None,
) -> str:
    """Generate a human-readable markdown report."""
    lines: list[str] = ["# Optimization Suggestions", ""]

    if report.summary:
        lines += [report.summary, ""]

    if report.total_est_speedup_low_pct or report.total_est_speedup_high_pct:
        lines.append(f"**Estimated total speedup: {report.total_est_speedup_low_pct:.0f}%"
                      f" - {report.total_est_speedup_high_pct:.0f}%**")
        lines.append("")

    lines.append("---")
    lines.append("")

    for c in report.candidates:
        priority_badge = {"critical": "!!!", "high": "!!", "medium": "!", "low": ""}.get(c.priority, "")
        excl_badge = f" (excl:{c.exclusive_group})" if c.exclusive_group else ""
        lines.append(f"### #{c.rank} {c.name} [{c.category}]{excl_badge} {priority_badge}")
        lines.append("")
        lines.append(f"**Priority:** {c.priority} | "
                      f"**Speedup:** {c.est_speedup_low_pct:.0f}%-{c.est_speedup_high_pct:.0f}% | "
                      f"**Effort:** {c.estimated_effort} | "
                      f"**Confidence:** {c.confidence}")
        lines.append("")

        if c.rationale:
            lines.append(f"> {c.rationale}")
            lines.append("")

        if c.bottlenecks_addressed:
            lines.append(f"**Addresses:** {', '.join(c.bottlenecks_addressed)}")
            lines.append("")

        if c.risks:
            lines.append("**Risks:**")
            for r in c.risks:
                lines.append(f"- {r}")
            lines.append("")

        if c.code_pattern:
            lines.append(f"**Implementation:** `{c.code_pattern}`")
            lines.append("")

        lines.append("---")
        lines.append("")

    if unranked:
        lines.append("## Also Applicable")
        lines.append("")
        for entry, relevance in sorted(unranked, key=lambda x: -x[1]):
            lines.append(f"- **{entry.name}** [{entry.category}] — {entry.description[:80]}...")
        lines.append("")

    if report.unstructured_notes:
        lines.append("## Notes")
        for note in report.unstructured_notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def _report_to_dict(report: SuggestionReport) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(report)
