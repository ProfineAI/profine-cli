"""LLM prompts for the Suggest Optimizations tool (plan 4.4).

The LLM ranks pre-filtered candidates by ROI and adds reasoning
about priority, effort, and interaction effects.
"""

from __future__ import annotations

import json
from typing import Any

from profine.catalog.schema import CatalogEntry
from profine.schema.bottleneck_report import BottleneckReport


SYSTEM_PROMPT = """\
You are an expert ML performance engineer. You receive:
1. A list of candidate optimizations that passed applicability checks
2. A bottleneck diagnosis report
3. An architecture record describing the training script

Your job is to RANK these candidates by expected ROI (impact / effort) and provide
reasoning for each. You must also identify:
- Which bottlenecks each optimization addresses
- Interaction effects (e.g., torch.compile subsumes some fused kernel benefits)
- Effort required to implement
- Risks specific to this codebase

Return ONLY valid JSON with this exact structure:
{
  "candidates": [
    {
      "entry_id": "...",
      "rank": 1,
      "priority": "critical|high|medium|low",
      "est_speedup_low_pct": 10,
      "est_speedup_high_pct": 30,
      "confidence": "high|medium|low",
      "rationale": "Why this ranks here, referencing specific bottlenecks",
      "bottlenecks_addressed": ["compute_bound", "attention"],
      "risks": ["specific risk for this codebase"],
      "estimated_effort": "trivial|small|medium|large"
    }
  ],
  "summary": "2-3 sentence executive summary of the optimization strategy",
  "total_est_speedup_low_pct": 25,
  "total_est_speedup_high_pct": 55,
  "interaction_notes": ["torch.compile may reduce gains from fused_layer_norm"],
  "unstructured_notes": ["any other observations"]
}

Rules:
- The candidate list is PRE-SORTED by deterministic ROI score
  (bottleneck match × catalog speedup ceiling). Treat the input order
  as your default ranking. Only override it when there's a concrete
  interaction effect — e.g. "torch.compile already covers fused_norm,
  so deprioritize fused_norm." Don't reshuffle on vibes.
- Account for diminishing returns when stacking optimizations.
- total_est_speedup should NOT be a simple sum — account for overlaps.
- Be conservative with speedup estimates — cite the bottleneck's
  time_share_pct as an upper bound.
- HARD CONSTRAINT: every "entry_id" you return MUST come from the
  candidate list verbatim. Entries you invent or recall from training
  data will be rejected by the parser and discarded. If no candidate
  is a good fit, return fewer candidates rather than fabricating new
  ones.
"""


def build_suggestion_prompt(
    candidates: list[tuple[CatalogEntry, float]],
    bottleneck_report: BottleneckReport | None = None,
    architecture_record: dict[str, Any] | None = None,
    user_preferences: str | None = None,
    profile_summary: dict[str, Any] | None = None,
) -> str:
    """Build the user message for the suggestion LLM call."""
    sections: list[str] = []

    # Candidate list
    candidate_data = []
    for entry, relevance in candidates:
        c: dict[str, Any] = {
            "entry_id": entry.id,
            "category": entry.category,
            "name": entry.name,
            "description": entry.description,
            "relevance_score": round(relevance, 2),
            "risks": entry.risks,
            "code_pattern": entry.code_pattern,
            "addresses_bottlenecks": entry.addresses_bottlenecks,
        }
        if entry.expected_speedup:
            c["catalog_speedup"] = {
                "kernel_low": entry.expected_speedup.kernel_low,
                "kernel_high": entry.expected_speedup.kernel_high,
                "e2e_low_pct": entry.expected_speedup.end_to_end_low_pct,
                "e2e_high_pct": entry.expected_speedup.end_to_end_high_pct,
                "depends_on": entry.expected_speedup.depends_on,
            }
        if entry.evidence:
            c["evidence"] = [{"kind": e.kind, "ref": e.ref} for e in entry.evidence]
        candidate_data.append(c)

    sections.append("## Candidate Optimizations\n```json\n"
                     + json.dumps(candidate_data, indent=2) + "\n```")

    # Bottleneck report
    if bottleneck_report:
        br: dict[str, Any] = {
            "summary": bottleneck_report.summary,
            "compute_bound": bottleneck_report.compute_bound,
            "memory_bandwidth_bound": bottleneck_report.memory_bandwidth_bound,
            "memory_capacity_bound": bottleneck_report.memory_capacity_bound,
            "data_pipeline_bound": bottleneck_report.data_pipeline_bound,
            "communication_bound": bottleneck_report.communication_bound,
            "bottlenecks": [
                {
                    "category": b.category,
                    "location": b.location,
                    "time_share_pct": b.time_share_pct,
                    "est_headroom_pct": b.est_headroom_pct,
                    "confidence": b.confidence,
                }
                for b in bottleneck_report.bottlenecks
            ],
        }
        sections.append("## Bottleneck Diagnosis\n```json\n"
                         + json.dumps(br, indent=2) + "\n```")

    # Architecture (compact)
    if architecture_record:
        compact = _compact_architecture(architecture_record)
        sections.append("## Architecture Record\n```json\n"
                         + json.dumps(compact, indent=2) + "\n```")

    if profile_summary:
        sections.append("## Profile Summary\n```json\n"
                         + json.dumps(profile_summary, indent=2) + "\n```")

    if user_preferences:
        sections.append(f"## User Preferences\n{user_preferences}")

    return "\n\n".join(sections)


def _compact_architecture(arch: dict[str, Any]) -> dict[str, Any]:
    """Extract just the values from an architecture record for a compact prompt."""
    compact: dict[str, Any] = {}
    for key, val in arch.items():
        if key in ("dependencies", "unstructured_notes", "script_path"):
            continue
        if isinstance(val, dict):
            if "value" in val:
                compact[key] = val["value"]
            else:
                # nested like optimizer, dataloader, etc.
                inner: dict[str, Any] = {}
                for k2, v2 in val.items():
                    if isinstance(v2, dict) and "value" in v2:
                        inner[k2] = v2["value"]
                if inner:
                    compact[key] = inner
    return compact
