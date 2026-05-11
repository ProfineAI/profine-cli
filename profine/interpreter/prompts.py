"""LLM prompts for the Profile Interpreter (plan 4.3)."""

from __future__ import annotations

import json
from typing import Any

from profine.schema.profile_record import ProfileRecord


SYSTEM_PROMPT = """\
You are an expert ML performance engineer diagnosing training bottlenecks.

You will receive:
1. A profile record with derived metrics (step times, kernel breakdown with absolute \
ms/step, memory utilization, cost estimates, top kernels by CUDA time).
2. An architecture record describing the training script (model, optimizer, dataloader, \
precision, distributed strategy, etc.) — may not always be present.
3. Optional user preferences (hardware constraints, goals, risk tolerance).

Your job is to produce a DIAGNOSIS — where the time is going and how much headroom exists. \
You do NOT suggest fixes. You only diagnose.

## Output Format

Return ONLY valid JSON (no markdown fences) with these keys:

{
  "bottleneck_report": {
    "bottlenecks": [
      {
        "category": "attention|matmul|data_pipeline|precision|memory_bandwidth|communication|optimizer|normalization|other",
        "location": "descriptive string, e.g. 'flash_fwd kernel (41% of CUDA time)' or 'DataLoader stall between steps'",
        "time_share_pct": <float, % of total step time>,
        "est_headroom_pct": <float, estimated % end-to-end speedup if fully addressed>,
        "confidence": "observed|inferred|guessed",
        "supporting_evidence": [
          {"metric": "kernel_breakdown.attention_pct", "value": 41.2},
          {"metric": "top_kernels[0].name", "value": "flash_fwd_splitkv_kernel"}
        ],
        "notes": "optional caveats"
      }
    ],
    "compute_bound": <bool>,
    "memory_bandwidth_bound": <bool>,
    "memory_capacity_bound": <bool>,
    "data_pipeline_bound": <bool>,
    "communication_bound": <bool>,
    "summary": "2-3 sentence executive summary of where time is going",
    "time_breakdown_narrative": "detailed paragraph explaining the time distribution, citing specific numbers",
    "unstructured_notes": ["any observations that don't fit the schema"]
  },
  "markdown_report": "human-readable markdown report with the same info, including inline numbers"
}

## Rules

- Rank bottlenecks by est_headroom_pct (highest first) — the biggest wins come first.
- est_headroom_pct should be realistic. If attention is 40% of step time, replacing manual \
attention with FlashAttention-2 might save 20-30% end-to-end (not 40%, because FA2 is faster \
but not zero-cost). Use your knowledge of real-world speedups.
- Every claim must cite a specific metric from the profile (kernel time, utilization, etc.).
- If the profile data is insufficient to quantify headroom, set confidence to "guessed" and explain.
- The profile data includes pre-computed absolute times per kernel category (ms/step). \
Use these to ground your analysis in concrete numbers.
- Include at least the top 3 bottlenecks, up to 5.
- Be specific: "attention kernels use 41% of CUDA time" not "attention is slow".
- NEVER suggest optimizations. Only diagnose. The suggester tool handles recommendations.
"""


def build_interpreter_prompt(
    profile_record: ProfileRecord,
    architecture_record: dict[str, Any] | None = None,
    user_preferences: str | None = None,
) -> str:
    """Build the user message for the interpreter LLM call."""
    profile_dict = _profile_to_prompt_dict(profile_record, architecture_record)

    parts = ["## Profile Record\n"]
    parts.append(json.dumps(profile_dict, indent=2, default=str))

    if architecture_record:
        parts.append("\n\n## Architecture Record\n")
        parts.append(json.dumps(architecture_record, indent=2, default=str))

    if user_preferences:
        parts.append("\n\n## User Preferences\n")
        parts.append(user_preferences)

    return "\n".join(parts)


def _profile_to_prompt_dict(
    record: ProfileRecord,
    architecture_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert ProfileRecord to a concise dict for the LLM prompt.

    Includes computed heuristics, derived metrics (cost, memory util,
    per-category absolute times), and truncated kernel names.
    """
    from profine.schema.hardware import get_hardware

    d: dict[str, Any] = {
        "status": record.status,
        "script_path": record.script_path,
        "hardware": record.hardware_name,
        "steps_completed": record.steps_completed,
        "runtime_seconds": round(record.runtime_seconds, 1),
    }

    # Hardware context
    try:
        hw = get_hardware(record.hardware_name)
        d["hardware_vram_gb"] = hw.vram_gb
        d["hardware_cost_per_hour"] = hw.cost_per_hour
    except ValueError:
        hw = None

    # Step times
    median_ms = record.step_time_median_ms
    if median_ms:
        d["step_time_median_ms"] = round(median_ms, 1)
        d["steady_state_steps"] = len(record.step_times_ms)

    # Cost
    if hw and record.runtime_seconds > 0:
        d["cost_this_run"] = round(record.runtime_seconds * hw.cost_per_hour / 3600, 2)
    if hw and median_ms:
        d["cost_per_1k_steps"] = round(median_ms / 1000 * 1000 * hw.cost_per_hour / 3600, 2)

    # Memory
    d["memory_peak_gb"] = round(record.memory_peak_gb, 2)
    d["memory_headroom_pct"] = round(record.memory_headroom_pct, 1)
    if hw:
        d["memory_utilization_pct"] = round(record.memory_peak_gb / hw.vram_gb * 100, 1)

    # Kernel breakdown + absolute times per step
    if record.kernel_breakdown and median_ms:
        bd = record.kernel_breakdown
        d["kernel_breakdown"] = {}
        for cat in ("matmul", "attention", "elementwise", "normalization",
                     "optimizer", "communication", "memory", "dataloader", "other"):
            pct = getattr(bd, f"{cat}_pct", 0.0)
            if pct > 0.1:
                d["kernel_breakdown"][cat] = {
                    "pct": round(pct, 1),
                    "ms_per_step": round(pct * median_ms / 100),
                }

    # Top kernels (truncated names)
    if record.top_kernels:
        d["top_kernels"] = [
            {"name": k.name[:40], "category": k.category, "pct": round(k.pct_of_total, 1)}
            for k in record.top_kernels[:10]
        ]

    # Profile flags
    d["precision"] = record.precision
    d["attention_impl"] = record.attention_impl
    if record.dataloader_stall_pct > 0:
        d["dataloader_stall_pct"] = record.dataloader_stall_pct
    if record.communication_overhead_pct > 0:
        d["communication_overhead_pct"] = record.communication_overhead_pct

    return d
