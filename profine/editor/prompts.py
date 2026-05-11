"""LLM prompts for the Code Editor tool (plan 4.5).

The LLM receives the original source, an optimization candidate, and the
architecture record, then produces the edited source + a change manifest.
"""

from __future__ import annotations

import json
from typing import Any

from profine.schema.optimization_candidate import OptimizationCandidate


SYSTEM_PROMPT = """\
You are an expert ML systems engineer. Your job is to apply a SINGLE optimization
to a PyTorch project. You will receive:

1. The entry script (with line numbers)
2. Other local project modules the entry script imports (also with line numbers).
   The training loop, model definition, or attention may live in these modules
   rather than the entry script.
3. The optimization to apply (name, description, code pattern, risks)
4. The architecture record describing the codebase

Rules:
- Apply ONLY the requested optimization. Do NOT refactor, clean up, or "improve" other code.
- Do NOT change hyperparameters (lr, weight decay, batch size, etc.) unless the optimization
  specifically requires it (e.g., enabling larger batch via gradient checkpointing).
- Do NOT change the model architecture, quantization, or loss function.
- Preserve all existing functionality — the project must produce identical training behavior.
- Add a brief comment near each change citing the optimization (e.g., "# profine: flash_attention_2").
- If the optimization requires new imports, add them at the top of the relevant file.
- Edit whichever file(s) actually need changing. If the entry script delegates the
  relevant work (training loop, forward/backward, attention, dataloading, optimizer
  setup) to an imported local module that appears under "Local Modules", edit that
  module — not the entry script — by returning a `file_edits` entry for it.
  Returning "not applicable" because the relevant code is in another file is WRONG
  when that file is provided in "Local Modules".
- Only return entries for files you actually modified. Do not echo unchanged files.
- HARD RULE — minimal edits: each edited file (the entry script and every file_edits
  entry) must be IDENTICAL to its input EXCEPT for the lines required by the
  optimization. Do NOT rename identifiers, reformat, reorder code, drop comments or
  docstrings, or "clean up" anything. Do NOT inline functions or classes from a Local
  Module into the entry script — Local Modules stay as separate files; you edit them
  in place via `file_edits`. Inlining is a common failure mode and is FORBIDDEN.
- If the optimization can be applied entirely inside Local Modules (e.g. autocast
  wrapping in a `Trainer.run()` style class, attention rewrite in a model file), the
  entry script's `edited_source` MUST be byte-identical to its input. The only
  legitimate entry-script changes alongside `file_edits` are top-level additions like
  a new import or a single config flag.
- If the optimization is genuinely not applicable across all provided files (e.g.
  the model already uses the optimized path, or no relevant code exists), explain why in
  "not_applicable_reason" and return no edits.

Return ONLY valid JSON with this exact structure (paths shown are illustrative —
use the actual project-relative paths from "Local Modules"):
{
  "edited_source": "...full modified source of the ENTRY script (or the original if unchanged)...",
  "applied": true,
  "file_edits": [
    {
      "path": "<project-relative path of an edited Local Module>",
      "edited_source": "...full modified source of that file..."
    }
  ],
  "changes": [
    {
      "path": "<entry script or a Local Module path>",
      "line_start": 1,
      "line_end": 1,
      "description": "Short description of the change",
      "original_snippet": "...",
      "new_snippet": "..."
    }
  ],
  "explanation": "Human-readable explanation of what changed and why",
  "new_imports": ["import torch"],
  "warnings": ["any caveats about the change"],
  "not_applicable_reason": null
}

`file_edits` is OPTIONAL — omit or use [] when the entry script alone covers the change.
`changes[].path` is OPTIONAL and defaults to the entry script.

If the optimization cannot be applied:
{
  "edited_source": "...original entry source unchanged...",
  "applied": false,
  "file_edits": [],
  "changes": [],
  "explanation": "",
  "new_imports": [],
  "warnings": [],
  "not_applicable_reason": "Why this optimization cannot be applied to this code"
}
"""


HEALING_SYSTEM = """\
You are an expert ML engineer. A previous code edit produced a script that fails
to parse or has an obvious error. You will receive the broken source and the error.
Fix ONLY the error — do not make other changes.

Return ONLY valid JSON:
{
  "edited_source": "...the fixed source code...",
  "fix_description": "what you fixed"
}
"""


def build_edit_prompt(
    source: str,
    candidate: OptimizationCandidate,
    architecture_record: dict[str, Any] | None = None,
    user_preferences: str | None = None,
    entry_path: str | None = None,
    local_modules: dict[str, str] | None = None,
) -> str:
    """Build the user message for the code editor LLM call."""
    sections: list[str] = []

    # Source code with line numbers
    numbered = "\n".join(f"{i+1:>4} | {line}" for i, line in enumerate(source.splitlines()))
    header = f"## Entry Script (`{entry_path}`)" if entry_path else "## Original Source Code"
    sections.append(f"{header}\n```python\n{numbered}\n```")

    # Local modules the entry script imports (read context — may be edited).
    if local_modules:
        mod_blocks: list[str] = ["## Local Modules"]
        for path, src in local_modules.items():
            num = "\n".join(f"{i+1:>4} | {line}" for i, line in enumerate(src.splitlines()))
            mod_blocks.append(f"### `{path}`\n```python\n{num}\n```")
        sections.append("\n\n".join(mod_blocks))

    # Optimization to apply
    opt_data: dict[str, Any] = {
        "entry_id": candidate.entry_id,
        "name": candidate.name,
        "category": candidate.category,
        "description": candidate.description,
        "code_pattern": candidate.code_pattern,
        "risks": candidate.risks,
        "bottlenecks_addressed": candidate.bottlenecks_addressed,
        "rationale": candidate.rationale,
    }
    sections.append("## Optimization to Apply\n```json\n"
                     + json.dumps(opt_data, indent=2) + "\n```")

    # Architecture (compact values only)
    if architecture_record:
        compact = _compact_architecture(architecture_record)
        sections.append("## Architecture Record\n```json\n"
                         + json.dumps(compact, indent=2) + "\n```")

    if user_preferences:
        sections.append(f"## User Preferences\n{user_preferences}")

    return "\n\n".join(sections)


def build_healing_prompt(broken_source: str, error: str) -> str:
    """Build the user message for healing a broken edit."""
    return (
        f"## Broken Source\n```python\n{broken_source}\n```\n\n"
        f"## Error\n```\n{error}\n```"
    )


def _compact_architecture(arch: dict[str, Any]) -> dict[str, Any]:
    """Extract just the values from an architecture record."""
    compact: dict[str, Any] = {}
    for key, val in arch.items():
        if key in ("dependencies", "unstructured_notes", "script_path"):
            continue
        if isinstance(val, dict):
            if "value" in val:
                compact[key] = val["value"]
            else:
                inner: dict[str, Any] = {}
                for k2, v2 in val.items():
                    if isinstance(v2, dict) and "value" in v2:
                        inner[k2] = v2["value"]
                if inner:
                    compact[key] = inner
    return compact
