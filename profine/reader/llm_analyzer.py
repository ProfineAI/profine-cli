"""LLM-driven code analyzer.

Takes deterministic CodeFacts + raw source and asks an LLM to produce:
  1. A structured ArchitectureRecord (machine-readable)
  2. A markdown architecture brief (human-readable)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from pathlib import Path

from profine.llm.backend import LlmBackend, create_backend
from profine.llm.utils import call_and_parse, parse_json_response
from profine.reader.extractor import CodeFacts
from profine.schema.architecture_record import validate_record


# Prompt

SYSTEM_PROMPT = """\
You are an expert ML systems engineer analyzing a PyTorch training script.

You will receive:
1. The full source code of the script.
2. Pre-extracted facts from static analysis (imports, calls, assignments, etc.) with line numbers.

Your job is to produce TWO outputs, both in a single JSON response:

A) "architecture_record" — a structured JSON object following the schema below.
B) "markdown_brief" — a human-readable markdown summary of the architecture.

## Rules

- Every field in the architecture_record MUST have:
  - "value": the actual value
  - "confidence": one of "observed" (you can point to exact code), "inferred" (you deduced it from context), or "guessed" (you are speculating)
  - "evidence": list of {"file": str, "line": int, "snippet": str, "kind": "code"|"comment"|"import"|"config"}
  - "notes": optional string for caveats

- If you cannot determine a field at all, omit it entirely. Do NOT guess just to fill fields.
- If you ARE guessing, you MUST set confidence to "guessed" and explain in notes.
- The markdown_brief should cite evidence as (file:line) inline.
- The markdown_brief should warn the user about any guessed fields.
- Be precise. "AdamW" not "adam". "causal_mha" not "attention".

## Field formats (downstream checks are exact-match)

- `precision.training_dtype.value`: one of "float32" | "bfloat16" | "float16".
  For hardware-conditional dtypes (e.g. `'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'`),
  resolve to the branch that runs on Ampere+ targets ("bfloat16") and note the conditional.
- `compile_mode.value`: a string, never a bool — "disabled" | "default" |
  "reduce-overhead" | "max-autotune". "default" means `torch.compile(model)` with no `mode=`.
- `optimizer.fused.value` / `optimizer.foreach.value`: only set if the kwarg is
  explicitly passed to the optimizer (including via `configure_optimizers`).
  Omit the field if absent — do not assume false.
- `attention_impl.value`: how attention is computed at runtime. One of
  "manual" (hand-written QK^T/softmax/V), "eager" (HF eager attention),
  "sdpa" (`F.scaled_dot_product_attention`), "flash_attention_2",
  "flash_attention_3", "xformers", or "custom". Distinct from
  `attention_type.value`, which describes the model (e.g. "causal_mha").
- Numeric fields (`gradient_accumulation_steps`, `world_size`, `batch_size`,
  `num_workers`, `context_length`, `hidden_size`, `num_layers`, `num_heads`,
  `vocab_size`, `head_dim`): `value` MUST be a number, not a string.
  Resolve expressions like `5 * 8` to `40`. If the value depends on a
  runtime condition (e.g. divided by `world_size`), put the literal default
  in `value` and explain the conditional in `notes`.

## Architecture Record Schema (field names)

Top-level fields: script_path, framework, model_family, model_class, model_variable,
layer_composition, attention_type, head_dim, num_heads, num_layers, hidden_size,
vocab_size, context_length, optimizer{name,fused,foreach,learning_rate,weight_decay},
dataloader{num_workers,batch_size,pin_memory,prefetch_factor,persistent_workers,shuffle,dataset_class},
loss_function, scheduler, distributed{strategy,world_size,gradient_accumulation_steps},
precision{training_dtype,autocast_enabled,grad_scaler}, compile_mode,
custom_kernels, gradient_checkpointing, dependencies, unstructured_notes.

## Response format

Return ONLY valid JSON (no markdown fences) with exactly two keys:
{
  "architecture_record": { ... },
  "markdown_brief": "..."
}
"""


def _build_user_message(
    source: str,
    facts: CodeFacts,
    local_modules: dict[str, str] | None = None,
) -> str:
    facts_dict = asdict(facts)
    for key in ("calls", "assignments"):
        items = facts_dict.get(key, [])
        if len(items) > 100:
            facts_dict[key] = items[:100]
            facts_dict.setdefault("_truncation_notes", []).append(
                f"{key} truncated to 100 items (had {len(items)})"
            )

    parts = [
        "## Source Code\n",
        _numbered_source(source),
        "\n## Extracted Facts (from static analysis)\n",
        json.dumps(facts_dict, indent=2, default=str),
    ]
    if local_modules:
        parts.append(
            "\n## Related Local Modules (imported by the script)\n"
            "Treat these as authoritative — defaults defined here are NOT "
            "guesses. Cite them in evidence using their relative path.\n"
        )
        for rel_path, text in local_modules.items():
            parts.append(f"\n### `{rel_path}`\n")
            parts.append(_numbered_source(text))
    return "\n".join(parts)


def _numbered_source(source: str) -> str:
    lines = source.splitlines()
    return "\n".join(f"{i+1:>4} | {line}" for i, line in enumerate(lines))


# Analysis

def analyze(
    source: str,
    facts: CodeFacts,
    provider: str = "openai",
    *,
    debug_dir: Path | str | None = None,
    local_modules: dict[str, str] | None = None,
    **llm_kwargs: Any,
) -> tuple[dict[str, Any], str]:
    """Run LLM analysis on extracted facts.

    Args:
        local_modules: Optional map of `relative_path -> source` for files
            the entry script transitively imports.

    Returns:
        (architecture_record_dict, markdown_brief)
    """
    backend = create_backend(provider, **llm_kwargs)
    user_msg = _build_user_message(source, facts, local_modules=local_modules)
    parsed = call_and_parse(
        backend, SYSTEM_PROMPT, user_msg,
        debug_dir=debug_dir, debug_label="reader_response",
    )
    record = parsed.get("architecture_record", {})
    brief = parsed.get("markdown_brief", "")

    errors = validate_record(record)
    if errors:
        retry_msg = (
            user_msg
            + "\n\n## Your previous response failed schema validation\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\nReturn corrected JSON using only the documented literal values."
        )
        parsed = call_and_parse(
            backend, SYSTEM_PROMPT, retry_msg,
            debug_dir=debug_dir, debug_label="reader_response_retry",
        )
        record = parsed.get("architecture_record", {})
        brief = parsed.get("markdown_brief", "")
        residual = validate_record(record)
        if residual:
            existing = record.get("unstructured_notes")
            notes = existing if isinstance(existing, list) else []
            notes.extend(f"schema warning: {e}" for e in residual)
            record["unstructured_notes"] = notes
    return record, brief


def _parse_response(raw: str) -> tuple[dict[str, Any], str]:
    """Parse the LLM JSON response into (record_dict, markdown_brief)."""
    parsed = parse_json_response(raw)
    record = parsed.get("architecture_record", {})
    brief = parsed.get("markdown_brief", "")
    return record, brief
