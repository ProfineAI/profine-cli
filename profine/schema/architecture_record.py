"""Architecture record schema.

This is the canonical structured output of the Read Code tool (plan section 4.1).
Every field carries an evidence list of file:line citations. Fields the LLM
cannot back with code evidence are marked confidence: "guessed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Catalog rules use `eq` against these fields; values outside these sets
# would silently bypass a rule.
FIELD_LITERALS: dict[str, set[Any]] = {
    "compile_mode": {"disabled", "default", "reduce-overhead",
                       "max-autotune", "max-autotune-no-cudagraphs"},
    "precision.training_dtype": {"float32", "bfloat16", "float16"},
    "attention_impl": {"manual", "eager", "sdpa", "flash_attention_2",
                        "flash_attention_3", "xformers", "custom"},
    "attention_type": {"causal_mha", "mha", "multi-head", "self-attention",
                        "mqa", "gqa", "bidirectional", "alibi_with_custom_mask"},
    "optimizer.fused": {True, False},
    "optimizer.foreach": {True, False},
}

FIELD_COERCIONS: dict[str, dict[Any, Any]] = {
    "precision.training_dtype": {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"},
    "compile_mode": {True: "default", False: "disabled"},
}

# LLM occasionally returns prose for these (e.g. "40 default, scaled by
# world_size under DDP"); validate_record extracts the leading int.
NUMERIC_FIELDS: set[str] = {
    "distributed.gradient_accumulation_steps",
    "distributed.world_size",
    "dataloader.batch_size",
    "dataloader.num_workers",
    "context_length",
    "hidden_size",
    "num_layers",
    "num_heads",
    "vocab_size",
    "head_dim",
}


@dataclass(slots=True)
class Evidence:
    """A single piece of evidence backing a claim."""
    file: str
    line: int
    snippet: str
    kind: str = "code"  # "code" | "comment" | "import" | "config"


@dataclass(slots=True)
class ArchitectureField:
    """A single field in the architecture record.

    Every claim the reader makes is wrapped in this so downstream tools
    always know *where* the claim came from and how much to trust it.
    """
    value: Any
    confidence: str = "observed"  # "observed" | "inferred" | "guessed"
    evidence: list[Evidence] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class DependencyInfo:
    """A single Python dependency detected in the script."""
    name: str
    line: int
    import_style: str = ""  # "import X" | "from X import Y"


@dataclass(slots=True)
class OptimizerInfo:
    """Optimizer configuration."""
    name: ArchitectureField | None = None          # e.g. "AdamW"
    fused: ArchitectureField | None = None          # True/False
    foreach: ArchitectureField | None = None        # True/False
    learning_rate: ArchitectureField | None = None
    weight_decay: ArchitectureField | None = None
    extra_params: dict[str, ArchitectureField] = field(default_factory=dict)


@dataclass(slots=True)
class DataLoaderInfo:
    """DataLoader configuration."""
    num_workers: ArchitectureField | None = None
    batch_size: ArchitectureField | None = None
    pin_memory: ArchitectureField | None = None
    prefetch_factor: ArchitectureField | None = None
    persistent_workers: ArchitectureField | None = None
    shuffle: ArchitectureField | None = None
    dataset_class: ArchitectureField | None = None


@dataclass(slots=True)
class DistributedInfo:
    """Distributed training configuration."""
    strategy: ArchitectureField | None = None       # "none" | "ddp" | "fsdp" | "deepspeed" | "data_parallel"
    world_size: ArchitectureField | None = None
    gradient_accumulation_steps: ArchitectureField | None = None


@dataclass(slots=True)
class PrecisionInfo:
    """Mixed precision configuration."""
    training_dtype: ArchitectureField | None = None   # "fp32" | "fp16" | "bf16"
    autocast_enabled: ArchitectureField | None = None
    grad_scaler: ArchitectureField | None = None


@dataclass(slots=True)
class ArchitectureRecord:
    """The canonical output of Read Code.

    This is *the* reference for 'what this codebase is'. Every downstream
    tool receives this instead of re-reading raw source.
    """
    # Identity
    script_path: str = ""
    framework: ArchitectureField | None = None       # "raw_pytorch" | "huggingface" | "lightning" | ...

    # Model
    model_family: ArchitectureField | None = None     # "GPT-2" | "LLaMA" | "ResNet" | ...
    model_class: ArchitectureField | None = None      # the actual class name used
    model_variable: ArchitectureField | None = None   # variable name the model is bound to
    layer_composition: ArchitectureField | None = None # list of layer types/counts
    attention_type: ArchitectureField | None = None    # "causal_mha" | "mqa" | "gqa" | "bidirectional" | ...
    attention_impl: ArchitectureField | None = None    # "manual" | "sdpa" | "flash_attention_2" | ...
    head_dim: ArchitectureField | None = None
    num_heads: ArchitectureField | None = None
    num_layers: ArchitectureField | None = None
    hidden_size: ArchitectureField | None = None
    vocab_size: ArchitectureField | None = None
    context_length: ArchitectureField | None = None

    # Training
    optimizer: OptimizerInfo | None = None
    dataloader: DataLoaderInfo | None = None
    loss_function: ArchitectureField | None = None
    scheduler: ArchitectureField | None = None

    # Infrastructure
    distributed: DistributedInfo | None = None
    precision: PrecisionInfo | None = None
    compile_mode: ArchitectureField | None = None     # None | "default" | "reduce-overhead" | "max-autotune"
    custom_kernels: ArchitectureField | None = None   # list of custom CUDA/Triton kernels found
    gradient_checkpointing: ArchitectureField | None = None

    # Dependencies
    dependencies: list[DependencyInfo] = field(default_factory=list)

    # LLM free-form notes (things that don't fit the schema)
    unstructured_notes: list[str] = field(default_factory=list)


# JSON schema for validation (used by downstream tools and the orchestrator).
ARCHITECTURE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ArchitectureRecord",
    "description": "Structured representation of a training script produced by the Read Code tool.",
    "type": "object",
    "properties": {
        "script_path": {"type": "string"},
        "framework": {"$ref": "#/$defs/field"},
        "model_family": {"$ref": "#/$defs/field"},
        "model_class": {"$ref": "#/$defs/field"},
        "model_variable": {"$ref": "#/$defs/field"},
        "layer_composition": {"$ref": "#/$defs/field"},
        "attention_type": {"$ref": "#/$defs/field"},
        "attention_impl": {"$ref": "#/$defs/field"},
        "head_dim": {"$ref": "#/$defs/field"},
        "num_heads": {"$ref": "#/$defs/field"},
        "num_layers": {"$ref": "#/$defs/field"},
        "hidden_size": {"$ref": "#/$defs/field"},
        "vocab_size": {"$ref": "#/$defs/field"},
        "context_length": {"$ref": "#/$defs/field"},
        "optimizer": {
            "type": "object",
            "properties": {
                "name": {"$ref": "#/$defs/field"},
                "fused": {"$ref": "#/$defs/field"},
                "foreach": {"$ref": "#/$defs/field"},
                "learning_rate": {"$ref": "#/$defs/field"},
                "weight_decay": {"$ref": "#/$defs/field"},
            },
        },
        "dataloader": {
            "type": "object",
            "properties": {
                "num_workers": {"$ref": "#/$defs/field"},
                "batch_size": {"$ref": "#/$defs/field"},
                "pin_memory": {"$ref": "#/$defs/field"},
                "prefetch_factor": {"$ref": "#/$defs/field"},
                "persistent_workers": {"$ref": "#/$defs/field"},
                "shuffle": {"$ref": "#/$defs/field"},
                "dataset_class": {"$ref": "#/$defs/field"},
            },
        },
        "loss_function": {"$ref": "#/$defs/field"},
        "scheduler": {"$ref": "#/$defs/field"},
        "distributed": {
            "type": "object",
            "properties": {
                "strategy": {"$ref": "#/$defs/field"},
                "world_size": {"$ref": "#/$defs/field"},
                "gradient_accumulation_steps": {"$ref": "#/$defs/field"},
            },
        },
        "precision": {
            "type": "object",
            "properties": {
                "training_dtype": {"$ref": "#/$defs/field"},
                "autocast_enabled": {"$ref": "#/$defs/field"},
                "grad_scaler": {"$ref": "#/$defs/field"},
            },
        },
        "compile_mode": {"$ref": "#/$defs/field"},
        "custom_kernels": {"$ref": "#/$defs/field"},
        "gradient_checkpointing": {"$ref": "#/$defs/field"},
        "dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "line": {"type": "integer"},
                    "import_style": {"type": "string"},
                },
                "required": ["name", "line"],
            },
        },
        "unstructured_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["script_path"],
    "$defs": {
        "evidence": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "line": {"type": "integer"},
                "snippet": {"type": "string"},
                "kind": {"type": "string", "enum": ["code", "comment", "import", "config"]},
            },
            "required": ["file", "line", "snippet"],
        },
        "field": {
            "type": "object",
            "properties": {
                "value": {},
                "confidence": {
                    "type": "string",
                    "enum": ["observed", "inferred", "guessed"],
                },
                "evidence": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/evidence"},
                },
                "notes": {"type": "string"},
            },
            "required": ["value", "confidence", "evidence"],
        },
    },
}


def _walk(record: dict, dotted: str) -> tuple[bool, Any, dict | None, str]:
    """Walk a dotted path through a record dict.

    Returns (found, value, parent_dict, leaf_key) where parent_dict[leaf_key]
    is the field dict whose ".value" we resolved (or None if the field is
    absent). Used by the validator to coerce in place.
    """
    parts = dotted.split(".")
    cur: Any = record
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return False, None, None, parts[-1]
        cur = cur[part]
    leaf = parts[-1]
    if not isinstance(cur, dict) or leaf not in cur or not isinstance(cur[leaf], dict):
        return False, None, None, leaf
    field_dict = cur[leaf]
    if "value" not in field_dict:
        return False, None, None, leaf
    return True, field_dict["value"], field_dict, "value"


def _coerce_numeric(value: Any) -> tuple[Any, str | None]:
    """If `value` is a string starting with a number, return (int, original).
    Otherwise return (value, None)."""
    if isinstance(value, str):
        import re
        m = re.match(r"\s*(-?\d+)", value)
        if m:
            return int(m.group(1)), value
    return value, None


def validate_record(record: dict) -> list[str]:
    """Coerce known aliases in place; return non-fatal validation errors."""
    errors: list[str] = []
    for path, allowed in FIELD_LITERALS.items():
        found, value, parent, leaf = _walk(record, path)
        if not found:
            continue
        coerced = FIELD_COERCIONS.get(path, {}).get(value, value)
        if coerced != value:
            parent[leaf] = coerced
            value = coerced
        if value not in allowed:
            errors.append(
                f"{path}.value = {value!r} is not in allowed literals "
                f"{sorted(allowed, key=str)}"
            )
    for path in NUMERIC_FIELDS:
        found, value, field_dict, _ = _walk(record, path)
        if not found:
            continue
        numeric, original = _coerce_numeric(value)
        if original is not None:
            field_dict["value"] = numeric
            existing = field_dict.get("notes", "")
            field_dict["notes"] = (existing + " | " if existing else "") + f"raw: {original!r}"
        elif not isinstance(value, (int, float)):
            errors.append(f"{path}.value = {value!r} is not numeric")
    return errors
