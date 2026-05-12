"""Fetch model metadata from HuggingFace Hub.

Given a model ID (e.g. "Qwen/Qwen3-0.6B"), retrieves config.json
and extracts ground-truth architecture facts that the AST extractor
and LLM analyzer can't determine from the script alone.
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from typing import Any


HF_CONFIG_URL = "https://huggingface.co/{model_id}/resolve/main/config.json"

_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "float64": "float32",
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}

# Matches "org/model-name" but not local paths or URLs
_HF_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9._-]+$")


def is_hf_model_id(value: str) -> bool:
    """Check if a string looks like a HuggingFace model ID."""
    return bool(_HF_ID_PATTERN.match(value))


def fetch_model_config(model_id: str, timeout: float = 10.0) -> dict[str, Any] | None:
    """Fetch config.json from HuggingFace Hub. Returns None on failure."""
    url = HF_CONFIG_URL.format(model_id=model_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "profine/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None


def _make_field(value: Any, snippet: str) -> dict[str, Any]:
    """Build an architecture record field from HF config data."""
    return {
        "value": value,
        "confidence": "observed",
        "evidence": [{"kind": "config", "ref": "HuggingFace config.json",
                      "snippet": snippet}],
        "notes": "From HuggingFace Hub config.json",
    }


def extract_facts_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture facts from an HF config.json.

    Returns a dict of dotted field paths -> record field dicts.
    """
    facts: dict[str, Any] = {}

    # Precision
    torch_dtype = config.get("torch_dtype")
    if torch_dtype and torch_dtype in _DTYPE_MAP:
        facts["precision.training_dtype"] = _make_field(
            _DTYPE_MAP[torch_dtype],
            f"torch_dtype: {torch_dtype}",
        )

    # Attention type (GQA vs MHA vs MQA)
    num_kv_heads = config.get("num_key_value_heads")
    num_heads = config.get("num_attention_heads")
    if num_kv_heads and num_heads:
        if num_kv_heads == 1:
            attn_type = "mqa"
        elif num_kv_heads < num_heads:
            attn_type = "gqa"
        else:
            attn_type = "mha"
        facts["attention_type"] = _make_field(
            attn_type,
            f"num_attention_heads: {num_heads}, num_key_value_heads: {num_kv_heads}",
        )

    # Model dimensions
    for hf_key, our_key in [
        ("num_hidden_layers", "num_layers"),
        ("hidden_size", "hidden_size"),
        ("num_attention_heads", "num_heads"),
        ("vocab_size", "vocab_size"),
        ("max_position_embeddings", "context_length"),
    ]:
        val = config.get(hf_key)
        if val is not None:
            facts[our_key] = _make_field(val, f"{hf_key}: {val}")

    # Head dim
    if num_heads and config.get("hidden_size"):
        facts["head_dim"] = _make_field(
            config["hidden_size"] // num_heads,
            "hidden_size / num_attention_heads",
        )

    # Model family
    model_type = config.get("model_type")
    if model_type:
        facts["model_family"] = _make_field(model_type, f"model_type: {model_type}")

    return facts


def enrich_record(record: dict[str, Any], model_id: str) -> list[str]:
    """Fetch HF config and upgrade guessed/inferred fields in the record.

    Only overwrites fields where confidence is NOT "observed" — preserving
    anything the LLM proved from the actual source code.

    Returns list of field names that were upgraded.
    """
    config = fetch_model_config(model_id)
    if config is None:
        return []

    hf_facts = extract_facts_from_config(config)
    upgraded: list[str] = []

    for dotted_key, hf_val in hf_facts.items():
        parts = dotted_key.split(".")
        target = record
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        field_name = parts[-1]
        existing = target.get(field_name)

        if existing is None or (
            isinstance(existing, dict)
            and existing.get("confidence") != "observed"
        ):
            target[field_name] = hf_val
            upgraded.append(dotted_key)

    return upgraded
