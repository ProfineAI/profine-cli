"""Run fingerprinting for telemetry aggregation.

Turns an ArchitectureRecord + HardwareConfig into a `Fingerprint`:

  * Seven bucketed dimensions that feed into a stable sha256 hash —
    the *k-anonymity surface*. Aggregations in the priors view group
    by this hash, so it stays coarse on purpose. Two runs of "similar
    kinds of work" collapse to the same hash.

  * A larger pool of richer enum dimensions (compile_mode,
    distributed_strategy, attention_impl, framework,
    gradient_checkpointing, has_grad_scaler) that are *recorded
    alongside* but **not** in the hash. The data collection is rich;
    the read path stays narrow.

Why this split: granular dimensions explode bucket counts and make
the k=5 floor unreachable for years. Booleans keep buckets dense so
priors actually accumulate. When a richer dimension has enough data
to clear k=5 on its own, we can promote it into the hash by bumping
the catalog/fingerprint version.

To minimise hardcoding, classifier rules are data-driven dicts at
module scope. Adding a new arch family or optimizer = adding one
entry, not editing branching logic. Bucket boundaries are also
single sources of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from profine.schema.architecture_record import ArchitectureRecord
from profine.schema.hardware import HardwareConfig


# =============================================================
# Enum-like sets. Anything outside a set falls into "other" or
# "unknown" so the priors view never sees a singleton tail.
# =============================================================

ARCH_CLASSES: frozenset[str] = frozenset({
    "transformer-decoder",
    "transformer-encoder",
    "transformer-enc-dec",
    "vit",
    "cnn",
    "mlp",
    "rnn",
    "diffusion",
    "rl-policy",
    "other",
})

OPTIMIZER_CLASSES: frozenset[str] = frozenset({
    "adam_family",
    "sgd_family",
    "adafactor",
    "lamb",
    "lion",
    "shampoo",
    "rmsprop",
    "other",
})

PRECISIONS: frozenset[str] = frozenset({
    "fp32", "fp16", "bf16", "fp8",
    "mixed_fp16", "mixed_bf16",
    "unknown",
})

FRAMEWORKS: frozenset[str] = frozenset({
    "raw_pytorch", "huggingface", "lightning", "accelerate", "trl", "fairscale", "other",
})

# torch.compile modes that actually exist as of PyTorch 2.x.
COMPILE_MODES: frozenset[str | None] = frozenset({
    None,
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
})

DISTRIBUTED_STRATEGIES: frozenset[str | None] = frozenset({
    None,
    "ddp",
    "fsdp",
    "deepspeed",
    "data_parallel",
    "tensor_parallel",
    "pipeline_parallel",
})

ATTENTION_IMPLS: frozenset[str | None] = frozenset({
    None,
    "manual",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    "xformers",
})


# =============================================================
# Bucket boundaries. Single source of truth — adjust here and
# both runtime + tests pick up the change. Buckets cover the full
# realistic range from tiny RL policies (~100k params) to
# frontier LLMs (>70B).
# =============================================================

# (label, exclusive upper bound). Upper=None means open-ended.
PARAM_BUCKETS: tuple[tuple[str, int | None], ...] = (
    ("<1M",        1_000_000),         # tiny RL policies, small MLPs
    ("1M-10M",     10_000_000),        # small CNNs (MobileNet, ResNet18), LoRA adapters
    ("10M-100M",   100_000_000),       # ResNet50, DistilBERT, GPT-2 small
    ("100M-1B",    1_000_000_000),     # ViT-L, BERT-large, GPT-2 XL
    ("1B-7B",      7_000_000_000),
    ("7B-13B",     13_000_000_000),
    ("13B-70B",    70_000_000_000),
    ("70B+",       None),
)

# NVIDIA compute capability threshold for native bf16 (Ampere+, sm80+).
_BF16_NATIVE_SM_MAJOR: int = 80
# Hopper (sm90+) adds fp8 support; not currently used for inference but
# kept as the threshold for any future fp8-aware decision.
_FP8_NATIVE_SM_MAJOR: int = 90


# =============================================================
# Classifier rules — data-driven dicts. Adding a new arch family
# or optimizer = one entry, no branching code.
# =============================================================
#
# Each rule list is checked in order; first match wins. The needle
# is the lowercased model_family + model_class + layer_composition
# joined by spaces, so any of those facts can trigger a match.

_ARCH_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Most specific first — diffusion subsumes "unet", which a generic
    # CNN check would otherwise grab.
    ("diffusion",            ("diffusion", "stablediffusion", "unet", "ddpm", "ddim", "dit")),
    ("vit",                  ("vit", "deit", "swin", "beit", "videomae")),
    ("transformer-enc-dec",  ("encoder-decoder", "encoder_decoder", "seq2seq", "t5", "bart")),
    ("cnn",                  ("resnet", "convnext", "efficientnet", "mobilenet", "vgg", "inception", "densenet")),
    ("rnn",                  ("lstm", "gru", "rnn")),
    ("rl-policy",            ("ppo", "sac", "dqn", "a2c", "policynet", "valuenet", "actor_critic")),
    # Generic mlp last to avoid catching e.g. "mlpmixer" early (mixers
    # mostly act like ViTs in practice; categorising as ViT is fine).
    ("mlp",                  ("mlp_only", "feedforward_only")),
)


def _keyword_in(needle: str, keyword: str) -> bool:
    """Word-boundary prefix match: 'resnet' matches 'resnet50' but
    't5' does NOT match 'resnet50' (because there's no boundary
    immediately before the 't5' inside 'resnet50'). The keyword may
    contain any characters; we escape it for regex safety."""
    pattern = r"\b" + re.escape(keyword)
    return re.search(pattern, needle) is not None

# Attention-type → arch_class shortcut. Checked when the keyword
# rules don't fire but we have explicit attention info.
_ATTENTION_TO_ARCH: dict[str, str] = {
    "causal_mha": "transformer-decoder",
    "mqa":        "transformer-decoder",
    "gqa":        "transformer-decoder",
    "bidirectional": "transformer-encoder",
}

# (substring tuple, optimizer_class). Substring match is case-insensitive
# on the optimizer name.
_OPTIMIZER_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("adamw", "adam", "nadam", "radam"), "adam_family"),
    (("sgd", "momentum"),                  "sgd_family"),
    (("adafactor",),                       "adafactor"),
    (("lamb",),                            "lamb"),
    (("lion",),                            "lion"),
    (("shampoo",),                         "shampoo"),
    (("rmsprop",),                         "rmsprop"),
)

# Import-name fragments that imply a framework. First match wins on
# scan over deps + imports.
_FRAMEWORK_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pytorch_lightning", "lightning.pytorch", "lightning_fabric"), "lightning"),
    (("trl",), "trl"),
    (("accelerate",), "accelerate"),
    (("fairscale",), "fairscale"),
    (("transformers", "datasets", "diffusers", "peft"), "huggingface"),
)


# =============================================================
# Public data classes
# =============================================================

@dataclass(slots=True, frozen=True)
class Fingerprint:
    """Bucketed run signature plus its stable hash and the richer
    enum dimensions we collect but do not yet feed into the hash.

    The first seven fields feed `fingerprint_hash` (the k-anonymity
    surface). The remaining fields are collected for future analytics
    and possible promotion.
    """

    # ----- in the hash (k-anonymity surface) -----
    arch_class: str
    param_bucket: str
    hardware_class: str
    precision: str
    optimizer_class: str
    has_compile: bool
    has_distributed: bool

    fingerprint_hash: str

    # ----- recorded only (richer enums; not in the hash) -----
    compile_mode: str | None = None
    distributed_strategy: str | None = None
    attention_impl: str | None = None
    framework: str | None = None
    gradient_checkpointing: bool = False
    has_grad_scaler: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================
# Public API
# =============================================================

def fingerprint_run(
    arch: ArchitectureRecord | None,
    hardware: HardwareConfig,
) -> Fingerprint:
    """Compute the fingerprint for a single run.

    `arch` may be None (e.g. when the reader step failed entirely);
    derived fields fall back to their "unknown"/"other" buckets in
    that case. The hash is still stable but it is unlikely to clear
    k=5 in priors.
    """
    arch_class = arch_class_of(arch)
    param_bucket = param_bucket_of(_estimate_params(arch))
    hardware_class = _normalize_hardware_name(hardware.name)
    precision = precision_of(arch, hardware)
    optimizer_class = optimizer_class_of(arch)
    compile_mode = _compile_mode(arch)
    distributed_strategy = _distributed_strategy(arch)
    attention_impl = _attention_impl(arch)
    framework = _framework(arch)
    gradient_checkpointing = _gradient_checkpointing(arch)
    has_grad_scaler = _has_grad_scaler(arch)

    has_compile = compile_mode is not None
    has_distributed = distributed_strategy is not None

    # Only the seven bucketed dims go into the hash. Promoting a
    # dimension later means appending it to this dict AND bumping
    # the catalog_version on outcome rows so old priors stay
    # partitioned.
    canonical = {
        "arch_class": arch_class,
        "param_bucket": param_bucket,
        "hardware_class": hardware_class,
        "precision": precision,
        "optimizer_class": optimizer_class,
        "has_compile": has_compile,
        "has_distributed": has_distributed,
    }
    fingerprint_hash = _stable_hash(canonical)

    return Fingerprint(
        arch_class=arch_class,
        param_bucket=param_bucket,
        hardware_class=hardware_class,
        precision=precision,
        optimizer_class=optimizer_class,
        has_compile=has_compile,
        has_distributed=has_distributed,
        fingerprint_hash=fingerprint_hash,
        compile_mode=compile_mode,
        distributed_strategy=distributed_strategy,
        attention_impl=attention_impl,
        framework=framework,
        gradient_checkpointing=gradient_checkpointing,
        has_grad_scaler=has_grad_scaler,
    )


def arch_class_of(arch: ArchitectureRecord | None) -> str:
    """Classify the architecture into one of ARCH_CLASSES.

    Rules are checked in `_ARCH_KEYWORD_RULES` order (most specific
    first). If no keyword fires but the attention type is set, we
    fall back to the attention-based shortcut.
    """
    if arch is None:
        return "other"

    needle = " ".join(filter(None, (
        _field_str(arch.model_family),
        _field_str(arch.model_class),
        _field_str(arch.layer_composition),
    ))).lower()

    for cls, keywords in _ARCH_KEYWORD_RULES:
        if any(_keyword_in(needle, kw) for kw in keywords):
            return cls

    attention = (_field_str(arch.attention_type) or "").lower()
    if attention in _ATTENTION_TO_ARCH:
        return _ATTENTION_TO_ARCH[attention]

    # Generic "transformer" anywhere but no attention info → assume decoder
    # (decoder is the majority case for modern training scripts).
    if _keyword_in(needle, "transformer"):
        return "transformer-decoder"

    return "other"


def optimizer_class_of(arch: ArchitectureRecord | None) -> str:
    """Map the optimizer name to one of OPTIMIZER_CLASSES."""
    if arch is None or arch.optimizer is None:
        return "other"
    name = (_field_str(arch.optimizer.name) or "").lower()
    if not name:
        return "other"
    for substrings, cls in _OPTIMIZER_PATTERNS:
        if any(s in name for s in substrings):
            return cls
    return "other"


def param_bucket_of(param_count: int | None) -> str:
    """Coarse log-scale bucket for parameter count.

    Range covers tiny RL policies (~100k) through frontier LLMs
    (>70B). For non-transformer arches with no explicit param count,
    callers pass None and the result is `unknown`.
    """
    if param_count is None or param_count <= 0:
        return "unknown"
    for label, upper in PARAM_BUCKETS:
        if upper is None or param_count < upper:
            return label
    return "70B+"  # unreachable due to None-terminated tuple


def precision_of(
    arch: ArchitectureRecord | None,
    hardware: HardwareConfig | None = None,
) -> str:
    """Normalize precision into one of PRECISIONS.

    For autocast on top of fp32, we infer the mixed flavour from the
    target hardware: Ampere+ (sm80+) → mixed_bf16, otherwise →
    mixed_fp16. This matches real-world usage where bf16 is the
    default mixed precision on A100/H100/L4 due to its wider exponent.
    """
    if arch is None or arch.precision is None:
        return "unknown"

    dtype = (_field_str(arch.precision.training_dtype) or "").lower()
    autocast = _field_bool(arch.precision.autocast_enabled) is True

    if dtype in ("fp16", "float16", "half"):
        return "mixed_fp16" if autocast else "fp16"
    if dtype in ("bf16", "bfloat16"):
        return "mixed_bf16" if autocast else "bf16"
    if dtype in ("fp8", "float8"):
        return "fp8"
    if dtype in ("fp32", "float32"):
        if not autocast:
            return "fp32"
        return "mixed_bf16" if _supports_native_bf16(hardware) else "mixed_fp16"
    if dtype == "" and autocast:
        # Common in HF training scripts: autocast=True with the base
        # dtype left implicit. Same hardware-aware choice applies.
        return "mixed_bf16" if _supports_native_bf16(hardware) else "mixed_fp16"
    return "unknown"


# =============================================================
# Internal helpers — richer enum extraction
# =============================================================

def _compile_mode(arch: ArchitectureRecord | None) -> str | None:
    """Return the recorded compile mode (string) or None if no compile."""
    if arch is None:
        return None
    mode = _field_str(arch.compile_mode)
    if not mode or mode.lower() == "none":
        return None
    mode_normalised = mode.strip().lower()
    return mode_normalised if mode_normalised in {m for m in COMPILE_MODES if m} else "other"


def _distributed_strategy(arch: ArchitectureRecord | None) -> str | None:
    if arch is None or arch.distributed is None:
        return None
    strat = (_field_str(arch.distributed.strategy) or "").strip().lower()
    if not strat or strat == "none":
        return None
    return strat if strat in {s for s in DISTRIBUTED_STRATEGIES if s} else "other"


def _attention_impl(arch: ArchitectureRecord | None) -> str | None:
    if arch is None:
        return None
    impl = (_field_str(arch.attention_impl) or "").strip().lower()
    if not impl:
        return None
    return impl if impl in {i for i in ATTENTION_IMPLS if i} else "other"


def _framework(arch: ArchitectureRecord | None) -> str | None:
    """Detect the training framework.

    Two signals: the LLM may set `arch.framework` directly, or we
    can sniff it from the dependency list. Direct setting wins.
    """
    if arch is None:
        return None

    direct = (_field_str(arch.framework) or "").strip().lower()
    if direct in FRAMEWORKS:
        return direct
    if direct in ("torch", "pytorch"):
        return "raw_pytorch"

    dep_names = {d.name.lower() for d in (arch.dependencies or [])}
    for substrings, fw in _FRAMEWORK_PATTERNS:
        if dep_names.intersection(substrings):
            return fw

    # If nothing else fired but the LLM hinted at *some* framework,
    # preserve that signal as "other" so we can mine it later.
    if direct:
        return "other"
    return None


def _gradient_checkpointing(arch: ArchitectureRecord | None) -> bool:
    if arch is None:
        return False
    return _field_bool(arch.gradient_checkpointing) is True


def _has_grad_scaler(arch: ArchitectureRecord | None) -> bool:
    if arch is None or arch.precision is None:
        return False
    return _field_bool(arch.precision.grad_scaler) is True


# =============================================================
# Internal helpers — math, normalisation, schema-field unwrapping
# =============================================================

def _estimate_params(arch: ArchitectureRecord | None) -> int | None:
    """Approximate parameter count.

    Uses the standard transformer body formula `12 * L * H^2` plus
    `V * H` for the embedding when those facts are available. Falls
    back to None when the inputs aren't all present. The bucketing
    is coarse enough that order-of-magnitude estimates are fine.
    """
    if arch is None:
        return None
    num_layers = _field_int(arch.num_layers)
    hidden = _field_int(arch.hidden_size)
    if not num_layers or not hidden:
        return None
    body = 12 * num_layers * hidden * hidden
    vocab = _field_int(arch.vocab_size) or 0
    embedding = vocab * hidden
    return body + embedding


_HW_NORMALIZER = re.compile(r"[^a-z0-9_]")


def _normalize_hardware_name(name: str) -> str:
    """Collapse case + punctuation so '1x_A100' and '1x-a100' tie."""
    return _HW_NORMALIZER.sub("_", (name or "unknown").lower())


_SM_PATTERN = re.compile(r"^sm(\d{2,3})$")


def _supports_native_bf16(hardware: HardwareConfig | None) -> bool:
    """True iff the GPU has native bf16 tensor cores (Ampere+, sm80+).

    Returns False (safe default → mixed_fp16) when we can't determine
    the SM version.
    """
    if hardware is None:
        return False
    cap = (hardware.compute_capability or "").strip().lower()
    m = _SM_PATTERN.match(cap)
    if not m:
        return False
    return int(m.group(1)) >= _BF16_NATIVE_SM_MAJOR


def _stable_hash(d: dict[str, Any]) -> str:
    """sha256 of the JSON-encoded canonical dict, sorted keys."""
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _field_str(f: Any) -> str | None:
    """Unwrap ArchitectureField.value, tolerate raw strings and None."""
    if f is None:
        return None
    if isinstance(f, str):
        return f
    return getattr(f, "value", None)


def _field_int(f: Any) -> int | None:
    if isinstance(f, (int, float)):
        return int(f)
    val = _field_str(f)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _field_bool(f: Any) -> bool | None:
    if isinstance(f, bool):
        return f
    val = _field_str(f)
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        lowered = val.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return None
