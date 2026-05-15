"""Tests for the telemetry fingerprinter and the privacy allowlist."""

from __future__ import annotations

import pytest

from profine.schema.architecture_record import (
    ArchitectureField,
    ArchitectureRecord,
    DistributedInfo,
    OptimizerInfo,
    PrecisionInfo,
)
from profine.schema.hardware import HardwareConfig
from profine.telemetry import (
    Fingerprint,
    arch_class_of,
    classify_crash,
    fingerprint_run,
    optimizer_class_of,
    param_bucket_of,
    precision_of,
)
from profine.telemetry.crash_class import CRASH_CLASSES
from profine.telemetry.fields import (
    ALLOWED_FINGERPRINT_FIELDS,
    ALLOWED_OUTCOME_FIELDS,
    filter_fingerprint,
    filter_outcome,
)
from profine.telemetry.fingerprint import (
    ARCH_CLASSES,
    OPTIMIZER_CLASSES,
    PRECISIONS,
)


# ----------------------------- helpers ------------------------------------


def _hw(name: str = "1x_a100") -> HardwareConfig:
    return HardwareConfig(
        name=name, label=name, modal_gpu="A100-80GB",
        gpu_count=1, gpu_kind="A100", vram_gb=80.0,
    )


def _arch(**fields) -> ArchitectureRecord:
    """Build an ArchitectureRecord by wrapping plain values in ArchitectureField."""
    out: dict = {}
    nested = {}
    for k, v in fields.items():
        if k in ("optimizer", "precision", "distributed"):
            nested[k] = v
        elif isinstance(v, ArchitectureField) or v is None:
            out[k] = v
        else:
            out[k] = ArchitectureField(value=v)
    out.update(nested)
    return ArchitectureRecord(**out)


# ----------------------------- arch_class --------------------------------


@pytest.mark.parametrize(
    "model_family,attention,expected",
    [
        ("LLaMA",           "causal_mha",    "transformer-decoder"),
        ("Mistral",         "gqa",           "transformer-decoder"),
        ("BERT",            "bidirectional", "transformer-encoder"),
        ("T5",              "",              "transformer-enc-dec"),
        ("ViT",             "",              "vit"),
        ("ResNet50",        "",              "cnn"),
        ("EfficientNet",    "",              "cnn"),
        ("StableDiffusion", "",              "diffusion"),
        ("LSTM",            "",              "rnn"),
        ("custom_thing",    "",              "other"),
    ],
)
def test_arch_class_basic(model_family, attention, expected):
    fields = {"model_family": model_family}
    if attention:
        fields["attention_type"] = attention
    if expected == "transformer-enc-dec":
        fields["model_family"] = "T5 encoder-decoder"
    assert arch_class_of(_arch(**fields)) == expected


def test_arch_class_none_arch():
    assert arch_class_of(None) == "other"


def test_arch_class_within_allowed_set():
    # Throw a bunch of random unstructured names; result must always
    # land in ARCH_CLASSES.
    for name in ("foo", "GPT-2", "RWKV", "Mamba", "RetNet", "SSM"):
        assert arch_class_of(_arch(model_family=name)) in ARCH_CLASSES


# ----------------------------- optimizer_class ---------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Adam",       "adam_family"),
        ("AdamW",      "adam_family"),
        ("NAdam",      "adam_family"),
        ("SGD",        "sgd_family"),
        ("Momentum",   "sgd_family"),
        ("Adafactor",  "adafactor"),
        ("Lamb",       "lamb"),
        ("Lion",       "lion"),
        ("Shampoo",    "shampoo"),
        ("RandomThing","other"),
        ("",           "other"),
    ],
)
def test_optimizer_class(name, expected):
    arch = _arch(optimizer=OptimizerInfo(name=ArchitectureField(value=name) if name else None))
    assert optimizer_class_of(arch) == expected


def test_optimizer_class_none():
    assert optimizer_class_of(None) == "other"
    assert optimizer_class_of(_arch()) == "other"


def test_optimizer_class_within_allowed_set():
    for n in ("foo", "BatchedAdam", "AnchoredSGD"):
        arch = _arch(optimizer=OptimizerInfo(name=ArchitectureField(value=n)))
        assert optimizer_class_of(arch) in OPTIMIZER_CLASSES


# ----------------------------- param_bucket ------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [
        (None, "unknown"),
        (0, "unknown"),
        (-5, "unknown"),
        # New buckets cover small models (RL policies, LoRA, small CNNs)
        (1_000, "<1M"),
        (999_999, "<1M"),
        (1_000_000, "1M-10M"),       # small CNNs, LoRA adapters
        (9_999_999, "1M-10M"),
        (10_000_000, "10M-100M"),    # ResNet50, distilBERT, GPT-2 small
        (50_000_000, "10M-100M"),
        (100_000_000, "100M-1B"),    # ViT-L, BERT-large
        (999_999_999, "100M-1B"),
        (1_000_000_000, "1B-7B"),
        (7_000_000_000, "7B-13B"),
        (13_000_000_000, "13B-70B"),
        (70_000_000_000, "70B+"),
        (175_000_000_000, "70B+"),
    ],
)
def test_param_bucket_edges(n, expected):
    assert param_bucket_of(n) == expected


# ----------------------------- precision ---------------------------------


@pytest.mark.parametrize(
    "dtype,autocast,expected",
    [
        ("fp32",       False, "fp32"),
        ("float32",    True,  "mixed_fp16"),   # fp32 + autocast -> conservative default
        ("fp16",       False, "fp16"),
        ("fp16",       True,  "mixed_fp16"),
        ("bf16",       False, "bf16"),
        ("bfloat16",   True,  "mixed_bf16"),
        ("fp8",        False, "fp8"),
        ("",           False, "unknown"),
        ("nonsense",   False, "unknown"),
    ],
)
def test_precision(dtype, autocast, expected):
    arch = _arch(precision=PrecisionInfo(
        training_dtype=ArchitectureField(value=dtype) if dtype else None,
        autocast_enabled=ArchitectureField(value=autocast),
    ))
    assert precision_of(arch) == expected


def test_precision_none_arch():
    assert precision_of(None) == "unknown"


def test_precision_within_allowed_set():
    for d, ac in (("anything", True), ("anything", False), ("", True)):
        arch = _arch(precision=PrecisionInfo(
            training_dtype=ArchitectureField(value=d) if d else None,
            autocast_enabled=ArchitectureField(value=ac),
        ))
        assert precision_of(arch) in PRECISIONS


# ----------------------------- full fingerprint --------------------------


def test_fingerprint_stable_under_repeat():
    arch = _arch(
        model_family="LLaMA",
        attention_type="causal_mha",
        num_layers=32,
        hidden_size=4096,
        vocab_size=32000,
        optimizer=OptimizerInfo(name=ArchitectureField(value="AdamW")),
        precision=PrecisionInfo(
            training_dtype=ArchitectureField(value="bf16"),
            autocast_enabled=ArchitectureField(value=True),
        ),
        compile_mode="default",
        distributed=DistributedInfo(strategy=ArchitectureField(value="ddp")),
    )
    fp1 = fingerprint_run(arch, _hw())
    fp2 = fingerprint_run(arch, _hw())
    assert fp1.fingerprint_hash == fp2.fingerprint_hash
    assert fp1 == fp2


def test_fingerprint_differs_on_arch_class_change():
    base = _arch(model_family="LLaMA", attention_type="causal_mha")
    other = _arch(model_family="ResNet50")
    assert (
        fingerprint_run(base, _hw()).fingerprint_hash
        != fingerprint_run(other, _hw()).fingerprint_hash
    )


def test_fingerprint_differs_on_hardware():
    arch = _arch(model_family="LLaMA", attention_type="causal_mha")
    fp_a100 = fingerprint_run(arch, _hw("1x_a100"))
    fp_h100 = fingerprint_run(arch, _hw("1x_h100"))
    assert fp_a100.fingerprint_hash != fp_h100.fingerprint_hash


def test_fingerprint_ignores_non_bucketed_facts():
    """Two runs that differ only in unbucketed fields (e.g. lr) tie."""
    arch1 = _arch(
        model_family="LLaMA",
        attention_type="causal_mha",
        num_layers=32,
        hidden_size=4096,
        optimizer=OptimizerInfo(
            name=ArchitectureField(value="AdamW"),
            learning_rate=ArchitectureField(value=0.001),
        ),
    )
    arch2 = _arch(
        model_family="LLaMA",
        attention_type="causal_mha",
        num_layers=32,
        hidden_size=4096,
        optimizer=OptimizerInfo(
            name=ArchitectureField(value="AdamW"),
            learning_rate=ArchitectureField(value=0.0003),  # different lr
        ),
    )
    assert fingerprint_run(arch1, _hw()).fingerprint_hash == fingerprint_run(arch2, _hw()).fingerprint_hash


def test_fingerprint_hardware_name_normalized():
    """Casing or punctuation variants tie."""
    arch = _arch(model_family="LLaMA", attention_type="causal_mha")
    fp_lower = fingerprint_run(arch, _hw("1x_a100"))
    fp_mixed = fingerprint_run(arch, _hw("1X-A100"))
    assert fp_lower.fingerprint_hash == fp_mixed.fingerprint_hash


def test_fingerprint_none_arch_does_not_crash():
    fp = fingerprint_run(None, _hw())
    assert fp.arch_class == "other"
    assert fp.param_bucket == "unknown"
    assert fp.precision == "unknown"
    assert fp.fingerprint_hash  # non-empty


def test_fingerprint_is_frozen():
    fp = fingerprint_run(None, _hw())
    with pytest.raises(Exception):
        fp.arch_class = "transformer-decoder"  # frozen dataclass


# ----------------------------- privacy allowlist --------------------------


def test_filter_fingerprint_drops_unknown_keys():
    payload = {
        "arch_class": "transformer-decoder",
        "fingerprint_hash": "deadbeef",
        # Unsafe extras that must not survive:
        "script_path": "/users/foo/secret.py",
        "user_email": "leak@example.com",
        "raw_traceback": "AssertionError at line 42",
    }
    out = filter_fingerprint(payload)
    assert "script_path" not in out
    assert "user_email" not in out
    assert "raw_traceback" not in out
    assert out["arch_class"] == "transformer-decoder"


def test_filter_outcome_drops_unknown_keys():
    payload = {
        "optimization_id": "compile_default",
        "speedup_factor": 1.4,
        "applied": True,
        "crashed": False,
        # Unsafe:
        "dataset_path": "/data/foo",
        "exception_message": "ValueError: bad",
    }
    out = filter_outcome(payload)
    assert "dataset_path" not in out
    assert "exception_message" not in out
    assert out["optimization_id"] == "compile_default"


def test_allowlist_sets_are_disjoint():
    """A field in fingerprint should never also be in outcome and vice versa;
    that would make audits ambiguous."""
    assert not (ALLOWED_FINGERPRINT_FIELDS & ALLOWED_OUTCOME_FIELDS)


# ----------------------------- crash classifier --------------------------


@pytest.mark.parametrize(
    "err,expected",
    [
        ("CUDA out of memory.",                              "oom"),
        ("torch.cuda.OutOfMemoryError",                      "oom"),
        ("BackendCompilerFailed: ...",                       "compile_fail"),
        ("TorchDynamo error: recompile limit",               "compile_fail"),
        ("loss is NaN at step 42",                           "nan_loss"),
        ("NaN/Inf in gradients detected",                    "nan_loss"),
        ("ModuleNotFoundError: No module named 'flash_attn'", "dep_missing"),
        ("ImportError: cannot import name",                  "dep_missing"),
        ("Container killed: wall-clock limit",               "timeout"),
        ("Timed out waiting for ...",                        "timeout"),
        ("DataLoader worker exited",                         "dataloader"),
        ("CUDA: illegal memory access",                      "kernel_error"),
        ("cuBLAS internal error",                            "kernel_error"),
        ("some weird condition we have no rule for",         "other"),
    ],
)
def test_classify_crash(err, expected):
    assert classify_crash(err) == expected


def test_classify_crash_none_and_empty():
    assert classify_crash(None) is None
    assert classify_crash("") is None
    assert classify_crash("   ") is None


def test_classify_crash_accepts_exception_object():
    exc = RuntimeError("CUDA out of memory")
    assert classify_crash(exc) == "oom"


def test_classify_crash_returns_only_known_classes():
    weird_inputs = [
        "asdf 1234",
        "Connection refused",
        "Permission denied",
        "Unknown error 0xDEADBEEF",
    ]
    for s in weird_inputs:
        result = classify_crash(s)
        assert result in CRASH_CLASSES, f"unexpected class {result!r} for {s!r}"
