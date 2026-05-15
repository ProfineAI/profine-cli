"""End-to-end fingerprint test against a minGPT-shaped training script.

The reader+LLM step that profine normally runs to produce an
ArchitectureRecord is non-deterministic (an LLM call), so we do this
in two layers:

  1. AST extraction (deterministic) — run the real extractor against
     a minGPT-shaped source string and assert the surface facts we
     care about (optimizer, distributed, compile, autocast) are picked
     up correctly.

  2. Fingerprinting (deterministic) — feed a hand-built
     ArchitectureRecord that models what the LLM analyzer would
     produce for that same script and assert the resulting
     Fingerprint is bucketed sensibly.

Why split: we want the test to fail loudly if either layer regresses,
without making CI depend on an LLM call. The hand-built record is the
contract: if the analyzer's output stops matching it, the analyzer is
what's wrong (and we add a fixture from a real run).

The script fixtures live as Python strings here rather than gitignored
example dirs so the test is self-contained and CI-friendly.
"""

from __future__ import annotations

import textwrap

import pytest

from profine.reader.extractor import extract
from profine.schema.architecture_record import (
    ArchitectureField,
    ArchitectureRecord,
    DependencyInfo,
    DistributedInfo,
    OptimizerInfo,
    PrecisionInfo,
)
from profine.schema.hardware import HardwareConfig
from profine.telemetry import (
    arch_class_of,
    classify_crash,
    fingerprint_run,
    optimizer_class_of,
    param_bucket_of,
    precision_of,
)


# ===========================================================
# Fixtures: hardware presets and minGPT-shaped sources
# ===========================================================


def _hw(name: str, compute_capability: str = "sm80") -> HardwareConfig:
    """A HardwareConfig minimal enough for fingerprinting."""
    return HardwareConfig(
        name=name, label=name, modal_gpu="A100-80GB",
        gpu_count=1, gpu_kind="A100", vram_gb=80.0,
        compute_capability=compute_capability,
    )


# A condensed, minGPT-style training script. Real minGPT has more
# files; this is the trainer fragment that exercises the bits the
# extractor cares about.
MINGPT_SOURCE = textwrap.dedent("""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    class CausalSelfAttention(nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            self.qkv = nn.Linear(n_embd, 3 * n_embd)

    class Block(nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            self.attn = CausalSelfAttention(n_embd, n_head)
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
            )

    class GPT(nn.Module):
        def __init__(self, vocab_size, n_layer, n_embd, n_head):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, n_embd)
            self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
            self.lm_head = nn.Linear(n_embd, vocab_size)

    def train():
        model = GPT(vocab_size=50257, n_layer=12, n_embd=768, n_head=12)
        model = torch.compile(model, mode="default")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loader = DataLoader(dataset, batch_size=32, num_workers=4)

        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch)
                loss = nn.functional.cross_entropy(logits.view(-1, 50257), batch.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if __name__ == "__main__":
        train()
""")


# ===========================================================
# Layer 1: AST extraction picks up the right facts
# ===========================================================


def test_extractor_finds_optimizer_in_mingpt():
    facts = extract(MINGPT_SOURCE, "mingpt_train.py")
    optimizer_names = [c.name for c in facts.optimizer_calls]
    # AdamW is invoked as `torch.optim.AdamW(...)` — extractor stores
    # the (dotted) call name. We assert AdamW is present without
    # pinning the exact form, so a future extractor improvement that
    # normalises dotted names doesn't break this.
    assert any("AdamW" in n for n in optimizer_names), (
        f"AdamW not detected; got {optimizer_names}"
    )


def test_extractor_finds_compile_call_in_mingpt():
    facts = extract(MINGPT_SOURCE, "mingpt_train.py")
    assert facts.compile_calls, "torch.compile was not picked up"


def test_extractor_finds_autocast_call_in_mingpt():
    facts = extract(MINGPT_SOURCE, "mingpt_train.py")
    assert facts.autocast_calls, "torch.autocast was not picked up"


def test_extractor_finds_dataloader_in_mingpt():
    facts = extract(MINGPT_SOURCE, "mingpt_train.py")
    assert facts.dataloader_calls, "DataLoader was not picked up"


def test_extractor_finds_model_classes_in_mingpt():
    facts = extract(MINGPT_SOURCE, "mingpt_train.py")
    class_names = [c.name for c in facts.classes]
    assert "GPT" in class_names
    assert "Block" in class_names
    assert "CausalSelfAttention" in class_names


# ===========================================================
# Layer 2: hand-built ArchitectureRecord → Fingerprint
# ===========================================================
#
# The record below is what we'd expect the LLM analyzer to produce
# for the minGPT script above. minGPT vanilla is GPT (~124M params:
# 12 layers, 768 hidden, 50257 vocab), causal MHA, AdamW, autocast
# bf16, no distributed.


def _mingpt_record() -> ArchitectureRecord:
    return ArchitectureRecord(
        framework=ArchitectureField(value="raw_pytorch"),
        model_family=ArchitectureField(value="GPT"),
        model_class=ArchitectureField(value="GPT"),
        attention_type=ArchitectureField(value="causal_mha"),
        attention_impl=ArchitectureField(value="manual"),
        num_layers=ArchitectureField(value=12),
        hidden_size=ArchitectureField(value=768),
        vocab_size=ArchitectureField(value=50257),
        num_heads=ArchitectureField(value=12),
        optimizer=OptimizerInfo(
            name=ArchitectureField(value="AdamW"),
            learning_rate=ArchitectureField(value=3e-4),
        ),
        precision=PrecisionInfo(
            training_dtype=ArchitectureField(value="bf16"),
            autocast_enabled=ArchitectureField(value=True),
        ),
        compile_mode=ArchitectureField(value="default"),
        distributed=DistributedInfo(strategy=ArchitectureField(value="none")),
        dependencies=[DependencyInfo(name="torch", line=1)],
    )


def test_mingpt_fingerprint_buckets_correctly_on_a100():
    fp = fingerprint_run(_mingpt_record(), _hw("1x_a100", "sm80"))
    # ----- in-hash dims -----
    assert fp.arch_class == "transformer-decoder"
    assert fp.param_bucket == "100M-1B"   # 12 * 12 * 768^2 + 50257*768 ≈ 124M
    assert fp.hardware_class == "1x_a100"
    assert fp.precision == "mixed_bf16"
    assert fp.optimizer_class == "adam_family"
    assert fp.has_compile is True
    assert fp.has_distributed is False
    assert fp.fingerprint_hash  # non-empty sha
    # ----- recorded enrichment -----
    assert fp.compile_mode == "default"
    assert fp.distributed_strategy is None
    assert fp.attention_impl == "manual"
    assert fp.framework == "raw_pytorch"


def test_mingpt_fingerprint_stable_under_repeat():
    fp1 = fingerprint_run(_mingpt_record(), _hw("1x_a100"))
    fp2 = fingerprint_run(_mingpt_record(), _hw("1x_a100"))
    assert fp1.fingerprint_hash == fp2.fingerprint_hash


def test_mingpt_fingerprint_changes_with_hardware_class():
    fp_a = fingerprint_run(_mingpt_record(), _hw("1x_a100"))
    fp_h = fingerprint_run(_mingpt_record(), _hw("1x_h100", "sm90"))
    assert fp_a.fingerprint_hash != fp_h.fingerprint_hash


def test_mingpt_fingerprint_unaffected_by_learning_rate():
    """LR change must NOT move the bucket — it is a non-bucketed fact."""
    base = _mingpt_record()
    tweaked = _mingpt_record()
    tweaked.optimizer.learning_rate = ArchitectureField(value=1e-5)
    assert (
        fingerprint_run(base, _hw("1x_a100")).fingerprint_hash
        == fingerprint_run(tweaked, _hw("1x_a100")).fingerprint_hash
    )


# ===========================================================
# Layer 2 variant: minGPT scaled up (nanoGPT-medium-ish)
# ===========================================================


def test_nanogpt_medium_lands_in_100m_1b_bucket():
    """nanoGPT 'gpt2-medium' setting: 24 layers, 1024 hidden, 50257 vocab.
    12 * 24 * 1024^2 + 50257*1024 ≈ 302M + 51M = 353M params → 100M-1B."""
    record = _mingpt_record()
    record.num_layers = ArchitectureField(value=24)
    record.hidden_size = ArchitectureField(value=1024)
    fp = fingerprint_run(record, _hw("1x_a100"))
    assert fp.param_bucket == "100M-1B"


def test_llama_7b_lands_in_1b_7b_bucket():
    """LLaMA-7B: 32 layers, 4096 hidden, 32000 vocab.
    12 * 32 * 4096^2 + 32000*4096 ≈ 6.4B + 0.13B → 1B-7B."""
    record = _mingpt_record()
    record.model_family = ArchitectureField(value="LLaMA")
    record.num_layers = ArchitectureField(value=32)
    record.hidden_size = ArchitectureField(value=4096)
    record.vocab_size = ArchitectureField(value=32000)
    fp = fingerprint_run(record, _hw("1x_a100"))
    assert fp.arch_class == "transformer-decoder"
    assert fp.param_bucket == "1B-7B"


# ===========================================================
# Layer 2 variant: hardware precision sensitivity
# ===========================================================


def test_fp32_autocast_picks_bf16_on_ampere_plus():
    """fp32 + autocast on A100 (sm80) → mixed_bf16."""
    record = _mingpt_record()
    record.precision = PrecisionInfo(
        training_dtype=ArchitectureField(value="fp32"),
        autocast_enabled=ArchitectureField(value=True),
    )
    fp = fingerprint_run(record, _hw("1x_a100", compute_capability="sm80"))
    assert fp.precision == "mixed_bf16"


def test_fp32_autocast_falls_back_to_fp16_on_pre_ampere():
    """fp32 + autocast on a T4 (sm75, no native bf16) → mixed_fp16."""
    record = _mingpt_record()
    record.precision = PrecisionInfo(
        training_dtype=ArchitectureField(value="fp32"),
        autocast_enabled=ArchitectureField(value=True),
    )
    fp = fingerprint_run(record, _hw("1x_t4", compute_capability="sm75"))
    assert fp.precision == "mixed_fp16"


def test_fp32_autocast_falls_back_on_missing_compute_capability():
    """Safe default when the hardware record lacks compute_capability."""
    record = _mingpt_record()
    record.precision = PrecisionInfo(
        training_dtype=ArchitectureField(value="fp32"),
        autocast_enabled=ArchitectureField(value=True),
    )
    hw = HardwareConfig(
        name="mystery_gpu", label="?", modal_gpu="?",
        gpu_count=1, gpu_kind="?", vram_gb=16.0,
        # compute_capability deliberately omitted → empty string
    )
    fp = fingerprint_run(record, hw)
    assert fp.precision == "mixed_fp16"


# ===========================================================
# Real-job crash-class coverage (rooted in minGPT-class failures)
# ===========================================================


@pytest.mark.parametrize("err,expected", [
    # OOM — common when batch_size + n_embd combine wrong
    ("torch.cuda.OutOfMemoryError: CUDA out of memory.", "oom"),
    # NaN — happens with too-high LR or fp16 underflow
    ("RuntimeError: loss is NaN at step 47", "nan_loss"),
    # Shape — common when context_length is changed without updating embeddings
    ("RuntimeError: The size of tensor a (512) must match the size of tensor b (1024)", "shape_mismatch"),
    # torch.compile — common with dynamic batch shapes
    ("torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised", "compile_fail"),
    # Script bug — typo or wrong attr on model
    ("AttributeError: 'GPT' object has no attribute 'tokenizer'", "script_bug"),
    # Dataloader — dataset path missing
    ("RuntimeError: DataLoader worker (pid 12345) exited unexpectedly", "dataloader"),
    # Distributed init — typical NCCL bring-up failure
    ("RuntimeError: NCCL failed during init_process_group", "dist_init"),
    # Distributed collective mid-run
    ("RuntimeError: NCCL communicator timeout on AllReduce", "dist_collective"),
    # Missing dep — common with flash-attn
    ("ModuleNotFoundError: No module named 'flash_attn'", "dep_missing"),
    # Auth — HF gated model
    ("HfHubHTTPError: 401 Client Error: Unauthorized for url https://huggingface.co/...", "auth"),
    # Network
    ("ConnectionError: Max retries exceeded with url", "network"),
    # Disk full mid-checkpoint
    ("OSError: [Errno 28] No space left on device", "disk"),
    # Wall-clock
    ("RuntimeError: Container killed: wall-clock limit exceeded", "timeout"),
    # OOM-killer
    ("Process killed by SIGKILL (oom-killer)", "process_killed"),
    # Kernel-level CUDA error
    ("RuntimeError: CUDA error: an illegal memory access was encountered", "kernel_error"),
])
def test_realistic_mingpt_class_crashes_get_bucketed(err, expected):
    assert classify_crash(err) == expected
