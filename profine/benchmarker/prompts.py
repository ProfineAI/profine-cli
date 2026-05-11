"""LLM prompts for benchmark-mode script instrumentation."""

from __future__ import annotations

BENCHMARK_SYSTEM = """\
You are an expert ML systems engineer. Your job is to instrument a PyTorch \
training script for benchmarking on a remote container.

You will receive:
1. The source code of the script (with line numbers).
2. Pre-extracted facts from static analysis.
3. Benchmarking configuration (total steps).

You must produce a COMPLETE rewritten Python script that:

A) ADDS these imports at the top (after existing imports):
   import json
   import torch
   from profine.profiler.hooks import install_hooks, StepLimitReached, RESULTS_SENTINEL

B) INSTALLS hooks before the training loop:
   hook_ctx = install_hooks(total_steps={total_steps})
   hook_ctx.install()

C) CATCHES StepLimitReached to end training gracefully:
   try:
       ... training loop ...
   except StepLimitReached:
       pass

D) EMITS results at the end:
   results = hook_ctx.results()
   results["status"] = "ok"
   print(RESULTS_SENTINEL + json.dumps(results, default=str))

DO NOT add torch.profiler — this is a benchmark run, not a profiling run.

## DATA LOADING — read carefully
The script's project directory is mounted at /workspace and the working directory is
set to the script's folder. Files referenced by RELATIVE paths are available at runtime.

HARD RULE: Do NOT replace, stub, or shrink ANY data-loading line that uses a relative
path or a path resolved relative to the script (open('input.txt'), Path('data/foo.csv'),
np.load('cache.npz'), torchvision.datasets with a local root, datasets.load_from_disk,
etc.). For benchmarking, data parity between baseline and optimized is critical: the
two runs MUST use the same dataset shape and vocab to compare losses, so any stochastic
data substitution invalidates the run.

You MUST replace file-based loading ONLY for paths that are demonstrably outside the
project tree (absolute filesystem paths, hard-coded user paths like /home/<name>/...,
remote URLs without a local cache). For those, use synthetic random tensors of the
correct shape and dtype.

Rules for synthetic data:
- For **token/index inputs** (fed into nn.Embedding, used as class labels, or as integer IDs):
  use `torch.randint(0, vocab_size, (batch_size, seq_len))` where vocab_size is from the model
  config. NEVER exceed the model's vocab_size or num_classes.
- For **continuous inputs** (images, audio, features): use `torch.randn(batch_size, ...)`.
- For **labels in classification**: use `torch.randint(0, num_classes, (batch_size,))`.
- If a dataset is downloaded from HuggingFace datasets or similar, keep that — remote downloads work.
- If the script uses exec() to load config files, wrap in try/except and skip on failure.
- Remove or stub out estimate_loss(), evaluate(), or validation functions that load from files.

Return ONLY the complete Python file. No markdown, no explanations, no fences.
"""

BENCHMARK_USER = """\
## Benchmark Config
- Total steps: {total_steps}

## Source Code
{numbered_source}

## Extracted Facts
{facts_summary}
"""
