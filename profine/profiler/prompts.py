"""LLM prompts for script instrumentation and error recovery."""

from __future__ import annotations

from profine.reader.extractor import CodeFacts


# Instrumentation prompt

INSTRUMENTATION_SYSTEM = """\
You are an expert ML systems engineer. Your job is to instrument a PyTorch \
training script for GPU profiling on a remote container.

You will receive:
1. The source code of the script (with line numbers).
2. Pre-extracted facts from static analysis.
3. Profiling configuration (total steps, warmup steps).

You must produce a COMPLETE rewritten Python script that:

A) ADDS these imports at the top (after existing imports):
   import json
   import torch
   from torch.profiler import profile, ProfilerActivity, schedule
   from profine.profiler.hooks import install_hooks, StepLimitReached, emit_results, RESULTS_SENTINEL

B) INSTALLS hooks before the training loop:
   hook_ctx = install_hooks(total_steps={total_steps})
   hook_ctx.install()
   hook_ctx.step_controller.set_profiler(profiler_instance)

C) WRAPS the training loop with torch.profiler.profile():
   with profile(
       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
       schedule=schedule(wait=0, warmup={warmup_steps}, active={active_steps}, repeat=1),
       record_shapes=True,
       with_flops=True,
       acc_events=True,
   ) as prof:
       ... training loop ...

D) CATCHES StepLimitReached to end training gracefully:
   try:
       ... training loop ...
   except StepLimitReached:
       pass

E) COLLECTS profiler events and emits results at the end:
   After the training loop, add:
   profiler_events = []
   for avg in prof.key_averages():
       profiler_events.append({{
           "name": getattr(avg, "key", str(avg)),
           "self_cpu_time_total_us": getattr(avg, "self_cpu_time_total", 0.0),
           "self_cuda_time_total_us": getattr(avg, "self_device_time_total", 0.0) or getattr(avg, "self_cuda_time_total", 0.0),
           "flops": getattr(avg, "flops", 0.0),
           "count": getattr(avg, "count", 1),
           "input_dtypes": [str(d) for d in (getattr(avg, "input_type", None) or [])],
       }})
   results = hook_ctx.results()
   results["profiler_events"] = profiler_events
   results["status"] = "ok"
   print(RESULTS_SENTINEL + json.dumps(results, default=str))

## CRITICAL RULES
- Do NOT remove or alter the actual training logic, model, optimizer, loss, or hyperparameters.
- For HuggingFace Trainer scripts: set trainer.args.max_steps = {total_steps} before trainer.train().
- For raw PyTorch scripts: find the training for/while loop and wrap it.
- Keep all original imports, argument parsing, and data loading.
- The profiler hooks handle step counting and memory — you just wrap the loop.
- If the script has argparse or sys.argv parsing, preserve it.
- Return ONLY the complete Python file. No markdown, no explanations, no fences.

## DATA LOADING — read carefully, this is the #1 source of broken instrumentation
The script's project directory is mounted at /workspace and the working directory is
set to the script's folder. Any file referenced by a RELATIVE path (e.g. open('input.txt'),
open('./data/foo.csv'), Path('input.txt')) IS available at runtime.

HARD RULE: Do NOT replace, stub, or shrink ANY data-loading line that uses a relative
path or a path resolved relative to the script. This includes open(...), Path(...).read_*,
np.load, pd.read_csv, np.memmap, json.load, torch.load, torchvision.datasets with a local
root, datasets.load_from_disk, etc. Keep those lines exactly as written. Even if the file
looks "large" or "test-data-like", assume it is present. Replacing it changes the
dataset's vocab/shape and silently invalidates downstream code (sample callbacks,
vocab-keyed dicts, embedding sizes derived from data).

You MUST replace file-based loading ONLY for paths that are demonstrably outside the
project tree:
  - Absolute filesystem paths (/datasets/..., /scratch/..., ~/data/...)
  - Hard-coded user paths (/home/<name>/..., C:\\..., /Users/...)
  - URLs to data buckets (s3://, gs://) when no local cache fallback exists
For those, replace with synthetic random tensors of the correct shape and dtype. If a
shape parameter is unknowable from the source, prefer keeping the loader and letting
profiling fail fast — a wrong-shape synthetic dataset is worse than a clear file-not-found.

When you DO substitute synthetic data, also stub any code further down that depends on
properties of the real data (eval/sample callbacks that index a vocab built from the
real text) — comment them out or guard them so a synthetic dataset doesn't trigger
KeyErrors or out-of-range indices.

Rules for synthetic data:
- For **token/index inputs** (fed into nn.Embedding, used as class labels, or as integer IDs):
  use `torch.randint(0, vocab_size, (batch_size, seq_len))` where vocab_size is from the model
  config. NEVER exceed the model's vocab_size or num_classes — out-of-range indices cause CUDA
  assertion failures.
- For **continuous inputs** (images, audio, features): use `torch.randn(batch_size, ...)`.
- For **labels in classification**: use `torch.randint(0, num_classes, (batch_size,))`.
- Keep batch_size, block_size, sequence_length, and other shape parameters from the script unchanged.
- If a dataset is downloaded from HuggingFace datasets or similar, keep that — remote downloads work.
- If the script uses exec() to load config files (e.g. exec(open('configurator.py').read())),
  wrap those in try/except and skip them on failure — they are optional CLI config overrides.
- Also remove or stub out any `estimate_loss()`, `evaluate()`, or validation functions that load
  from files. For profiling we only need the training loop, not evaluation.
"""

INSTRUMENTATION_USER = """\
## Profiling Config
- Total steps: {total_steps}
- Warmup steps: {warmup_steps}
- Active steps (for torch.profiler): {active_steps}

## Source Code
{numbered_source}

## Extracted Facts
{facts_summary}
"""

def build_instrumentation_prompt(
    source: str,
    facts: CodeFacts,
    total_steps: int,
    warmup_steps: int,
    active_steps: int,
    benchmark_mode: bool = False,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for instrumentation.

    Returns:
        (system, user) prompt strings.
    """
    # Summarize facts concisely for the prompt
    facts_lines = []
    if facts.model_loader_calls:
        facts_lines.append(f"Model loaders: {[(c.name, c.line) for c in facts.model_loader_calls]}")
    if facts.optimizer_calls:
        facts_lines.append(f"Optimizers: {[(c.name, c.line) for c in facts.optimizer_calls]}")
    if facts.dataloader_calls:
        facts_lines.append(f"DataLoaders: {[(c.name, c.line) for c in facts.dataloader_calls]}")
    if facts.distributed_calls:
        facts_lines.append(f"Distributed: {[(c.name, c.line) for c in facts.distributed_calls]}")
    if facts.autocast_calls:
        facts_lines.append(f"Autocast: {[(c.name, c.line) for c in facts.autocast_calls]}")
    if facts.compile_calls:
        facts_lines.append(f"Compile: {[(c.name, c.line) for c in facts.compile_calls]}")

    has_trainer = any("Trainer" in c.name for c in facts.calls)
    has_training_args = any("TrainingArguments" in c.name for c in facts.calls)
    framework = "HuggingFace Trainer" if (has_trainer or has_training_args) else "Raw PyTorch"
    facts_lines.insert(0, f"Framework: {framework}")

    numbered = _numbered_source(source)
    facts_str = "\n".join(facts_lines) if facts_lines else "No significant patterns detected."

    if benchmark_mode:
        from profine.benchmarker.prompts import BENCHMARK_SYSTEM, BENCHMARK_USER
        system = BENCHMARK_SYSTEM.format(total_steps=total_steps)
        user = BENCHMARK_USER.format(
            total_steps=total_steps,
            numbered_source=numbered,
            facts_summary=facts_str,
        )
    else:
        system = INSTRUMENTATION_SYSTEM.format(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            active_steps=active_steps,
        )
        user = INSTRUMENTATION_USER.format(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            active_steps=active_steps,
            numbered_source=numbered,
            facts_summary=facts_str,
        )
    return system, user


# Error healing prompt

HEALING_SYSTEM = """\
You are an expert ML systems engineer. A profiling-instrumented training script \
crashed during remote execution. Your job is to fix the script so it runs successfully.

You will receive:
1. The instrumented script that crashed.
2. The error traceback.
3. The original (uninstrumented) script for reference.

Fix ONLY the issue causing the crash. Do not remove the profiler instrumentation.

NEVER change the model name/path, dataset, or hyperparameters. The user chose that \
model deliberately — if the crash is caused by a missing dependency, add the import \
or install rather than swapping to a different model.

Common issues:
- Missing imports
- FileNotFoundError for data files: The script runs on a remote container with NO local data \
files. Replace file-based data loading (np.memmap, open(), pd.read_csv) with synthetic random \
tensors of the correct shape and dtype.
- CUDA index out of bounds / embedding assertion failures: synthetic token data must use \
torch.randint(0, vocab_size, ...) where vocab_size matches the model's embedding table. \
Never exceed vocab_size. For targets/labels, use the same range constraint.
- Validation/eval functions that try to load data files: stub them out or make them use \
the same synthetic data as training.
- Argument parsing errors (Modal doesn't pass CLI args)
- exec(open(...).read()) failing: wrap in try/except and skip (optional config overrides)
- CUDA/GPU initialization issues
- Module not found errors

Return ONLY the complete fixed Python file. No markdown, no explanations.
"""

HEALING_USER = """\
## Error Traceback
{traceback}

## Instrumented Script (crashed)
{instrumented_source}

## Original Script (reference)
{original_source}
"""


def build_healing_prompt(
    instrumented_source: str,
    error_traceback: str,
    original_source: str,
) -> tuple[str, str]:
    """Build (system, user) prompt for error recovery."""
    user = HEALING_USER.format(
        traceback=error_traceback,
        instrumented_source=_numbered_source(instrumented_source),
        original_source=_numbered_source(original_source),
    )
    return HEALING_SYSTEM, user


# Dependency validation prompt

DEPENDENCY_VALIDATION_SYSTEM = """\
You are an expert at Python packaging. Given a training script and its \
statically-discovered dependencies, identify any MISSING pip packages.

The remote environment pre-installs:
- Python + standard library
- torch, torchvision, torchaudio

Rules:
- Return ONLY a JSON array of pip package names to ADD (not the full list).
- Use correct pip names (e.g. "opencv-python" not "cv2", "PyYAML" not "yaml").
- Only include packages the script actually imports or requires at runtime.
- If the list is already complete, return [].
- No markdown, no explanation — just the JSON array.
"""

DEPENDENCY_VALIDATION_USER = """\
## Dependencies discovered by static analysis
{deps_json}

## Script
{source}
"""


def build_dependency_validation_prompt(
    source: str,
    discovered_deps: list[str],
) -> tuple[str, str]:
    """Build (system, user) for LLM dependency validation."""
    import json
    user = DEPENDENCY_VALIDATION_USER.format(
        deps_json=json.dumps(discovered_deps, indent=2),
        source=source,
    )
    return DEPENDENCY_VALIDATION_SYSTEM, user


def _numbered_source(source: str) -> str:
    lines = source.splitlines()
    return "\n".join(f"{i+1:>4} | {line}" for i, line in enumerate(lines))
