# profine

[![PyPI](https://img.shields.io/pypi/v/profine.svg)](https://pypi.org/project/profine/)
[![CI](https://github.com/ProfineAI/profine-cli/actions/workflows/test.yml/badge.svg)](https://github.com/ProfineAI/profine-cli/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/profine.svg)](https://pypi.org/project/profine/)

**Check us out at** [profine.ai](https://profine.ai/)

Profile your PyTorch code on real GPUs. Get reviewable optimizations. Ship measured speedups before the multi-hour run.

▶ [**Watch the demo on YouTube**](https://youtu.be/CY9aW1Dcrn0)

## Quickstart

```bash
pip install profine
profine auth login                                                            # one-time
profine run-all examples/minGPT/projects/chargpt/chargpt.py --hardware 1x_a100
```

profine prints a one-line cost summary, runs the full pipeline on Modal, and produces a benchmark report in ~10 minutes.

## Results

On [Karpathy's minGPT](https://github.com/karpathy/minGPT) `chargpt` config, **median of 3 independent runs per GPU**, full optimization stack applied (BF16 Mixed Precision + TF32 matmul + torch.compile max-autotune + SDPA + Fused AdamW):

| GPU | Baseline step | Optimized step | Speedup | Peak mem Δ |
|---|---|---|---|---|
| **A10G** (24 GB) | 43.8 ms | 16.5 ms | **2.75× faster** (63.7%) | −71.1% |
| **A100** (80 GB) | 25.2 ms | 7.5 ms | **3.48× faster** (71.3%) | −68.7% |

Per-run speedups (3 reps each): A10G 2.42× / 2.75× / 4.73×; A100 2.14× / 3.48× / 3.51×.
Correctness is checked by replaying baseline and optimized loss curves step-for-step on the same seed; both stay inside the BF16-widened tolerance (`rtol=0.05, atol=0.01`, the documented bf16-vs-fp32 drift budget) on every rep. Median loss-curve max diff: 0.013 (A10G), 0.098 (A100).

Reproducible:

```bash
profine run-all examples/minGPT/projects/chargpt/chargpt.py --hardware 1x_a100
```

Full artifacts in [`examples/minGPT/profine_output/`](examples/minGPT/profine_output/) (start with [`SUMMARY.md`](examples/minGPT/profine_output/SUMMARY.md)); the multi-rep comparison data lives under [`runs/bench_mingpt/`](runs/bench_mingpt/).

## Setup

Requires:
- A [Modal](https://modal.com) account (the GPU backend)
- An LLM: OpenAI, Anthropic, or any OpenAI-compatible local server (Ollama, vLLM, LM Studio, llama.cpp, LiteLLM)

The fastest path is `profine auth login` which is an interactive prompt that saves keys to `~/.profine/auth.json` (chmod 0600):

```bash
profine auth login        # paste in MODAL_*, OPENAI/ANTHROPIC, HF_TOKEN
profine auth status       # show what's saved (redacted)
profine auth set OPENAI_API_KEY sk-...
profine auth logout                         # clear all
profine auth logout OPENAI_API_KEY          # clear one
```

Environment variables always win over the saved file, so CI keeps working:

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
export OPENAI_API_KEY=...      # or ANTHROPIC_API_KEY
export HF_TOKEN=...            # optional, gated models only
```

### Local LLMs

profine talks to any OpenAI-compatible server. Pass `--provider local` plus `--model`, and optionally `--base-url`.

```bash
# Ollama (default endpoint http://localhost:11434/v1)
ollama serve &
ollama pull llama3.1:8b
profine run-all train.py --provider local --model llama3.1:8b

# vLLM, LM Studio, llama.cpp server, or LiteLLM. Point --base-url at the server.
profine run-all train.py \
  --provider local \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1
```

`--base-url` is also picked up from `PROFINE_LOCAL_BASE_URL`.

> The agent loop expects strong instruction-following and clean JSON. Models ≤7B often fail at the `interpret`/`suggest`/`edit` steps; we recommend 70B-class or larger for end-to-end reliability.

## How it works

```
read → profile → interpret → suggest → edit → benchmark
```

Each stage reads the previous stage's output from `profine_output/` and writes its own. `run-all` chains them all; the individual `profine <stage>` commands let you re-run any single step.

### Features

- **Pre-flight cost summary**: one inline line before the run, no prompt unless the estimate exceeds `$5` (override via `PROFINE_COST_PROMPT_THRESHOLD`).
- **Resume on failure**: re-run the same command after any mid-pipeline crash; stages with existing artifacts under `--output` are skipped. Pass `--no-resume` to force a clean run.
- **Probe-and-adapt**: if step times in your script are slow enough that the configured `--steps` would overshoot the wall-clock budget, profine measures the actual step time after a few probe iterations and trims `total_steps` so the run finishes inside the budget.
- **Auto-peel on regression**: if a runtime crash on the optimized run can't be healed, profine drops the most recent optimization from the stack and re-benchmarks. Loop repeats until success or only one optimization is left.
- **Honest confidence intervals**: the benchmark report shows per-run p25/p50/p75 + CV. The headline speedup adds a `lo×–hi×` band when the run is noisy.

### Global flags (every stage)

| Flag | Default | Description |
|---|---|---|
| `--provider` | `openai` | `openai`, `anthropic`, or `local` |
| `--api-key` | from auth/env | Overrides saved auth + env var |
| `--model` | provider default | Required for `--provider local` |
| `--base-url` | none | For `--provider local`; env: `PROFINE_LOCAL_BASE_URL` |
| `--seed` | `42` | LLM seed. Temperature is always 0 |
| `-o/--output` | `profine_output` | Output directory |
| `--prefs` | none | Markdown of user preferences (biases ranking + edits) |
| `--no-telemetry` | off | Disable anonymous telemetry for this run |

Run `profine env` to see every `PROFINE_*` variable profine reads with its current resolved value.

## Stages

### `run-all`

```bash
profine run-all examples/minGPT/projects/chargpt/chargpt.py
```

| Flag | Default | Description |
|---|---|---|
| `--hardware` | required | Preset name. See [Hardware](#hardware) |
| `--steps` | `60` | Total measured steps |
| `--warmup` | `30` | Warmup steps (stripped before measurement) |
| `--timeout` | `900` | Modal container timeout (s). Auto-extends on timeout |
| `--warmstart` | off | Reuse the deployed Modal app between runs |
| `--top` | all | Apply top N optimizations sequentially, each stacked on the previous |
| `--rtol` / `--atol` | `0.01` / `0.0001` | Loss tolerances (auto-widened for BF16/FP16, quantization) |
| `--no-resume` | off | Re-run every stage from scratch |
| `--yes`, `-y` | off | Skip the cost prompt |

### `read`

```bash
profine read train.py
```

Reads model/optimizer/dataloader/precision/distributed-strategy facts via AST + LLM, plus any local modules the script imports. Output: `profine_output/read/architecture_record.json`.

### `profile`

```bash
profine profile train.py
```

Instruments the script and runs it on Modal with `torch.profiler`. Collects step times, kernel breakdown, GPU utilization, memory. Same `--hardware` / `--steps` / `--warmup` / `--timeout` / `--warmstart` flags as `run-all`. Output: `profine_output/profile/profile_record.json`.

### `interpret`

```bash
profine interpret --profile-dir profine_output/profile
```

Deterministic analysis + LLM diagnosis. Output: `profine_output/interpret/bottleneck_report.json`.

### `suggest`

```bash
profine suggest --interpret-dir profine_output/interpret
```

Filters the catalog by applicability, then ranks remaining candidates by ROI. Output: `profine_output/suggest/suggestion_report.json`.

### `edit`

```bash
profine edit train.py --suggestion-dir profine_output/suggest          # top-ranked only
profine edit train.py --suggestion-dir profine_output/suggest --top 3  # stack the top 3
profine edit train.py --suggestion-dir profine_output/suggest --optimization torch_compile
```

Multi-file aware: discovers local modules the entry script imports and edits whichever file owns the code being optimized. Patched library files land under `profine_output/edit/files/<rel-path>`, and your source tree is never touched. With `--top N`, per-iteration artifacts go in `profine_output/edit/NN_<entry_id>/`; cumulative result at `profine_output/edit/edited_train.py`.

### `benchmark`

```bash
profine benchmark train.py                                          # uses <output>/edit/edited_train.py
profine benchmark train.py --optimized profine_output/edit/edited_train.py
```

Runs original and optimized back-to-back on the same hardware. Files under `profine_output/edit/files/` are overlaid on the optimized run. Loss tolerance auto-widens for numerics-perturbing optimizations (BF16/mixed precision: rtol 5%; quantization: rtol 10%). When widened, the headline verdict surfaces it explicitly. Output: `profine_output/benchmark/`.

## Hardware

Hardware presets live in [`profine/config/hardware.yaml`](profine/config/hardware.yaml). Pass one explicitly via `--hardware`.

| Preset | GPU | VRAM | Cost/hr |
|---|---|---|---|
| `1x_t4` | T4 | 16 GB | $0.59 |
| `1x_l4` | L4 | 24 GB | $0.80 |
| `1x_a10g` | A10G | 24 GB | $1.10 |
| `1x_a100` | A100 | 80 GB | $2.50 |
| `1x_h100` | H100 | 80 GB | $3.95 |

Prices from [modal.com/pricing](https://modal.com/pricing). The hardware preset, optimization catalog, kernel patterns, and extractor patterns are all editable YAML, so you can extend them without code changes.

## Auxiliary commands

| Command | What it does |
|---|---|
| `profine auth login` / `status` / `set` / `logout` | Manage saved credentials in `~/.profine/auth.json` |
| `profine telemetry status` / `enable` / `disable` | Anonymous telemetry consent (or `PROFINE_NO_TELEMETRY=1`) |
| `profine env` | List every `PROFINE_*` env var with its current value |

## License

Apache 2.0. See [LICENSE](LICENSE).
