# profine

**Website: [profine.ai](https://profine.ai/)**

Profile your PyTorch code on real GPUs. Get a transparent rewrite. Ship measured speedups before the multi-hour run.

## Demo

[![Watch the demo](https://img.youtube.com/vi/CY9aW1Dcrn0/hqdefault.jpg)](https://youtu.be/CY9aW1Dcrn0)

## Results

On [Karpathy's minGPT](https://github.com/karpathy/minGPT), single-A100:

| Metric | Baseline | profine | Δ |
|---|---|---|---|
| Step time | 1.00× | **3.1× faster** | −67.7% ms/step |
| Peak memory | 1.00× | **−66.4%** | substantial headroom for larger batch |

Numbers reproduced with `profine run-all examples/minGPT/projects/chargpt/chargpt.py --hardware 1x_a100`.

## Install

```bash
pip install profine
```

Requires a Modal account and an LLM API key (OpenAI or Anthropic).

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY
export HF_TOKEN=...              # optional, for gated models
```

## Pipeline

```
read → profile → interpret → suggest → edit → benchmark
```

Each step reads the previous step's output from `profine_output/`.

Global flags (all commands): `--provider {openai,anthropic}` (default `openai`), `--api-key`, `--model`, `-o/--output` (default `profine_output`), `--prefs`.

### Auto (`run-all`)

Run the entire pipeline end-to-end on one script.

```bash
profine run-all examples/minGPT/projects/chargpt/chargpt.py --hardware 1x_a100
```

| Flag | Default | Description |
|---|---|---|
| `--hardware` | `1x_a100` | Hardware preset |
| `--steps` | `60` | Total optimizer steps |
| `--warmup` | `30` | Warmup steps |
| `--timeout` | `900` | Modal container timeout (s) |
| `--warmstart` | off | Reuse deployed Modal app between runs |
| `--top` | all | Apply top N ranked optimizations |
| `--rtol` / `--atol` | `0.01` / `0.0001` | Loss tolerances (auto-widened for precision/quantization) |

Aborts on any failed step. Per-step artifacts land in their usual subdirectories under `profine_output/`.

### 1. Read

Extract model architecture, optimizer, dataloader, precision, and distributed strategy via AST + LLM.

```bash
profine read nanoGPT/train.py
```

No additional flags. Output: `profine_output/read/architecture_record.json`

### 2. Profile

Instrument the script and run on Modal with torch.profiler; collects step times, kernel breakdown, GPU utilization, and memory.

```bash
profine profile nanoGPT/train.py --hardware 1x_a100 --steps 20 --warmup 10
```

| Flag | Default | Description |
|---|---|---|
| `--hardware` | `1x_a100` | Hardware preset name |
| `--steps` | `60` | Total optimizer steps |
| `--warmup` | `30` | Warmup steps (discarded) |
| `--timeout` | `900` | Modal container timeout (s) |
| `--warmstart` | off | Reuse deployed Modal app between runs |

Output: `profine_output/profile/profile_record.json`

### 3. Interpret

Deterministic analysis (cost, memory utilization, per-category kernel times) + LLM bottleneck diagnosis.

```bash
profine interpret --profile-dir profine_output/profile
```

| Flag | Default | Description |
|---|---|---|
| `--profile-dir` | required | Directory containing `profile_record.json` |

Output: `profine_output/interpret/bottleneck_report.json`

### 4. Suggest

Filter applicable optimizations from the catalog; LLM ranks by ROI.

```bash
profine suggest --interpret-dir profine_output/interpret
```

| Flag | Default | Description |
|---|---|---|
| `--interpret-dir` | required | Directory containing `bottleneck_report.json` |
| `--arch-dir` | auto | Directory containing `architecture_record.json` |
| `--profile-dir` | auto | Directory containing `profile_record.json` |

Output: `profine_output/suggest/suggestion_report.json`

### 5. Edit

Apply an optimization. Multi-file aware: discovers local modules the entry script imports and edits whichever file owns the code being optimized. Patched library files land under `profine_output/edit/files/<rel-path>` — your source tree is never modified.

```bash
profine edit nanoGPT/train.py --suggestion-dir profine_output/suggest
profine edit nanoGPT/train.py --suggestion-dir profine_output/suggest --optimization torch_compile
profine edit nanoGPT/train.py --suggestion-dir profine_output/suggest --top 3
```

| Flag | Default | Description |
|---|---|---|
| `--suggestion-dir` | required | Directory containing `suggestion_report.json` |
| `--optimization` | `1` | Rank (`1`, `2`, ...) or entry ID (`torch_compile`). Ignored when `--top` is set. |
| `--top` | unset | Apply top N ranked optimizations sequentially, stacked. |

With `--top N`, per-iteration artifacts go in `profine_output/edit/01_<entry_id>/`, `02_<entry_id>/`, etc.; cumulative result at `profine_output/edit/edited_train.py`. Optimizations the LLM declines are recorded in the manifest's `skipped` list and the loop continues.

Output: `profine_output/edit/edited_train.py`, `profine_output/edit/files/`, `profine_output/edit/change_manifest.json`

### 6. Benchmark

Run original and optimized back-to-back on the same hardware. Patched library files in `profine_output/edit/files/` are overlaid on the optimized run. Loss tolerance auto-widens for numerics-perturbing classes (BF16/mixed precision: rtol 5%, quantization: rtol 10%).

```bash
profine benchmark nanoGPT/train.py --optimized profine_output/edit/edited_train.py --hardware 1x_a100 --steps 20 --warmup 10
```

| Flag | Default | Description |
|---|---|---|
| `--optimized` | required | Path to the optimized script |
| `--hardware` | `1x_a100` | Hardware preset name |
| `--steps` | `60` | Total optimizer steps |
| `--warmup` | `30` | Warmup steps |
| `--rtol` | `0.01` | Relative tolerance for loss check (auto-widened) |
| `--atol` | `0.0001` | Absolute tolerance for loss check (auto-widened) |
| `--edit-dir` | `<output>/edit` | Directory whose `files/` subtree is overlaid |
| `--timeout` | `900` | Modal container timeout (s) |
| `--warmstart` | off | Reuse deployed Modal app between runs |

Output: `profine_output/benchmark/`

## Hardware Presets

Defined in `profine/config/hardware.yaml`.

| Preset | GPU | VRAM | Cost/hr |
|---|---|---|---|
| `1x_t4` | T4 | 16 GB | $0.59 |
| `1x_l4` | L4 | 24 GB | $0.73 |
| `1x_a10g` | A10G | 24 GB | $1.10 |
| `1x_a100` | A100 | 80 GB | $3.73 |
| `1x_h100` | H100 | 80 GB | $6.98 |

All data tables (hardware, optimization catalog, kernel patterns, extractor patterns) live in `profine/config/*.yaml` and can be extended without code changes.

## License

MIT. See [LICENSE](LICENSE).
