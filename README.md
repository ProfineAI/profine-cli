# profine

Profile and optimize PyTorch training scripts on real Modal GPUs. Six-step pipeline: read, profile, interpret, suggest, edit, benchmark.

## Install

```bash
pip install -e .
```

Requires a Modal account and an LLM API key (OpenAI or Anthropic).

## Setup

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

### 1. Read — analyze the training script

```bash
profine read nanoGPT/train.py
```

Extracts model architecture, optimizer, dataloader, precision, and distributed strategy via AST + LLM analysis. Output: `profine_output/read/architecture_record.json`

### 2. Profile — run on a real GPU

```bash
profine profile nanoGPT/train.py --hardware 1x_a100 --steps 20 --warmup 10
```

Instruments the script, executes on Modal with torch.profiler, and collects step times, kernel breakdown, GPU utilization, and memory usage. Output: `profine_output/profile/profile_record.json`

### 3. Interpret — diagnose bottlenecks

```bash
profine interpret --profile-dir profine_output/profile
```

Deterministic analysis (cost, memory utilization, per-category kernel times) + LLM diagnosis of bottlenecks. Output: `profine_output/interpret/bottleneck_report.json`

### 4. Suggest — rank optimizations

```bash
profine suggest --interpret-dir profine_output/interpret
```

Filters applicable optimizations from the catalog, LLM ranks by ROI. Output: `profine_output/suggest/suggestion_report.json`

### 5. Edit — apply an optimization

```bash
profine edit nanoGPT/train.py --suggestion-dir profine_output/suggest
profine edit nanoGPT/train.py --suggestion-dir profine_output/suggest --optimization 2
profine edit nanoGPT/train.py --suggestion-dir profine_output/suggest --optimization torch_compile
profine edit nanoGPT/train.py --suggestion-dir profine_output/suggest --top 3
```

The editor is multi-file aware: it discovers local modules the entry script imports and edits whichever file actually owns the code being optimized (e.g. a `Trainer` class or model module in a separate file). Patched library files land under `profine_output/edit/files/<rel-path>` — your source tree is never modified.

`--top N` applies the N top-ranked candidates **sequentially**, each layered on the previous edit. Per-iteration artifacts go in `profine_output/edit/01_<entry_id>/`, `02_<entry_id>/`, etc.; the cumulative result lands at the standard `profine_output/edit/edited_train.py` + `files/` paths so `profine benchmark` picks it up unchanged. Optimizations the LLM declines (`applied: false`) are recorded in the manifest's `skipped` list and the loop continues.

Output: `profine_output/edit/edited_train.py`, `profine_output/edit/files/<patched library files>`, `profine_output/edit/change_manifest.json`

### 6. Benchmark — measure the improvement

```bash
profine benchmark nanoGPT/train.py --optimized profine_output/edit/edited_train.py --hardware 1x_a100 --steps 20 --warmup 10
```

Runs original and optimized back-to-back on the same hardware. Auto-loads patched library files from `profine_output/edit/files/` as workspace overlays so multi-file edits actually take effect on the optimized run. When the entry script is unchanged (multi-file edit), the same instrumented script is reused for both runs to guarantee data parity. Loss tolerance is widened automatically for optimization classes that legitimately perturb numerics (BF16 / mixed precision: rtol 5%, quantization: rtol 10%); for stacked edits the loosest applicable tolerance wins. Output: `profine_output/benchmark/`

## Hardware Presets

Defined in `profine/config/hardware.yaml`. Add new GPUs by editing the YAML.

| Preset | GPU | VRAM | Cost/hr |
|---|---|---|---|
| `1x_t4` | T4 | 16 GB | $0.59 |
| `1x_l4` | L4 | 24 GB | $0.73 |
| `1x_a10g` | A10G | 24 GB | $1.10 |
| `1x_a100` | A100 | 80 GB | $3.73 |
| `1x_h100` | H100 | 80 GB | $6.98 |

## Configuration

All data tables (hardware presets, optimization catalog, kernel patterns, extractor patterns) live in `profine/config/*.yaml` and can be extended without code changes.

## Command Reference

### Global flags (all commands)

| Flag | Default | Description |
|---|---|---|
| `--provider` | `openai` | LLM provider (`openai` or `anthropic`) |
| `--api-key` | env var | API key override |
| `--model` | provider default | Model name override |
| `-o, --output` | `profine_output` | Output directory |
| `--prefs` | none | Path to user preferences markdown |

### `profine read <script>`

No additional flags.

### `profine profile <script>`

| Flag | Default | Description |
|---|---|---|
| `--hardware` | `1x_a100` | Hardware preset name |
| `--steps` | `60` | Total optimizer steps |
| `--warmup` | `30` | Warmup steps (discarded from analysis) |
| `--timeout` | `900` | Modal container timeout in seconds |
| `--warmstart` | off | Reuse deployed Modal app between runs |

### `profine interpret`

| Flag | Default | Description |
|---|---|---|
| `--profile-dir` | required | Directory containing `profile_record.json` |

### `profine suggest`

| Flag | Default | Description |
|---|---|---|
| `--interpret-dir` | required | Directory containing `bottleneck_report.json` |
| `--arch-dir` | auto-detect | Directory containing `architecture_record.json` |
| `--profile-dir` | auto-detect | Directory containing `profile_record.json` |

### `profine edit <script>`

| Flag | Default | Description |
|---|---|---|
| `--suggestion-dir` | required | Directory containing `suggestion_report.json` |
| `--optimization` | `1` (top-ranked) | Rank number (`1`, `2`, ...) or entry ID (`torch_compile`). Ignored when `--top` is set. |
| `--top` | unset | Apply the top N ranked optimizations sequentially, each stacked on the previous edit. |

### `profine benchmark <script>`

| Flag | Default | Description |
|---|---|---|
| `--optimized` | required | Path to the optimized script |
| `--hardware` | `1x_a100` | Hardware preset name |
| `--steps` | `60` | Total optimizer steps |
| `--warmup` | `30` | Warmup steps |
| `--rtol` | `0.01` | Relative tolerance for loss correctness check (auto-widened for `precision`/`quantization` classes) |
| `--atol` | `0.0001` | Absolute tolerance for loss correctness check (auto-widened for `precision`/`quantization` classes) |
| `--edit-dir` | `<output>/edit` | Directory whose `files/` subtree is overlaid onto the optimized run |
| `--timeout` | `900` | Modal container timeout in seconds |
| `--warmstart` | off | Reuse deployed Modal app between runs |

## License

MIT. See [LICENSE](LICENSE).
