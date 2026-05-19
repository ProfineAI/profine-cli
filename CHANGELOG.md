# Changelog

All notable changes to `profine` are documented here. This project follows [Semantic Versioning](https://semver.org/).

## [0.5.1] — 2026-05-19

### Fixed

- **Telemetry: `_loss_ok_from_bench` was reading a non-existent key.** The function read `correctness.verdict` from `benchmark_comparison.json`, but the correctness sub-dict has a `passed` (bool) key, not a `verdict`. (`verdict` is a top-level `BenchmarkComparison` field.) The bug emitted `loss_ok=None` for every run sent to the telemetry backend, which made the server-side `optimization_priors` materialized view's `success_rate` column NULL for every (fingerprint, optimization) pair across the entire user base. The test fixture was buggy in the same way as the production code (it also used the `verdict` key), explaining the silent regression. Adds a focused regression test (`test_loss_ok_reads_correctness_passed_not_verdict`) that pins both the real and old-buggy shapes.

## [0.5.0] — 2026-05-18

### Breaking

- **`--hardware` is now required** on `profile`, `benchmark`, and `run-all`. The previous `auto` default silently chose a "smallest preset that fits" using a heuristic that mis-sized GPUs for unknown architectures; making it explicit prevents that footgun. Pick one of: `1x_t4`, `1x_l4`, `1x_a10g`, `1x_a100`, `1x_h100`. The `auto_select_hardware()` helper and the param-bucket preset table have been removed.

### Added

- **`profine telemetry doctor`.** Synchronous probe of the telemetry endpoint that reports consent state, endpoint URL, HTTP status code, and per-attempt latency. Use this to verify the round-trip works (or to warm a sleeping Render dyno before a real run).
- **Update-check nudge on CLI startup.** Profine now checks PyPI for the latest release once every 24 hours (cached in `~/.profine/`) and prints a one-line nudge if your installed version is behind. Silenced via `PROFINE_NO_UPDATE_CHECK=1`.
- **Low-sample warning.** Benchmark reports surface a warning when fewer than 10 step samples survive warmup stripping — so users notice when the median is built on thin data.
- **`PROFINE_TELEMETRY_RETRY_BACKOFF` env var.** Test-and-CI knob for the telemetry retry backoff. Defaults to 2.0s in production.

### Changed

- **Telemetry HTTP transport: timeout 5s → 15s, one retry with 2s backoff.** The anon endpoint is hosted on Render's free/starter tier, where the first request after idle takes ~9s to wake the dyno. Under the old 5s timeout that first POST was always silently dropped. Final-attempt failures now log at WARNING (was DEBUG) so silent data loss is no longer invisible.
- **Verdict string for fast-but-wrong runs** now reads `FAIL (correctness; speedup measured but loss diverged)` instead of leading with `PASS`. A run that ships incorrect numerics is not a pass, regardless of its step time.
- **README results section** replaced with a median-of-3 multi-GPU table (A10G + A100). Honest framing of variance + range rather than a single fast-run headline.

### Fixed

- **`_projected_savings` divide-by-zero** when speedup approached 100% (zero-sample candidate). Clamped `fraction_saved` to 0.99.
- **`_maybe_adapt` step-time estimate poisoned by torch.compile cold-start.** The adaptive step controller previously used `elapsed / steps_completed`, which is dominated by a 2.8s first-step compile when the steady state is ~17ms. Now uses median of recorded step times when available.
- **`_strip_warmup` could strip more samples than existed**, producing a zero-sample comparison with a bogus "100% faster / ∞× speedup" result. Capped to keep at least 3 samples on both `benchmarker.benchmarker` and `profiler.orchestrator`.
- **`--edit-dir` outside `--output`** now correctly resolves the suggest report via `edit_dir.parent / "suggest"`. Without this, the BF16-aware tolerance widening never fired on standalone `benchmark` invocations, and every BF16-stack benchmark spuriously failed correctness.
- **`_resolve_hardware`** in `telemetry/emit.py` now prefers the explicit `hardware_name` argument over `profile_record.hardware_name`. Batch / replay callers re-emitting from on-disk artifacts for a *different* GPU than the one that produced the profile record were having their rows mis-tagged.

### Internal

- 9 new regression tests pinning each surface bug above; 584 tests total.
- Six empty package directories deleted (`heuristics/`, `modifiers/`, `output/`, `preflight/`, `search/`, `resources/`) — vestigial scaffolding from a past refactor.
- LLM backends (`profine/llm/backend.py`) gained exponential-backoff retry for transient API errors (timeouts, 5xx, rate limits), bounded at 3 attempts and env-tunable.
- Modal executor (`profine/modal/executor.py`) filters benign Inductor autotune log spam (`No valid triton configs`, `OutOfMemoryError: out of resource: triton_mm`) so successful autotune sweeps don't read as crashes; also wires `PROFINE_WALL_CLOCK_LIMIT` so the script's `StepController` stays below Modal's container timeout.
- Stacked edits in `profine/editor/editor.py` are wrapped in try/except so one bad LLM candidate surfaces as a non-applied `EditResult` instead of blowing away previously-successful edits.
- Reader feeds sibling modules to the analyzer LLM, so defaults defined in imported files (e.g. `mingpt/model.py`) no longer come back as "guessed" zeros.
- File-not-found errors now hint that a sibling `prepare.py` needs to run when the missing path looks like a tokenized dataset (nanoGPT/minGPT layout).

## [0.4.0] — 2026-05-16

### Added
- **`profine auth` — saved API keys.** Paste your credentials once with `profine auth login` and every subsequent command picks them up — no more re-exporting env vars in every shell. Keys live in `~/.profine/auth.json` (chmod 0600, honors `PROFINE_HOME`). Manages `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `PROFINE_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `HF_TOKEN`. Subcommands: `login` (interactive `getpass` paste flow), `status` (redacted listing), `set KEY [VALUE]`, `logout [KEY]`. The environment always wins on conflict, so CI and one-off `KEY=… profine …` invocations keep working unchanged.
- **Missing-key errors now suggest `profine auth login`** as the first option, alongside `export` and `--api-key`.
- **Anonymous, opt-in telemetry.** The first interactive run of any `profine` command prompts you to share anonymous run statistics. Saying yes contributes a small bucketed signature to the shared optimization-priors flywheel: arch class (e.g. `transformer-decoder`), parameter bucket (e.g. `1B-7B`), hardware class (e.g. `1x_a100`), precision, optimizer family, plus per-run profile stats (step time p50/p95, GPU util, memory peak, primary bottleneck) and per-optimization outcomes (speedup factor, success, crash class). **What is never sent: source code, dataset paths, model checkpoints, file paths, raw exception text, identifying hyperparameter values.** A single explicit allowlist (`profine/telemetry/fields.py`) gates everything that leaves the process. Saying no is a silent no-op; the CLI works identically.
- **`profine telemetry` subcommand.** `status` shows your current consent, where the file lives, and whether environment overrides (`PROFINE_NO_TELEMETRY`, `PROFINE_API_KEY`) are active. `enable` and `disable` toggle the OSS consent file (`~/.profine/telemetry_consent.json`).
- **`--no-telemetry` shared flag.** Disable telemetry for a single invocation without changing the stored consent. Works before or after the subcommand.
- **`PROFINE_NO_TELEMETRY=1` env var.** Same effect as `--no-telemetry`, useful for CI and scripts.

### Changed
- **License: MIT → Apache 2.0** (carried forward from 0.3.6 release notes; no change since 0.3.5 from the user's perspective beyond the additional patent grant).

### Notes
- This release does not change the behaviour of any existing command. Telemetry is purely additive and off by default until you grant consent.
- Paying customers configure telemetry through `PROFINE_API_KEY` and the server-side opt-out toggle; OSS consent is bypassed when that key is set.

## [0.3.5] — 2026-05-12

### Fixed
- **Verdict no longer flags REGRESSION on a clear speedup with a util drop.** The old rule treated any GPU-utilization drop ≥ 15pp as REGRESSION regardless of throughput. That was wrong: when an optimization makes each step finish faster, the GPU naturally sits idle more between steps — util drops, but throughput improves. The new rule only flags util-drop as REGRESSION when speedup is *also* weak (< 3%). Concrete example: 54.9% faster + correctness PASS + GPU util −20pp now correctly verdicts as **PASS** instead of REGRESSION.

## [0.3.4] — 2026-05-12

### Fixed
- **`.env` loading from installed binary.** `load_dotenv()` defaults to searching upward from the entry-script directory (e.g. `~/anaconda3/bin/`), which never has a `.env`. Now passes `find_dotenv(usecwd=True)` explicitly, so `OPENAI_API_KEY` (and friends) in your project's `.env` are picked up when you run `profine` from your project root.
- **Spurious "correctness: FAIL" verdicts.** `_strip_warmup` used to chop `loss_values` per-payload using auto-detected stabilization points, which yielded different warmup counts for baseline vs. optimized (e.g. 10 vs. 15 with `torch.compile`). The correctness check then compared baseline-step-10 against optimized-step-15 — five training steps apart — and flagged the natural divergence as a failure. Now `_strip_warmup` only touches `step_times_ms`; losses stay aligned by original training-step index.
- **Cleaner `SUMMARY.md` architecture section.** Evidence-wrapped and compound architecture fields now render as `key=value, key=value` instead of dumping raw nested JSON.

## [0.3.3] — 2026-05-12

### Added
- **Deterministic optimization rankings.** New `--seed` flag (best-effort) plumbed through every LLM-using stage. Default LLM `temperature` is now `0.0` (was `0.2`). With a fixed seed and temperature=0, OpenAI and OpenAI-compatible local providers produce the same ranking and apply/skip decisions across runs; Anthropic ignores `seed` but honors temperature=0.

### Changed
- All backends now accept `temperature` and `seed` constructor arguments; defaults are deterministic.

## [0.3.2] — 2026-05-12

### Added
- **Colored CLI output** (rich-backed): stage banners, success/error/warn markers, highlighted file paths. Auto-disables when stdout isn't a TTY (so `tee run.log` stays clean) and honors the `NO_COLOR` environment variable.
- **Karpathy's tinyshakespeare** corpus (`examples/minGPT/projects/chargpt/input.txt`, 1.1 MB) so the `chargpt` example runs out-of-the-box without external downloads.

### Fixed
- **Benchmark hard-fails when baseline crashes.** Previously, an exhausted-retry baseline produced an empty payload that was silently compared against the optimized run, yielding a meaningless `NO-OP (0% change)` verdict. Now the benchmarker raises with an actionable message and `run-all` aborts with `exit 1`.

## [0.3.1] — 2026-05-12

### Added
- `run-all` now writes a consolidated **`SUMMARY.md`** at the end with the headline verdict, architecture, bottleneck, optimizations applied, benchmark metrics, and an artifacts index. This is the one file to read after the pipeline finishes.
- Benchmark report (`benchmark_report.md`) is now decision-useful:
  - TL;DR headline (✅/⚠️/❌/➖) with speedup multiplier
  - Projected time-and-cost-saved table (1 hr / 10 hr / 100 hr / 1000 hr of training)
  - Explicit "ship it / hold / revert" recommendation
- `generate_report()` now accepts `hardware` and `cost_per_hour` for the savings projection.

### Changed
- Benchmark report headline now states a clear human verdict instead of just a percentage.

## [0.3.0] — 2026-05-12

### Added
- **Local LLM support** via `--provider local`. profine now talks to any OpenAI-compatible server: Ollama, vLLM, LM Studio, llama.cpp server, LiteLLM. Configure with `--model` and (optionally) `--base-url` or the `PROFINE_LOCAL_BASE_URL` environment variable. Default endpoint: Ollama at `http://localhost:11434/v1`.
- New `--base-url` CLI flag (shared across all commands).

### Changed
- `--provider` now accepts `local` in addition to `openai` and `anthropic`; values are validated by argparse.
- API-key gate skipped for `--provider local` (which doesn't need one); `--model` is required instead.

## [0.2.0] — 2026-05-11

### Added
- `profine run-all <script>` — end-to-end pipeline (read → profile → interpret → suggest → edit → benchmark) in one command.
- HuggingFace config reader (`profine/reader/hf_config.py`).
- Catalog `exclusive_group` support — mutually-exclusive optimizations (attention impl, precision, compiler, optimizer variant, distributed) can no longer be stacked.
- Karpathy's [minGPT](https://github.com/karpathy/minGPT) added to `examples/`.

### Changed
- Comparator verdict now surfaces correctness fails alongside speed verdicts (e.g. `PASS (correctness: FAIL)`) instead of collapsing to `REGRESSION`.
- Updated Modal GPU pricing in `profine/config/hardware.yaml` (L4 $0.80, A100-80GB $2.50, H100 $3.95).
- Misc improvements to executor, image builder, reader, suggester.

### Fixed
- `pyproject.toml` Repository URL now points at the public `profine-cli` repo.

## [0.1.1]

Initial public release.
