# Changelog

All notable changes to `profine` are documented here. This project follows [Semantic Versioning](https://semver.org/).

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
