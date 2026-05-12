# Changelog

All notable changes to `profine` are documented here. This project follows [Semantic Versioning](https://semver.org/).

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
