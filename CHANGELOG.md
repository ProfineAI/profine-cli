# Changelog

All notable changes to `profine` are documented here. This project follows [Semantic Versioning](https://semver.org/).

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
