# profine run-all — examples/minGPT/projects/chargpt/chargpt.py

**Hardware:** `1x_a100`

## ✅ 67.8% faster (3.11× speedup), correctness preserved.

## Architecture (what we found in your code)

- **Model:** GPT
- **Framework:** PyTorch
- **Precision:** training_dtype=float32
- **Optimizer:** name=AdamW, learning_rate=0.0005
- **Dataloader:** dataset_class=CharDataset

## Bottleneck (what's slowing it down)

- **matmul**
- **elementwise**
- **optimizer**

## Optimizations

**Ranked by LLM ROI:**

1. `bf16_mixed_precision` — This can materially accelerate the matmul-heavy workload on A100 by moving compute to tensor cores and reducing activation bandwidth, but it is more intrusive t
2. `fp16_mixed_precision` — Similar upside to bf16 for compute throughput, but slightly riskier because it needs GradScaler and is more prone to underflow
3. `tf32_matmul` — Best ROI: the run is dominated by fp32 GEMMs (46
4. `torch_compile_max_autotune` — Strong fit because the profile shows substantial elementwise work (15
5. `torch_compile` — Useful for fusing the observed elementwise work and reducing Python/kernel-launch overhead, but likely lower ROI than max-autotune

**Skipped (4):** `fp16_mixed_precision` (exclusive group 2 — conflicts with already-applied optimization), `torch_compile` (exclusive group 3 — conflicts with already-applied optimization), `dataloader_workers` (The input pipeline is already lightweight and the optimization record states that increasing DataLoader workers would add CPU/memory overhead with essentially no expected return for this CharDataset, so it should be skipped.), `foreach_adamw` (exclusive group 4 — conflicts with already-applied optimization)

## Benchmark (measured on-GPU)

| Metric | Δ |
|---|---|
| Step time | **+67.8%** (3.11× faster) |
| Peak memory | -68.7% |
| GPU utilization | -11.4% |
| Verdict | **PASS** |
| Correctness | ✓ pass |

## Artifacts

- [`read/architecture_record.json`](read/architecture_record.json) — Parsed architecture (JSON)
- [`read/architecture_brief.md`](read/architecture_brief.md) — Architecture brief (MD)
- [`profile/profile_record.json`](profile/profile_record.json) — Profile data (JSON)
- [`profile/profile_report.md`](profile/profile_report.md) — Profile report (MD)
- [`interpret/bottleneck_report.json`](interpret/bottleneck_report.json) — Bottleneck diagnosis (JSON)
- [`interpret/bottleneck_brief.md`](interpret/bottleneck_brief.md) — Bottleneck brief (MD)
- [`suggest/suggestion_report.json`](suggest/suggestion_report.json) — Ranked optimizations (JSON)
- [`suggest/suggestion_brief.md`](suggest/suggestion_brief.md) — Suggestion brief (MD)
- [`edit/edited_train.py`](edit/edited_train.py) — Optimized training script
- [`edit/change_manifest.json`](edit/change_manifest.json) — What was changed and why
- [`benchmark/benchmark_comparison.json`](benchmark/benchmark_comparison.json) — Benchmark comparison (JSON)
- [`benchmark/benchmark_report.md`](benchmark/benchmark_report.md) — Benchmark report (MD)
