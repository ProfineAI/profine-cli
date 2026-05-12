# Benchmark Report

## ✅ 67.8% faster (3.11× speedup), correctness preserved.

**Optimization applied:** BF16 Mixed Precision + TF32 matmul precision + torch.compile mode='max-autotune' + Scaled Dot Product Attention (SDPA) + Fused AdamW
**Hardware:** 1x_a100 ($2.50/hr)
**Verdict:** PASS
**Notes:** PASS | 67.8% faster | 68.7% less memory | GPU util -11pp | correctness: PASS

---

## Metrics

| Metric | Baseline | Optimized | Δ | |
|---|---|---|---|---|
| Step time (ms) | 25.22 | 8.11 | -67.8% | ↑ improved |
| Throughput (steps/s) | 39.66 | 123.29 | +210.9% | ↑ improved |
| Peak memory (GB) | 1.43 | 0.45 | -68.7% | ↑ improved |
| GPU utilization (%) | 15.6 | 4.2 | -73.3% | ↓ regressed |

## Projected Savings

For every **100 hours** of training time saved at the optimized step time, you'd have spent **311 hours** on the baseline.

| Baseline run length | Time saved | Cost saved |
|---|---|---|
| 1 hr | 0.68 hr (41 min) | $1.70 |
| 10 hr | 6.78 hr (407 min) | $16.96 |
| 100 hr | 67.83 hr (4070 min) | $169.57 |
| 1000 hr | 678.30 hr (40698 min) | $1695.75 |

## Correctness

- **Loss curves match:** Yes ✓
- **Max loss diff:** 0.033886
- **Tolerance:** rtol=0.05, atol=0.01
- **Notes:** Loss curves match across 15 steps (max diff: 0.033886)

## Recommendation

**Ship it.** Speedup exceeds the 3% threshold and correctness passed.
