# Performance Analysis

**Hardware**: 1x A100 (80.0 GB) @ $2.5/hr
**Cost**: $0.04 this run | $0.02/1K steps
**Step time**: 0.02s (median, 9 steady-state steps)
**Memory**: 1.43 GB / 80.0 GB (1.8% used, 98.2% headroom)

## Time per Step by Category

| Category | ms/step | % |
|---|---|---|
| matmul | 11 | 48.2% |
| elementwise | 4 | 17.5% |
| optimizer | 3 | 13.2% |
| other | 3 | 13.2% |
| attention | 1 | 4.4% |
| normalization | 1 | 4.4% |
| memory | 1 | 4.4% |

## Top Kernels

| Kernel | Category | % | Time (ms) |
|---|---|---|---|
| aten::mm | matmul | 12.8% | 117.4 |
| aten::addmm | matmul | 7.1% | 65.2 |
| aten::bmm | attention | 6.5% | 59.7 |
| aten::mul | elementwise | 6.4% | 59.0 |
| ampere_sgemm_32x128_tn | matmul | 5.1% | 46.6 |
| ampere_sgemm_32x32_sliced1x4_nt | matmul | 4.4% | 40.3 |
| Optimizer.step#AdamW.step | optimizer | 4.4% | 40.2 |
| ampere_sgemm_32x128_nn | matmul | 4.3% | 39.3 |
| void at::native::vectorized_elementwise_ | other | 3.4% | 31.2 |
| void at::native::vectorized_elementwise_ | other | 3.1% | 28.1 |

## Profile Flags

- **Precision**: fp32
- **Attention**: manual
- **DataLoader stall**: 0.9%

## Bottleneck Diagnosis (LLM)

## Diagnosis

### Executive summary
This training run is **compute-bound** on a single A100. The largest cost is **matmul**, which consumes **46.2% of step time** (**11 ms/step** out of a **22.8 ms median step**). Secondary costs come from **elementwise ops** (**15.6%**, **4 ms/step**), **AdamW optimizer work** (**11.1%**, **3 ms/step**), and **attention** (**6.5%**, **1 ms/step**). Memory is not a limiter: peak VRAM is only **1.43 GB** on an **80 GB** GPU (**1.8% utilization**), and dataloader stalls are only **0.916%**.

### Bottlenecks ranked by estimated headroom
1. **Matmul / GEMM compute**
   - **Time share:** 46.2% of step time
   - **Absolute cost:** 11 ms/step
   - **Evidence:** `kernel_breakdown.matmul.pct = 46.2`, `kernel_breakdown.matmul.ms_per_step = 11`
   - **Top kernels:** `aten::mm` (12.8%), `aten::addmm` (7.1%), `ampere_sgemm_32x128_tn` (5.1%)
   - **Estimated headroom:** ~18% end-to-end
   - **Confidence:** observed

2. **Elementwise tensor ops**
   - **Time share:** 15.6% of step time
   - **Absolute cost:** 4 ms/step
   - **Evidence:** `kernel_breakdown.elementwise.pct = 15.6`, `kernel_breakdown.elementwise.ms_per_step = 4`
   - **Top kernels:** `aten::mul` (6.4%), `void at::native::vectorized_elementwise_` (3.4%, 3.1%)
   - **Estimated headroom:** ~7% end-to-end
   - **Confidence:** observed

3. **Optimizer step (AdamW)**
   - **Time share:** 11.1% of step time
   - **Absolute cost:** 3 ms/step
   - **Evidence:** `kernel_breakdown.optimizer.pct = 11.1`, `kernel_breakdown.optimizer.ms_per_step = 3`
   - **Top kernel:** `Optimizer.step#AdamW.step` (4.4%)
   - **Estimated headroom:** ~5% end-to-end
   - **Confidence:** observed

4. **Attention path**
   - **Time share:** 6.5% of step time
   - **Absolute cost:** 1 ms/step
   - **Evidence:** `kernel_breakdown.attention.pct = 6.5`, `kernel_breakdown.attention.ms_per_step = 1`
   - **Top kernel:** `aten::bmm` (6.5%)
   - **Estimated headroom:** ~3% end-to-end
   - **Confidence:** observed

5. **Data pipeline stalls**
   - **Time share:** 0.916% of step time
   - **Evidence:** `dataloader_stall_pct = 0.9162680750916005`
   - **Estimated headroom:** ~0.9% end-to-end
   - **Confidence:** observed

### Bound analysis
- **Compute-bound:** Yes
- **Memory bandwidth bound:** No strong evidence; the profile is dominated by GEMM compute rather than a bandwidth-heavy kernel mix.
- **Memory capacity bound:** No. Peak memory is **1.43 GB** out of **80 GB** (**98.2% headroom**).
- **Data pipeline bound:** No. Stalls are under **1%**.
- **Communication bound:** No. The run is single-GPU (`world_size = 1`).

### Time distribution narrative
The step is dominated by GPU kernel execution rather than waiting or synchronization. Matmul accounts for **46.2%** of total step time, and the top kernels are standard GEMM ops (`aten::mm`, `aten::addmm`, and Ampere SGEMM variants), indicating that dense linear algebra is the main consumer of runtime. Beyond that, pointwise elementwise work takes **15.6%**, optimizer stepping takes **11.1%**, and attention is comparatively small at **6.5%**. The input pipeline contributes only **0.916%** stall time, and memory usage is tiny relative to available VRAM, so neither data loading nor capacity pressure materially affects throughput.

### Additional notes
- Precision is **fp32**, and the script shows **manual attention** rather than a specialized attention kernel path.
- The model is a GPT-style character-level language model with **context length 128**.
- No distributed strategy is active, so there is no cross-device communication overhead in this profile.