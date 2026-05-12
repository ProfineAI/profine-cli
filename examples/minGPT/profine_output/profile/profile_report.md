# Profile Report: examples/minGPT/projects/chargpt/chargpt.py

- **Hardware**: 1x_a100
- **Status**: ok
- **Steps**: 60/60 (warmup: 50)
- **Runtime**: 61.7s

## Step Time

- **Median (steady-state)**: 22.80 ms
- **Steps measured**: 9
- **Warmup median**: 25.20 ms (1.1x steady-state)
- **Final loss**: 2.5020

## GPU Utilization

- **Mean**: 17.3%
- **Pattern**: periodic_gaps
- **Samples**: 11

## Memory

- **Peak**: 1.43 GB
- **Headroom**: 98.2%

## Top Kernels by CUDA Time

| Kernel | Category | Time (%) | Count |
|--------|----------|----------|-------|
| aten::mm | matmul | 12.8% | 1020 |
| aten::addmm | matmul | 7.1% | 480 |
| aten::bmm | attention | 6.5% | 720 |
| aten::mul | elementwise | 6.4% | 1700 |
| ampere_sgemm_32x128_tn | matmul | 5.1% | 380 |
| ampere_sgemm_32x32_sliced1x4_nt | matmul | 4.4% | 363 |
| Optimizer.step#AdamW.step | optimizer | 4.4% | 20 |
| ampere_sgemm_32x128_nn | matmul | 4.3% | 383 |
| void at::native::vectorized_elementwise_kernel<4, ... | other | 3.4% | 1106 |
| void at::native::vectorized_elementwise_kernel<4, ... | other | 3.1% | 602 |

## Kernel Category Breakdown

  Matmul          ██████████████████ 46.2%
  Elementwise     ██████ 15.6%
  Other           █████ 12.5%
  Optimizer       ████ 11.1%
  Attention       ██ 6.5%
  Normalization   █ 4.2%
  Memory          █ 3.9%

## Phase Breakdown

- **Forward**: 76.0%
- **Backward**: 14.0%
- **Optimizer**: 9.9%
- **DataLoader**: 0.2%
- **Other**: 0.0%

## Data Loading

- **Stall**: 0.9% of step time

## Warnings

- Very low GPU utilization (17%)