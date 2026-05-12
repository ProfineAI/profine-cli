# Optimization Suggestions

The highest-ROI path is to attack the dominant fp32 matmul bottleneck first: TF32 is the cheapest win, followed by mixed precision if numerical validation is acceptable. After that, use compiler-based fusion/optimization and SDPA to trim elementwise and attention overhead, then clean up the smaller optimizer bottleneck with fused AdamW.

**Estimated total speedup: 18% - 38%**

---

### #1 BF16 Mixed Precision [precision] (excl:2) !!

**Priority:** high | **Speedup:** 10%-25% | **Effort:** small | **Confidence:** medium

> This can materially accelerate the matmul-heavy workload on A100 by moving compute to tensor cores and reducing activation bandwidth, but it is more intrusive than TF32 and requires validating numerical stability. Because the model is already memory-light, the main value is compute throughput rather than memory savings. If training quality is sensitive, bf16 is generally safer than fp16 and preferable on Ampere.

**Addresses:** precision, compute_bound, memory_bandwidth_bound

**Risks:**
- Requires bf16-capable hardware
- Potential numerical issues in some loss functions

**Implementation:** `Wrap forward pass with torch.autocast('cuda', dtype=torch.bfloat16). For HuggingFace: set bf16=True in TrainingArguments.`

---

### #2 FP16 Mixed Precision with GradScaler [precision] (excl:2) !

**Priority:** medium | **Speedup:** 10%-22% | **Effort:** small | **Confidence:** medium

> Similar upside to bf16 for compute throughput, but slightly riskier because it needs GradScaler and is more prone to underflow. Since the codebase is currently pure fp32 and not memory constrained, the main benefit is faster GEMMs rather than capacity relief. It ranks below bf16 because bf16 is typically the safer first mixed-precision choice on A100.

**Addresses:** precision, compute_bound, memory_bandwidth_bound

**Risks:**
- Requires GradScaler for stable training
- More numerical issues than bf16

**Implementation:** `Wrap with torch.autocast('cuda', dtype=torch.float16) + GradScaler`

---

### #3 TF32 matmul precision [precision] !!!

**Priority:** critical | **Speedup:** 8%-18% | **Effort:** trivial | **Confidence:** high

> Best ROI: the run is dominated by fp32 GEMMs (46.2% matmul time, with aten::mm/addmm and Ampere SGEMM kernels on an A100). TF32 is a one-line change that directly targets the largest bottleneck with minimal implementation effort and no structural code changes. It also complements the current fp32-only setup without introducing autocast/scaler complexity.

**Addresses:** compute_bound

**Risks:**
- Small numerical drift versus strict fp32, though usually acceptable for training

**Implementation:** `torch.set_float32_matmul_precision('high')`

---

### #4 torch.compile mode='max-autotune' [compiler] (excl:3) !!

**Priority:** high | **Speedup:** 6%-15% | **Effort:** small | **Confidence:** medium

> Strong fit because the profile shows substantial elementwise work (15.6%) plus matmul-heavy transformer compute, and max-autotune can improve both fusion and kernel selection. It is more invasive than TF32 due to compile overhead and possible graph breaks, but likely higher upside than plain torch.compile because this model is steady-state and kernel-dominated. It may also reduce the benefit of separate fused-kernel point optimizations, so it should be tried before niche kernel swaps.

**Addresses:** compute_bound, memory_bandwidth_bound

**Risks:**
- Very slow first iteration
- Possible recompilation with dynamic shapes
- Some ops may not be supported

**Implementation:** `model = torch.compile(model, mode='max-autotune')`

---

### #5 torch.compile [compiler] (excl:3) !

**Priority:** medium | **Speedup:** 3%-10% | **Effort:** small | **Confidence:** medium

> Useful for fusing the observed elementwise work and reducing Python/kernel-launch overhead, but likely lower ROI than max-autotune. Since max-autotune already subsumes torch.compile's fusion/optimization benefits, plain torch.compile should be deprioritized relative to it unless compile time or compatibility is a concern. It remains a reasonable fallback if max-autotune is unstable.

**Addresses:** compute_bound, memory_bandwidth_bound, latency_bound

**Risks:**
- First iteration is slow
- Dynamic shapes may trigger recompilation
- Not all ops supported

**Implementation:** `model = torch.compile(model)  # or torch.compile(model, mode='reduce-overhead')`

---

### #6 Increase DataLoader num_workers [data_pipeline] 

**Priority:** low | **Speedup:** 0%-1% | **Effort:** small | **Confidence:** high

> The profile shows dataloader stalls under 1% of time, so there is very little headroom here. Even a perfect loader optimization cannot move the needle meaningfully because the run is overwhelmingly GPU-kernel bound. This is low priority unless future data preprocessing changes increase stall time.

**Addresses:** data_pipeline_bound, data_pipeline

**Risks:**
- Higher CPU/memory usage
- May cause issues with some datasets

**Implementation:** `DataLoader(..., num_workers=N, pin_memory=True, persistent_workers=True, prefetch_factor=2)`

---

### #7 Scaled Dot Product Attention (SDPA) [attention] (excl:1) !

**Priority:** medium | **Speedup:** 2%-6% | **Effort:** small | **Confidence:** high

> Directly addresses the observed manual causal attention path (6.5% of step time) and is a low-risk zero-config replacement. The absolute upside is limited because attention is not the dominant bottleneck, but it is a clean win and may also reduce some elementwise overhead around softmax/masking. This is a good follow-on after the larger matmul/precision wins.

**Addresses:** attention, compute_bound

**Risks:**
- Requires PyTorch >= 2.0

**Implementation:** `Replace manual Q@K^T/sqrt(d) matmul + softmax + V with F.scaled_dot_product_attention(Q, K, V)`

---

### #8 Fused AdamW [optimizer] (excl:4) 

**Priority:** low | **Speedup:** 1%-3% | **Effort:** trivial | **Confidence:** high

> The optimizer step is only 11.1% of step time, so the ceiling is modest even though fused AdamW is easy to adopt. It addresses the observed AdamW kernel overhead and memory bandwidth usage, but the absolute gain is capped by the optimizer's limited share of runtime. This is a sensible incremental improvement after the larger compute-side wins.

**Addresses:** optimizer, memory_bandwidth_bound

**Risks:**
- Requires PyTorch >= 2.0
- Requires CUDA

**Implementation:** `torch.optim.AdamW(params, fused=True)`

---

### #9 Foreach AdamW [optimizer] (excl:4) 

**Priority:** low | **Speedup:** 0%-2% | **Effort:** trivial | **Confidence:** medium

> This targets the same optimizer bottleneck as fused AdamW but is generally weaker and can increase peak memory. Because memory is abundant on this A100, the memory downside is not fatal, but the expected throughput gain is small and mostly redundant if fused AdamW is available. It is best treated as a fallback when fused AdamW is unavailable.

**Addresses:** optimizer, memory_bandwidth_bound

**Risks:**
- Higher peak memory due to batching all params

**Implementation:** `torch.optim.AdamW(params, foreach=True)`

---

## Also Applicable

- **DistributedDataParallel (DDP)** [distributed] — Replace DataParallel with DDP for efficient multi-GPU training. Overlaps gradien...
- **Gradient Checkpointing** [memory] — Trade compute for memory by recomputing activations during backward pass instead...
- **Fully Sharded Data Parallel (FSDP)** [distributed] — Shard model parameters, gradients, and optimizer states across GPUs. Enables tra...
- **8-bit AdamW (bitsandbytes)** [optimizer] — Replace AdamW with bitsandbytes.optim.AdamW8bit. Quantizes optimizer states to 8...
- **Paged AdamW (bitsandbytes)** [optimizer] — Use bitsandbytes paged AdamW (PagedAdamW32bit / PagedAdamW8bit) which offloads o...
- **Fused LayerNorm / RMSNorm** [kernel_fusion] — Replace separate elementwise ops in normalization with a single fused CUDA kerne...
- **Activation Offloading to CPU** [memory] — Offload activations to CPU RAM during forward pass, reload during backward. Extr...

## Notes
- Memory-capacity optimizations are not justified here because peak VRAM usage is only 1.43 GB on an 80 GB A100.
- Gradient checkpointing, FSDP, paged AdamW, and activation offloading are poor fits for this run because they either add overhead or solve a bottleneck that is not present.
