# 2.1.4 Nsight Systems Profiling (nsys_profile)

## Setup

- Hardware: 1× NVIDIA GeForce RTX 3090 (24 GiB), driver 580.95.05, PyTorch 2.11.0+cu130, Nsight Systems 2026.1.3.
- Script: `cs336_systems/benchmarking_script.py` (extended with `--context_length`, `--batch_size`, `--annotate`
  and NVTX ranges `measure` / `forward` / `backward` / `optimizer_step`; `--annotate` swaps in an
  NVTX-annotated `scaled_dot_product_attention` with sub-ranges `computing attention scores`,
  `computing softmax`, `final matmul`). In `forward-only` mode the pass runs under
  `torch.inference_mode()` (true inference benchmarking; also avoids retaining the autograd graph).
- Batch size 4, FP32, 5 warmup + 10 measurement steps per profile. Profiles were collected with
  `nsys profile --trace=cuda,nvtx -o profiles/<name> python cs336_systems/benchmarking_script.py ...`
  and analyzed with `nsys stats` (`nvtx_gpu_proj_sum`, `nvtx_kern_sum`, `cuda_gpu_kern_sum`),
  filtering kernels by NVTX range so warmup steps are excluded from the per-phase numbers.

Profiled combinations (two model sizes, three power-of-two context lengths > 128):

| profile | model | ctx | mode | timeit fwd (ms) |
|---|---|---|---|---|
| small_ctx256_fwd  | small (d=768, L=12) | 256  | forward-only | 30.5 |
| small_ctx1024_fwd | small | 1024 | forward-only | 114.7 |
| small_ctx4096_fwd | small | 4096 | forward-only | 977.4 |
| large_ctx512_fwd  | large (d=1280, L=36) | 512  | forward-only | 299.0 |
| large_ctx2048_fwd | large | 2048 | forward-only | 1856.2 |
| small_ctx1024_fwd_bwd | small | 1024 | forward+backward | fwd 113.9 / bwd 239.5 |
| small_ctx1024_train   | small | 1024 | full training step | fwd 112.7 / bwd 240.4 / opt 31.5 |
| small_ctx1024_fwd_ann | small | 1024 | forward-only, annotated attention | 114.6 |

Note on "longest context that fits in memory": `large` at ctx 4096 OOMs on 24 GiB even in inference
mode (the FP32 attention-score temporaries alone are ~5 GiB each at that length), so ctx 2048 is the
longest length profiled for `large`; `small` fits ctx 4096.

## (a) Total forward-pass time; does it match the timeit measurements?

The GPU time attributed to the `forward` NVTX range (sum of kernel time projected onto the range,
per step) matches the `timeit` wall-clock measurements to within ~1% on every profile:

| profile | nsys GPU fwd/step | timeit fwd/step |
|---|---|---|
| small/256  | 30.06 ms  | 30.52 ms |
| small/1024 | 114.53 ms | 114.75 ms |
| small/4096 | 977.10 ms | 977.44 ms |
| large/512  | 298.84 ms | 299.01 ms |
| large/2048 | 1855.94 ms | 1856.24 ms |

They agree almost exactly, which is expected: the script synchronizes the CPU with the GPU around
each phase, so wall-clock time equals GPU kernel time.

## (b) Top CUDA kernel in the forward pass; same kernel for fwd+bwd?

During the forward pass the kernel with the most cumulative GPU time is `ampere_sgemm_128x64_tn`
(an FP32 GEMM); for small/1024 it is invoked 73 times per forward pass (once per linear/matmul
instance across the 12 layers, the rest of the 109 matmul calls landing on other `sgemm` tilings)
and accounts for ~38 ms of the ~114 ms forward. When profiling forward+backward together, the top
kernel overall is still an SGEMM but a different tiling — `ampere_sgemm_128x64_nn` (669 ms vs. the
tn variant's 562 ms over the run) — because the backward pass adds NN-layout gradient GEMMs; i.e.,
matmuls dominate in both cases, with backward adding ~2× more GEMM time than forward alone.

## (c) Non-matmul kernels with non-trivial runtime in the forward pass

Several elementwise/reduction kernels each take ~4–6.5 ms per forward (small/1024, ~45% of forward
GPU time collectively): `exp_kernel_cuda` (softmax), `reduce_kernel` (softmax max/sum reductions),
`MulFunctor` elementwise kernels (RoPE rotation, 1/√d scaling, causal masking via `where`),
`CUDAFunctor_add` (residual adds / bias), and `sigmoid`+`mul` (SwiGLU gating). Their share grows
quickly with context length — at small/4096 the GEMM share of forward drops from 68.7% (ctx 256) to
37.9%, because the O(S²) attention-score temporaries make these memory-bound elementwise passes as
expensive as the GEMMs.

## (d) Full training step vs. inference: how does the matmul fraction change?

For small/1024 a full step takes ~385 ms GPU time (forward 112.5, backward ~240, AdamW ~31).
Matmuls drop from 54.7% of kernel time in the forward-only profile to 48.5% over the full training
step, while elementwise kernels grow correspondingly: backward contributes large pointwise kernels
(gradient elementwise ops, ~500 ms and ~286 ms over the run) and AdamW is pure elementwise work
(10,125 invocations of its update kernels, 0% GEMM). Absolute GEMM time per step rises from ~62 ms
(forward only) to ~179 ms (fwd+bwd), but the step is ~3.4× longer, so the matmul fraction falls.

## (e) Softmax vs. matmul runtime inside self-attention (forward, small/1024)

From the annotated profile, per forward step (12 layers): the softmax NVTX range accounts for
~24.3 ms GPU time, while the two attention matmuls — QKᵀ (`computing attention scores` GEMM, ~4.8 ms)
and P·V (`final matmul` GEMM, ~7.6 ms) — total only ~12.4 ms. So softmax is ~2× slower than both
attention matmuls combined despite doing ~50× fewer FLOPs (≈0.25 GFLOP of exp/max/sum/divide vs.
≈12.9 GFLOP for the two 2·B·H·S²·d_k matmuls): softmax is memory-bandwidth-bound, streaming the
192 MiB FP32 score tensor through ~5 elementwise/reduction passes per layer, while the GEMMs are
compute-bound and run near SGEMM peak. (The masking/scaling elementwise ops in the scores range
cost another ~13 ms/step, more than the QKᵀ GEMM itself.)

## Reproduce

```sh
nsys profile --trace=cuda,nvtx --force-overwrite=true -o profiles/small_ctx1024_train \
  python cs336_systems/benchmarking_script.py \
    --model_size small --context_length 1024 --mode full-training-steps
nsys stats --report nvtx_gpu_proj_sum --report nvtx_kern_sum --report cuda_gpu_kern_sum \
  --format csv -o profiles/small_ctx1024_train profiles/small_ctx1024_train.nsys-rep
```

Raw artifacts: `profiles/*.nsys-rep`, parsed summaries: `profiles/*_nvtx_*.csv`,
`profiles/*_cuda_gpu_kern_sum.csv`.
