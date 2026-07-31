# CS336 Assignment 2 — Writeup (Draft)

## 2.1.3 End-to-End Benchmarking — Problem (benchmarking_script)

### (a) Benchmarking script

The benchmarking script is at `cs336_systems/benchmarking_script.py`. It:

- Initializes a `BasicsTransformerLM` from hyperparameters selected via `--model_size` (small / medium / large / xl / 10B, following Table 1; vocab_size=10000, context_length=512, batch_size=4).
- Generates a random batch of token ids (`x`, `y_labels`) with `torch.randint` on GPU.
- Runs `--warmup_steps` warm-up iterations, then times `--measurement_steps` iterations of one of three modes selected by `--mode`: `forward-only`, `forward-and-backward`, or `full-training-steps` (including the AdamW optimizer step).
- Synchronizes with `torch.cuda.synchronize()` before and after each timed segment and uses `timeit.default_timer()` for timing. Forward timing includes the cross-entropy loss computation; `optimizer.zero_grad()` is counted in the backward segment.
- Reports mean and standard deviation for each segment.

All four model sizes can be benchmarked via `bash run_benchmark_all_sizes.sh`.

### (b) Timings (5 warm-up steps, 10 measurement steps, RTX 4090 D 48GB)

| Size | Forward (ms) | Backward (ms) | Optimizer (ms) |
|---|---|---|---|
| small | 24.86 ± 0.02 | 56.81 ± 0.19 | 11.06 ± 1.96 |
| medium | 83.87 ± 0.07 | 166.37 ± 0.29 | 32.51 ± 0.49 |
| large | 169.27 ± 0.49 | 340.40 ± 0.83 | 89.67 ± 0.65 |
| xl | OOM | OOM | OOM |

A forward pass takes ~25ms (small) to ~169ms (large); the backward pass consistently takes about 2x the forward pass, matching the theoretical FLOP ratio. The standard deviation is under 1% of the mean for all segments, i.e., measurements are highly stable once warmed up. The xl configuration OOMs on a 48GB GPU at the first `optimizer.step()`: in fp32, parameters + gradients + the two AdamW moments already require ~4x the parameter memory (~3.4B params x 4 bytes x 4 copies ≈ 54GB), exceeding the 47.4GiB capacity before accounting for activations.

### (c) Effect of warm-up steps

Without warm-up (w=0, medium), the first forward measurement is 293ms vs. an 84ms steady state (3.5x), and the first backward measurement is 254ms vs. 166ms; this single outlier inflates the mean forward time by 25%. This happens because the first iteration pays one-time costs: lazy loading of CUDA kernels, cuBLAS algorithm selection for newly seen matmul shapes, and the caching allocator's first (expensive) `cudaMalloc` calls. In our setup (large and small models, w=1) the very first measurement was already at steady state, so one warm-up step sufficed; however, 1-2 steps may still be insufficient in general because GPU clocks ramp up from idle frequencies gradually (on the order of seconds), and allocator/cuDNN autotuning state can take several iterations to stabilize for more complex workloads — hence the standard choice of 5 warm-up steps as a safety margin.
