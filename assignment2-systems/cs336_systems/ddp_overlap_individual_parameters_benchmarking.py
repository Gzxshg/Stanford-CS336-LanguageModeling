"""Benchmark DDP with per-parameter gradient communication overlapped with the
backward pass (Section 5.3.2).

Identical setup to cs336_systems/naive_ddp_benchmarking.py (same model, batch,
optimizer, timing methodology) except the model is wrapped in
cs336_systems/ddp_overlap_individual_parameters.py's DDPOverlap, which issues an
asynchronous all-reduce for each parameter gradient as soon as it is ready during
the backward pass. Note the segment semantics differ from the naive/flat runs:
the "backward" segment now includes the *launch* of gradient communication (the
NCCL kernels run concurrently on their own stream), and "grad_sync" measures only
the residual wait in finish_gradient_synchronization().

Run from the repo root, e.g.:
  CUDA_VISIBLE_DEVICES=0,1 python cs336_systems/ddp_overlap_individual_parameters_benchmarking.py
Results are appended to profiles/ddp_overlap_benchmark.csv.
"""

import argparse

import torch.multiprocessing as mp

from cs336_systems.naive_ddp_benchmarking import benchmark_worker

RESULTS_CSV = "profiles/ddp_overlap_benchmark.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4, help="global batch size")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measurement_steps", type=int, default=10)
    parser.add_argument("--port", type=int, default=29612)
    args = parser.parse_args()
    args.variant = "overlap"
    args.results_csv = RESULTS_CSV

    assert args.batch_size % args.world_size == 0
    mp.spawn(benchmark_worker, args=(args.world_size, args, args.port), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
