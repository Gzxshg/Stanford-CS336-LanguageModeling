"""Benchmark minimal DDP with flattened gradients (Section 5.3.1).

Identical setup to cs336_systems/naive_ddp_benchmarking.py (same model, batch,
optimizer, timing methodology) except gradients are concatenated into a single
flat tensor and communicated with one all-reduce per step
(DDP(flat_gradients=True)), instead of one all-reduce per parameter tensor.
The harness is shared with the naive benchmark so the comparison is
apples-to-apples.

Run from the repo root, e.g.:
  CUDA_VISIBLE_DEVICES=0,1 python cs336_systems/minimal_ddp_flat_benchmarking.py
Results are appended to profiles/minimal_ddp_flat_benchmark.csv.
"""

import argparse

import torch.multiprocessing as mp

from cs336_systems.naive_ddp_benchmarking import benchmark_worker

RESULTS_CSV = "profiles/minimal_ddp_flat_benchmark.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4, help="global batch size")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measurement_steps", type=int, default=10)
    parser.add_argument("--port", type=int, default=29611)
    args = parser.parse_args()
    args.variant = "flat"
    args.results_csv = RESULTS_CSV

    assert args.batch_size % args.world_size == 0
    mp.spawn(benchmark_worker, args=(args.world_size, args, args.port), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
