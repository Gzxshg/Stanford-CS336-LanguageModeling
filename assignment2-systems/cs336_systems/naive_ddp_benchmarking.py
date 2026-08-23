"""Benchmark naive DDP training: total step time and gradient-communication share.

Each rank holds a full replica of the model (wrapped in cs336_systems.naive_ddp.DDP),
trains on a disjoint shard of every batch, and all-reduces each parameter's
gradient individually after the backward pass
(DDP.finish_gradient_synchronization).

Per training step we measure four segments, each bracketed by
torch.cuda.synchronize(): forward, backward, gradient communication
(finish_gradient_synchronization), and optimizer step. Timings are aggregated
across ranks on rank 0 via dist.all_gather_object.

Note: the handout specifies the xl model (Section 2.1.2), but xl (3.41B params)
needs ~25.4 GiB for FP32 parameters + gradients alone, which exceeds the 24 GiB
RTX 3090s available here, so we substitute the large model (0.97B params,
~14.4 GiB static FP32 training state incl. AdamW moments) -- same substitution
as in the memory-profiling section of the writeup.

Run from the repo root, e.g.:
  CUDA_VISIBLE_DEVICES=0,1 python cs336_systems/naive_ddp_benchmarking.py
Results are appended to profiles/naive_ddp_benchmark.csv.
"""

import argparse
import csv
import os
import statistics
import time

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.naive_ddp import DDP
from cs336_systems.ddp_overlap_individual_parameters import DDPOverlap

hyper_model_para_dict = {
    "small": {"vocab_size": 10000, "context_length": 512, "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"vocab_size": 10000, "context_length": 512, "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"vocab_size": 10000, "context_length": 512, "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"vocab_size": 10000, "context_length": 512, "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

RESULTS_CSV = "profiles/naive_ddp_benchmark.csv"
FIELDNAMES = [
    "model_size",
    "world_size",
    "global_batch_size",
    "context_length",
    "forward_ms",
    "backward_ms",
    "grad_sync_ms",
    "optimizer_ms",
    "total_ms",
    "comm_fraction",
    "peak_memory_gib",
]


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def benchmark_worker(rank: int, world_size: int, args: argparse.Namespace, port: int) -> None:
    setup(rank, world_size, port)
    device = f"cuda:{rank}"
    config = hyper_model_para_dict[args.model_size]

    # Identical initial weights on all ranks (DDP also broadcasts from rank 0).
    torch.manual_seed(42)
    model = BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=args.context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    ).to(device)
    variant = getattr(args, "variant", "naive")
    if variant == "overlap":
        ddp_model = DDPOverlap(model)
    else:
        ddp_model = DDP(model, flat_gradients=(variant == "flat"))
    optimizer = AdamW(ddp_model.parameters(), lr=1e-4)

    # One global batch, generated identically on every rank; each rank trains on
    # its own disjoint shard.
    torch.manual_seed(1234)
    all_x = torch.randint(0, config["vocab_size"], (args.batch_size, args.context_length))
    all_y = torch.randint(0, config["vocab_size"], (args.batch_size, args.context_length))
    local_bs = args.batch_size // world_size
    offset = rank * local_bs
    x = all_x[offset : offset + local_bs].to(device)
    y = all_y[offset : offset + local_bs].to(device)

    def train_step():
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with nvtx.range("forward"):
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        with nvtx.range("backward"):
            loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        with nvtx.range("grad_sync"):
            ddp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        with nvtx.range("optimizer_step"):
            optimizer.step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        # forward / backward / gradient communication / optimizer / total
        return t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0

    for _ in range(args.warmup_steps):
        train_step()

    torch.cuda.reset_peak_memory_stats()
    timings = [train_step() for _ in range(args.measurement_steps)]
    peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30

    gathered = [None] * world_size
    dist.all_gather_object(gathered, (timings, peak_memory_gib))
    if rank == 0:
        flat = [t for rank_timings, _ in gathered for t in rank_timings]
        peak_gib = max(p for _, p in gathered)
        means = [statistics.mean(col) for col in zip(*flat)]
        fwd, bwd, sync, opt, total = (m * 1e3 for m in means)
        row = {
            "model_size": args.model_size,
            "world_size": world_size,
            "global_batch_size": args.batch_size,
            "context_length": args.context_length,
            "forward_ms": f"{fwd:.2f}",
            "backward_ms": f"{bwd:.2f}",
            "grad_sync_ms": f"{sync:.2f}",
            "optimizer_ms": f"{opt:.2f}",
            "total_ms": f"{total:.2f}",
            "comm_fraction": f"{sync / total:.3f}",
            "peak_memory_gib": f"{peak_gib:.2f}",
        }
        variant = getattr(args, "variant", "naive")
        results_csv = getattr(args, "results_csv", RESULTS_CSV)
        write_header = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[{variant} ddp: {args.model_size}, world_size={world_size}, "
            f"global batch={args.batch_size}, ctx={args.context_length}]"
        )
        print(f"  total per step:      {total:8.2f} ms")
        print(f"  forward:             {fwd:8.2f} ms")
        print(f"  backward:            {bwd:8.2f} ms")
        print(f"  gradient all-reduce: {sync:8.2f} ms  ({sync / total:.1%} of step)")
        print(f"  optimizer step:      {opt:8.2f} ms")
        print(f"  peak GPU memory:     {peak_gib:8.2f} GiB")

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=list(hyper_model_para_dict), default="large")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4, help="global batch size")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measurement_steps", type=int, default=10)
    parser.add_argument("--port", type=int, default=29610)
    parser.add_argument("--variant", choices=["naive", "flat", "overlap"], default="naive")
    parser.add_argument("--results_csv", type=str, default=RESULTS_CSV)
    args = parser.parse_args()

    assert args.batch_size % args.world_size == 0
    mp.spawn(benchmark_worker, args=(args.world_size, args, args.port), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
