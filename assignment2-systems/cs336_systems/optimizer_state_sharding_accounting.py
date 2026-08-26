"""Memory and runtime accounting for optimizer state sharding (Section 6).

Compares DDP training with a full (non-sharded) AdamW optimizer vs. the
ShardedOptimizer from cs336_systems/optimizer_state_sharding.py, under the
standard configuration of this chapter (1 node x 2 GPUs; large model standing
in for xl, which needs ~25.4 GiB for FP32 parameters + gradients alone and does
not fit on these 24 GiB RTX 3090s -- same substitution as earlier sections).

Per variant we record:
  - memory_allocated right after model construction (parameters only),
  - memory_allocated directly before the optimizer step (parameters + gradients
    + optimizer states, on a warmed-up step so states exist),
  - memory_allocated directly after the optimizer step,
  - peak memory allocated,
  - per-step segment timings (forward / backward / grad sync / optimizer step;
    for the sharded variant the optimizer step includes the parameter-shard
    broadcast that keeps replicas in sync).

Both variants are measured back-to-back in the same process group so the
comparison is internally consistent even on a shared machine.

Run from the repo root, e.g.:
  CUDA_VISIBLE_DEVICES=5,6 python cs336_systems/optimizer_state_sharding_accounting.py
Results are appended to profiles/optimizer_state_sharding.csv.
"""

import argparse
import csv
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.naive_ddp_benchmarking import hyper_model_para_dict
from cs336_systems.ddp_overlap_individual_parameters import DDPOverlap
from cs336_systems.optimizer_state_sharding import ShardedOptimizer

RESULTS_CSV = "profiles/optimizer_state_sharding.csv"
FIELDNAMES = [
    "variant",
    "model_size",
    "world_size",
    "global_batch_size",
    "context_length",
    "mem_after_init_gib",
    "mem_before_opt_step_gib",
    "mem_after_opt_step_gib",
    "peak_memory_gib",
    "forward_ms",
    "backward_ms",
    "grad_sync_ms",
    "optimizer_step_ms",
    "total_ms",
]
GIB = 2**30


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def measure_variant(rank: int, world_size: int, args: argparse.Namespace, sharded: bool):
    config = hyper_model_para_dict[args.model_size]
    device = f"cuda:{rank}"

    torch.manual_seed(42)  # identical initial weights on all ranks and variants
    model = BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=args.context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    ).to(device)
    ddp_model = DDPOverlap(model)
    if sharded:
        optimizer = ShardedOptimizer(ddp_model.parameters(), AdamW, lr=1e-4)
    else:
        optimizer = AdamW(ddp_model.parameters(), lr=1e-4)

    mem_after_init = torch.cuda.memory_allocated() / GIB

    torch.manual_seed(1234)
    all_x = torch.randint(0, config["vocab_size"], (args.batch_size, args.context_length))
    all_y = torch.randint(0, config["vocab_size"], (args.batch_size, args.context_length))
    local_bs = args.batch_size // world_size
    offset = rank * local_bs
    x = all_x[offset : offset + local_bs].to(device)
    y = all_y[offset : offset + local_bs].to(device)

    def train_step(probe_memory: bool = False):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = ddp_model(x)
        loss = cross_entropy(logits, y)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        ddp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        before_opt = torch.cuda.memory_allocated() / GIB if probe_memory else None
        optimizer.step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        after_opt = torch.cuda.memory_allocated() / GIB if probe_memory else None

        return (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0), before_opt, after_opt

    train_step()  # first step allocates optimizer states
    torch.cuda.reset_peak_memory_stats()
    _, mem_before_opt, mem_after_opt = train_step(probe_memory=True)
    for _ in range(args.warmup_steps):
        train_step()
    timings = [train_step()[0] for _ in range(args.measurement_steps)]
    peak = torch.cuda.max_memory_allocated() / GIB
    return mem_after_init, mem_before_opt, mem_after_opt, peak, timings


def worker(rank: int, world_size: int, args: argparse.Namespace, port: int) -> None:
    setup(rank, world_size, port)
    sharded = args.sharded
    mem_after_init, mem_before, mem_after, peak, timings = measure_variant(
        rank, world_size, args, sharded
    )

    gathered = [None] * world_size
    dist.all_gather_object(gathered, (mem_after_init, mem_before, mem_after, peak, timings))
    if rank == 0:
        mem_after_init, mem_before, mem_after = (
            statistics.mean(m[i] for m in gathered) for i in range(3)
        )
        peak = max(m[3] for m in gathered)
        timings = [t for m in gathered for t in m[4]]
        fwd, bwd, sync, opt, total = (statistics.mean(c) * 1e3 for c in zip(*timings))
        row = {
            "variant": "sharded" if sharded else "full",
            "model_size": args.model_size,
            "world_size": world_size,
            "global_batch_size": args.batch_size,
            "context_length": args.context_length,
            "mem_after_init_gib": f"{mem_after_init:.2f}",
            "mem_before_opt_step_gib": f"{mem_before:.2f}",
            "mem_after_opt_step_gib": f"{mem_after:.2f}",
            "peak_memory_gib": f"{peak:.2f}",
            "forward_ms": f"{fwd:.2f}",
            "backward_ms": f"{bwd:.2f}",
            "grad_sync_ms": f"{sync:.2f}",
            "optimizer_step_ms": f"{opt:.2f}",
            "total_ms": f"{total:.2f}",
        }
        write_header = not os.path.exists(RESULTS_CSV)
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[{'sharded' if sharded else 'full'} optimizer] "
            f"mem init/before-opt/after-opt/peak: {mem_after_init:.2f} / {mem_before:.2f} / "
            f"{mem_after:.2f} / {peak:.2f} GiB | step {total:.2f} ms "
            f"(fwd {fwd:.2f}, bwd {bwd:.2f}, grad-sync {sync:.2f}, opt {opt:.2f})"
        )
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=list(hyper_model_para_dict), default="large")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4, help="global batch size")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--measurement_steps", type=int, default=5)
    parser.add_argument("--port", type=int, default=29613)
    args = parser.parse_args()

    assert args.batch_size % args.world_size == 0
    # Each variant runs in a fresh set of spawned processes: the DDP overlap
    # hooks anchor the model in memory (param -> C++ autograd meta -> hook ->
    # DDP container -> module -> param), which Python gc cannot reclaim, so an
    # in-process variant switch would leak the previous model.
    port = args.port
    for sharded in (False, True):
        args.sharded = sharded
        mp.spawn(worker, args=(args.world_size, args, port), nprocs=args.world_size, join=True)
        port += 1


if __name__ == "__main__":
    main()
