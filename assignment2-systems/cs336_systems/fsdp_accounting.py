"""FSDP accounting (Section 7, fsdp_accounting): memory and all-gather timing.

Runs the xl model under FSDP on 2 GPUs -- a configuration that does not fit
with plain DDP on 24 GiB cards (FP32 parameters + gradients alone need ~25.4
GiB) but fits once weights and gradients are sharded.

Two pragmatic choices, both disclosed in the writeup:
  - compute_dtype=fp32: the cs336 transformer keeps its RoPE frequency cache as
    an fp32 buffer, so fp16 compute would need buffer casting beyond the
    weight casting FSDP performs (the fp16 weight path itself is covered by
    tests/test_fsdp.py on the toy model).
  - SGD instead of AdamW: sharded xl keeps 6.83 GiB/rank for master weights and
    6.83 GiB/rank for gradients; AdamW would add another 6.83 GiB/rank of
    sharded moments, which exceeds the free memory left by a co-tenant job on
    one of the two GPUs. The question answered here -- whether weight
    all-gathers finish in time for the forward pass -- is orthogonal to the
    optimizer.

Per rank we report memory_allocated after model construction (sharded master
weights), before/after the optimizer step, and the peak, plus per-step segment
timings (forward / backward / residual grad sync / optimizer step). NVTX ranges
are emitted so the nsys trace can be sliced per step; the question answered in
the writeup is whether the weight all-gathers complete in time for the forward
pass on this machine's interconnect.

Run from the repo root, e.g.:
  CUDA_VISIBLE_DEVICES=5,6 python cs336_systems/fsdp_accounting.py
  nsys profile --trace=cuda,nvtx -o profiles/fsdp_xl_trace --force-overwrite true \
      python cs336_systems/fsdp_accounting.py --profile_steps 3
Results are appended to profiles/fsdp_accounting.csv.
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
from cs336_systems.naive_ddp_benchmarking import hyper_model_para_dict
from cs336_systems.fsdp import FSDP

RESULTS_CSV = "profiles/fsdp_accounting.csv"
FIELDNAMES = [
    "model_size",
    "world_size",
    "global_batch_size",
    "context_length",
    "compute_dtype",
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


def worker(rank: int, world_size: int, args: argparse.Namespace, port: int) -> None:
    setup(rank, world_size, port)
    config = hyper_model_para_dict[args.model_size]
    device = f"cuda:{rank}"
    compute_dtype = {"fp32": None, "fp16": torch.float16}[args.compute_dtype]

    torch.manual_seed(42)  # identical init on all ranks
    model = BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=args.context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    ).to(device)
    fsdp_model = FSDP(model, compute_dtype=compute_dtype)
    # FSDP shards the parameters themselves, so a plain optimizer over
    # fsdp_model.parameters() updates local master shards with local shard
    # gradients. SGD keeps no moments, which is what lets this profile fit on
    # the partially-occupied GPUs available here (see module docstring).
    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=1e-4)

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

        with nvtx.range("forward"):
            logits = fsdp_model(x)
            loss = cross_entropy(logits, y)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        with nvtx.range("backward"):
            loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        with nvtx.range("grad_sync"):
            fsdp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        before_opt = torch.cuda.memory_allocated() / GIB if probe_memory else None
        with nvtx.range("optimizer_step"):
            optimizer.step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        after_opt = torch.cuda.memory_allocated() / GIB if probe_memory else None
        return (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0), before_opt, after_opt

    train_step()  # allocates optimizer states
    torch.cuda.reset_peak_memory_stats()
    _, mem_before, mem_after = train_step(probe_memory=True)
    timings = []
    for _ in range(args.profile_steps):
        timings.append(train_step()[0])
    peak = torch.cuda.max_memory_allocated() / GIB

    gathered = [None] * world_size
    dist.all_gather_object(gathered, (mem_after_init, mem_before, mem_after, peak, timings))
    if rank == 0:
        m0 = statistics.mean(g[0] for g in gathered)
        m1 = statistics.mean(g[1] for g in gathered)
        m2 = statistics.mean(g[2] for g in gathered)
        pk = max(g[3] for g in gathered)
        flat_t = [t for g in gathered for t in g[4]]
        fwd, bwd, sync, opt, total = (statistics.mean(c) * 1e3 for c in zip(*flat_t))
        row = {
            "model_size": args.model_size,
            "world_size": world_size,
            "global_batch_size": args.batch_size,
            "context_length": args.context_length,
            "compute_dtype": args.compute_dtype,
            "mem_after_init_gib": f"{m0:.2f}",
            "mem_before_opt_step_gib": f"{m1:.2f}",
            "mem_after_opt_step_gib": f"{m2:.2f}",
            "peak_memory_gib": f"{pk:.2f}",
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
            f"[fsdp {args.model_size} {args.compute_dtype}, world_size={world_size}] "
            f"mem init/before-opt/after-opt/peak: {m0:.2f} / {m1:.2f} / {m2:.2f} / {pk:.2f} GiB | "
            f"step {total:.2f} ms (fwd {fwd:.2f}, bwd {bwd:.2f}, grad-sync {sync:.2f}, opt {opt:.2f})"
        )
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=list(hyper_model_para_dict), default="xl")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2, help="global batch size")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--compute_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--profile_steps", type=int, default=3)
    parser.add_argument("--port", type=int, default=29614)
    args = parser.parse_args()

    assert args.batch_size % args.world_size == 0
    mp.spawn(worker, args=(args.world_size, args, args.port), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
