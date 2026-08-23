"""Benchmark NCCL all-reduce runtime in a single-node multi-process setup.

Sweeps two factors:
  - data size of the float32 tensor being all-reduced (1MB, 10MB, 100MB, 1GB)
  - number of worker processes/GPUs (default: 2 and 4)

Methodology (following the assignment's best-practice guidelines):
  - 5 warm-up all-reduce calls before timing (NCCL needs warm-up)
  - torch.cuda.synchronize() before/after every timed call, because
    async_op=False only guarantees the op is *queued*, not *finished*
  - per-rank timings are collected on rank 0 via dist.all_gather_object
    and aggregated across ranks and iterations

Run from the repo root, e.g.:
  CUDA_VISIBLE_DEVICES=0,3 python cs336_systems/distributed_communication_single_node.py --world-sizes 2
  CUDA_VISIBLE_DEVICES=0,1,2,3 python cs336_systems/distributed_communication_single_node.py --world-sizes 4
Results are appended to profiles/allreduce_benchmark.csv and plotted to
writeup_assets/allreduce_benchmark.png once all configurations finish.
"""

import argparse
import csv
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MB = [1, 10, 100, 1024]  # 1024 MB == 1 GB
WARMUP_ITERS = 5
TIMED_ITERS = 20
BASE_PORT = 29600
RESULTS_CSV = "profiles/allreduce_benchmark.csv"
RESULTS_PNG = "writeup_assets/allreduce_benchmark.png"
FIELDNAMES = [
    "world_size",
    "size_mb",
    "mean_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "algbw_gbps",
    "busbw_gbps",
]


def size_label(size_mb: int) -> str:
    return "1GB" if size_mb == 1024 else f"{size_mb}MB"


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)  # per-rank GPU binding
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def benchmark_worker(rank: int, world_size: int, size_mb: int, port: int) -> None:
    setup(rank, world_size, port)

    numel = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
    data = torch.randn(numel, device=f"cuda:{rank}")

    for _ in range(WARMUP_ITERS):
        dist.all_reduce(data)
    torch.cuda.synchronize()

    times = []
    for _ in range(TIMED_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(data)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Gather per-rank timings on rank 0 and aggregate.
    gathered = [None] * world_size
    dist.all_gather_object(gathered, times)
    if rank == 0:
        flat = [t for rank_times in gathered for t in rank_times]
        mean_s = statistics.mean(flat)
        size_bytes = size_mb * 1024 * 1024
        algbw = size_bytes / mean_s / 1e9  # algorithmic bandwidth
        busbw = algbw * 2 * (world_size - 1) / world_size  # ring all-reduce bus bw
        row = {
            "world_size": world_size,
            "size_mb": size_mb,
            "mean_ms": f"{mean_s * 1e3:.3f}",
            "median_ms": f"{statistics.median(flat) * 1e3:.3f}",
            "min_ms": f"{min(flat) * 1e3:.3f}",
            "max_ms": f"{max(flat) * 1e3:.3f}",
            "algbw_gbps": f"{algbw:.2f}",
            "busbw_gbps": f"{busbw:.2f}",
        }
        write_header = not os.path.exists(RESULTS_CSV)
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[world_size={world_size} size={size_label(size_mb)}] "
            f"mean={row['mean_ms']} ms  busbw={row['busbw_gbps']} GB/s"
        )

    dist.destroy_process_group()


def plot_results(csv_path: str, png_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(
                (int(row["world_size"]), int(row["size_mb"]), float(row["mean_ms"]), float(row["busbw_gbps"]))
            )

    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ws in sorted({r[0] for r in rows}):
        pts = sorted([r for r in rows if r[0] == ws], key=lambda r: r[1])
        xs = [p[1] for p in pts]
        ax_t.plot(xs, [p[2] for p in pts], "o-", label=f"{ws} GPUs")
        ax_b.plot(xs, [p[3] for p in pts], "o-", label=f"{ws} GPUs")

    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("all-reduce data size (MB)")
    ax_t.set_ylabel("time per all-reduce (ms)")
    ax_t.set_title("all-reduce runtime vs. data size")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend()

    ax_b.set_xscale("log")
    ax_b.set_xlabel("all-reduce data size (MB)")
    ax_b.set_ylabel("busbw (GB/s)")
    ax_b.set_title("effective bus bandwidth vs. data size")
    ax_b.grid(True, which="both", alpha=0.3)
    ax_b.legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"plot saved to {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-sizes", type=str, default="2,4")
    parser.add_argument("--sizes-mb", type=str, default=",".join(str(s) for s in SIZES_MB))
    args = parser.parse_args()
    world_sizes = [int(x) for x in args.world_sizes.split(",")]
    sizes_mb = [int(x) for x in args.sizes_mb.split(",")]

    n_visible = torch.cuda.device_count()
    port = BASE_PORT
    for ws in world_sizes:
        if ws > n_visible:
            print(f"skipping world_size={ws}: only {n_visible} GPUs visible")
            continue
        for size_mb in sizes_mb:
            mp.spawn(benchmark_worker, args=(ws, size_mb, port), nprocs=ws, join=True)
            port += 1

    if os.path.exists(RESULTS_CSV):
        plot_results(RESULTS_CSV, RESULTS_PNG)


if __name__ == "__main__":
    main()