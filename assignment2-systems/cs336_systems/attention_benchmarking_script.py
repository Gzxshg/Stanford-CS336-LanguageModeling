"""pytorch_attention: benchmark the naive PyTorch scaled_dot_product_attention.

Sweeps d_model x seq_len, times 100 forward and 100 backward passes, and records
memory in use right before the backward pass starts. Batch 8, no head dim, FP32.
"""
import argparse
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention

D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
BATCH = 8
WARMUP = 5
ITERS = 100

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=ITERS)
parser.add_argument("--warmup", type=int, default=WARMUP)
parser.add_argument("--compile", action="store_true", help="benchmark a torch.compile'd attention instead")
args = parser.parse_args()

attn = scaled_dot_product_attention
if args.compile:
    attn = torch.compile(scaled_dot_product_attention)

print(f"config: batch={BATCH} dtype=fp32 iters={args.iters} compile={args.compile}")
print("d_model,seq_len,fwd_ms,bwd_ms,mem_before_bwd_MiB,status")

for d in D_MODELS:
    for S in SEQ_LENS:
        try:
            Q = torch.randn(BATCH, S, d, device="cuda", requires_grad=True)
            K = torch.randn(BATCH, S, d, device="cuda", requires_grad=True)
            V = torch.randn(BATCH, S, d, device="cuda", requires_grad=True)

            def one_step():
                out = attn(Q, K, V)
                loss = out.sum()
                loss.backward()
                Q.grad = K.grad = V.grad = None

            for _ in range(args.warmup):
                one_step()

            fwd_times, bwd_times = [], []
            mem_before_bwd = None
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = timeit.default_timer()
                out = attn(Q, K, V)
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                if mem_before_bwd is None:
                    mem_before_bwd = torch.cuda.memory_allocated() / 2**20
                out.sum().backward()
                torch.cuda.synchronize()
                t2 = timeit.default_timer()
                Q.grad = K.grad = V.grad = None
                fwd_times.append(t1 - t0)
                bwd_times.append(t2 - t1)
            print(f"{d},{S},{1e3*sum(fwd_times)/len(fwd_times):.3f},"
                  f"{1e3*sum(bwd_times)/len(bwd_times):.3f},{mem_before_bwd:.1f},ok", flush=True)
        except torch.OutOfMemoryError:
            print(f"{d},{S},,,,OOM", flush=True)
        finally:
            torch.cuda.empty_cache()
