"""flash_benchmarking: Triton FlashAttention-2 vs naive PyTorch attention.

batch 1, causal masking, sweep seq_len x d_model x dtype. Reports forward,
backward, and end-to-end latencies via triton.testing.do_bench.
"""
import torch
import triton
from einops import einsum

from cs336_systems.flash_attention_triton import FlashAttentionTriton

torch.random.manual_seed(0)

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DIMS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]


def pytorch_attention(q, k, v):
    d = q.shape[-1]
    S = einsum(q, k, "... q d, ... k d -> ... q k") / (d ** 0.5)
    nq, nk = S.shape[-2], S.shape[-1]
    q_idx = torch.arange(nq, device=S.device)[:, None]
    k_idx = torch.arange(nk, device=S.device)[None, :]
    S = torch.where(q_idx >= k_idx, S, torch.tensor(-1e6, dtype=S.dtype, device=S.device))
    P = torch.softmax(S, dim=-1)
    return einsum(P, v, "... q k, ... k d -> ... q d")


def bench(fn, grad=None):
    if grad is None:
        return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")
    def fwd_bwd():
        out = fn()
        out.backward(grad)
    return triton.testing.do_bench(fwd_bwd, warmup=10, rep=50, return_mode="median")


print("impl,dtype,d,seq_len,fwd_ms,bwd_ms,e2e_ms,status", flush=True)
for dtype in DTYPES:
    for d in DIMS:
        for S in SEQ_LENS:
            for impl_name, impl in [("triton", lambda q, k, v: FlashAttentionTriton.apply(q, k, v, True)),
                                    ("pytorch", pytorch_attention)]:
                tag = f"{impl_name},{str(dtype).split('.')[-1]},{d},{S}"
                try:
                    q = torch.randn(1, S, d, device="cuda", dtype=dtype, requires_grad=True)
                    k = torch.randn(1, S, d, device="cuda", dtype=dtype, requires_grad=True)
                    v = torch.randn(1, S, d, device="cuda", dtype=dtype, requires_grad=True)
                    do = torch.randn(1, S, d, device="cuda", dtype=dtype)
                    try:
                        fwd_ms = bench(lambda: impl(q, k, v))
                    except torch.OutOfMemoryError:
                        print(f"{tag},,,,OOM(fwd)", flush=True)
                        continue
                    for t in (q, k, v):
                        t.grad = None
                    try:
                        e2e_ms = bench(lambda: impl(q, k, v), grad=do)
                        bwd_ms = e2e_ms - fwd_ms
                        print(f"{tag},{fwd_ms:.3f},{bwd_ms:.3f},{e2e_ms:.3f},ok", flush=True)
                    except torch.OutOfMemoryError:
                        print(f"{tag},{fwd_ms:.3f},,,OOM(bwd)", flush=True)
                except torch.OutOfMemoryError:
                    print(f"{tag},,,,OOM", flush=True)
                finally:
                    torch.cuda.empty_cache()
