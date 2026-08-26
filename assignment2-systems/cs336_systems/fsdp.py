"""Fully-sharded data parallel (FSDP) training wrapper (Section 7).

Each Linear/Embedding weight is flattened and sharded across ranks (zero-padded
to an even split); every rank keeps only its FP32 master shard. Compute uses
custom autograd Functions:

  - forward: all-gather the full weight (casting shards to compute_dtype *before*
    communication to save bandwidth when mixed precision is requested), run the
    layer math, and release the gathered weight (nothing full-size is saved for
    backward -- activations are recomputed inputs only),
  - backward: all-gather the weight again (Linear needs it for grad_input;
    Embedding's backward needs no weight at all), compute the full weight
    gradient, then reduce-scatter it so each rank keeps the summed gradient for
    its own shard. Gradients are divided by world_size in
    finish_gradient_synchronization (sum over ranks of per-shard local means =
    world_size x global mean, since losses average over local batches).

To hide communication, the weight all-gather for layer i+2 is prefetched
 asynchronously when layer i's forward completes (and symmetrically for earlier
layers during backward); layers whose prefetch hasn't landed gather
synchronously on demand. Small modules (norms etc.) stay replicated and their
gradients are plain all-reduced in finish_gradient_synchronization.
"""

import math
from types import SimpleNamespace

import torch
import torch.distributed as dist
from einops import einsum

from cs336_basics.model import Embedding, Linear


def _reduce_scatter_shard(flat_full: torch.Tensor, shard_size: int, rank: int, world_size: int) -> torch.Tensor:
    """Sum a flat gradient across ranks and return this rank's shard."""
    if dist.get_backend() == "nccl":
        out = torch.empty(shard_size, dtype=flat_full.dtype, device=flat_full.device)
        dist.reduce_scatter_tensor(out, flat_full, op=dist.ReduceOp.SUM)
        return out
    # gloo has no reduce_scatter for CUDA tensors: all-reduce, then slice.
    dist.all_reduce(flat_full, op=dist.ReduceOp.SUM)
    return flat_full[rank * shard_size : (rank + 1) * shard_size].contiguous()


class _ShardedLinear(torch.autograd.Function):
    """y = x @ W^T with W sharded across ranks (matches cs336_basics Linear math)."""

    @staticmethod
    def forward(ctx, x, shard, meta, fsdp):
        ctx.meta = meta
        ctx.fsdp = fsdp
        ctx.save_for_backward(x)
        W = fsdp._gather_weight(meta, fsdp._fwd_prefetch)
        return einsum(x, W, "... d_in, d_out d_in -> ... d_out")

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        meta, fsdp = ctx.meta, ctx.fsdp
        W = fsdp._gather_weight(meta, fsdp._bwd_prefetch)
        # Now that this layer's weight is materialized, kick off the gather for
        # the layer two steps earlier in the backward order.
        fsdp._start_prefetch(meta.idx - 2, fsdp._bwd_prefetch)
        dx = einsum(dy, W, "... d_out, d_out d_in -> ... d_in")
        dW = einsum(dy, x, "... d_out, ... d_in -> d_out d_in")
        dshard = fsdp._reduce_weight_grad(meta, dW)
        return dx, dshard, None, None


class _ShardedEmbedding(torch.autograd.Function):
    """y = W[token_ids] with W sharded across ranks (matches cs336_basics Embedding)."""

    @staticmethod
    def forward(ctx, token_ids, shard, meta, fsdp):
        ctx.meta = meta
        ctx.fsdp = fsdp
        ctx.save_for_backward(token_ids)
        W = fsdp._gather_weight(meta, fsdp._fwd_prefetch)
        return W[token_ids]

    @staticmethod
    def backward(ctx, dout):
        (token_ids,) = ctx.saved_tensors
        meta, fsdp = ctx.meta, ctx.fsdp
        vocab_size, d_model = meta.orig_shape
        flat_ids = token_ids.reshape(-1)
        src = dout.reshape(-1, d_model)
        if torch.are_deterministic_algorithms_enabled():
            # index_add_ on CUDA is non-deterministic under repeated indices, so
            # use a matmul with one-hot rows instead.
            one_hot = torch.nn.functional.one_hot(flat_ids, vocab_size).to(src.dtype)
            dW = one_hot.T @ src
        else:
            dW = torch.zeros(vocab_size, d_model, dtype=src.dtype, device=src.device)
            dW.index_add_(0, flat_ids, src)
        fsdp._start_prefetch(meta.idx - 2, fsdp._bwd_prefetch)
        dshard = fsdp._reduce_weight_grad(meta, dW)
        return None, dshard, None, None


class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._layers: list[SimpleNamespace] = []
        self._sharded_ids: set[int] = set()
        self._meta_by_param: dict[int, SimpleNamespace] = {}
        self._fwd_prefetch: dict[int, tuple] = {}
        self._bwd_prefetch: dict[int, tuple] = {}
        self._shard_and_patch()
        self._register_prefetch_hooks()

    # ---- setup ----

    def _shard_and_patch(self) -> None:
        targets = [m for m in self.module.modules() if isinstance(m, (Linear, Embedding))]
        for idx, mod in enumerate(targets):
            W = mod.weight
            numel = W.numel()
            shard_size = math.ceil(numel / self.world_size)
            padded_numel = shard_size * self.world_size

            # Ensure identical starting points before slicing (ranks may have
            # been seeded differently).
            dist.broadcast(W.data, src=0)
            flat = W.detach().reshape(-1).to(torch.float32)
            if padded_numel != numel:
                flat = torch.cat([flat, torch.zeros(padded_numel - numel, dtype=flat.dtype, device=flat.device)])
            shard = flat[self.rank * shard_size : (self.rank + 1) * shard_size].contiguous().clone()

            meta = SimpleNamespace(
                module=mod,
                orig_shape=tuple(W.shape),
                numel=numel,
                shard_size=shard_size,
                padded_numel=padded_numel,
                idx=idx,
                is_embedding=isinstance(mod, Embedding),
            )
            shard_param = torch.nn.Parameter(shard, requires_grad=W.requires_grad)
            mod.weight = shard_param  # replaces the full-weight Parameter
            self._layers.append(meta)
            self._sharded_ids.add(id(shard_param))
            self._meta_by_param[id(shard_param)] = meta

            if meta.is_embedding:
                mod.forward = lambda token_ids, m=mod, mt=meta: _ShardedEmbedding.apply(token_ids, m.weight, mt, self)
            else:
                mod.forward = lambda x, m=mod, mt=meta: _ShardedLinear.apply(x, m.weight, mt, self)

    def _register_prefetch_hooks(self) -> None:
        for meta in self._layers:
            # When layer i's forward completes, start gathering layer i+2's weight.
            meta.module.register_forward_hook(
                lambda mod, inp, out, i=meta.idx: self._start_prefetch(i + 2, self._fwd_prefetch)
            )

    # ---- communication helpers ----

    def _cast_for_compute(self, shard: torch.Tensor) -> torch.Tensor:
        if self.compute_dtype is not None and shard.dtype != self.compute_dtype:
            return shard.to(self.compute_dtype)
        return shard

    def _start_prefetch(self, idx: int, cache: dict) -> None:
        if idx < 0 or idx >= len(self._layers) or idx in cache:
            return
        meta = self._layers[idx]
        src = self._cast_for_compute(meta.module.weight.detach()).contiguous()
        buf = [torch.empty_like(src) for _ in range(self.world_size)]
        work = dist.all_gather(buf, src, async_op=True)
        cache[idx] = (work, buf)

    def _gather_weight(self, meta: SimpleNamespace, cache: dict) -> torch.Tensor:
        if meta.idx in cache:
            work, buf = cache.pop(meta.idx)
            work.wait()
            flat = torch.cat(buf)
        else:
            src = self._cast_for_compute(meta.module.weight.detach()).contiguous()
            buf = [torch.empty_like(src) for _ in range(self.world_size)]
            dist.all_gather(buf, src)
            flat = torch.cat(buf)
        return flat[: meta.numel].view(meta.orig_shape)

    def _reduce_weight_grad(self, meta: SimpleNamespace, dW_full: torch.Tensor) -> torch.Tensor:
        grad = dW_full.to(torch.float32).reshape(-1)  # master dtype stays fp32
        if grad.numel() < meta.padded_numel:
            grad = torch.cat(
                [grad, torch.zeros(meta.padded_numel - grad.numel(), dtype=grad.dtype, device=grad.device)]
            )
        return _reduce_scatter_shard(grad.contiguous(), meta.shard_size, self.rank, self.world_size)

    # ---- public interface ----

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # Shard gradients are already summed across ranks (reduce-scatter);
        # replicated parameters (norms etc.) still hold per-rank local means and
        # need a full all-reduce. Both are divided by world_size so the result
        # is the mean over the global batch.
        for p in self.module.parameters():
            if p.grad is None:
                continue
            if id(p) not in self._sharded_ids:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= self.world_size

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full = {}
        for name, p in self.module.named_parameters():
            if id(p) in self._sharded_ids:
                meta = self._meta_by_param[id(p)]
                src = p.detach().contiguous()  # fp32 master shard, never cast
                buf = [torch.empty_like(src) for _ in range(self.world_size)]
                dist.all_gather(buf, src)
                full[name] = torch.cat(buf)[: meta.numel].view(meta.orig_shape)
            else:
                full[name] = p.data
        return full
