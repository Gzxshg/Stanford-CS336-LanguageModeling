"""Naive distributed data parallel (DDP) training wrapper.

At construction, the wrapped module's parameters and buffers are broadcast from
rank 0 so every rank starts from an identical replica. Gradient synchronization
happens after the backward pass: `finish_gradient_synchronization()` all-reduces
(sums) each parameter's gradient across ranks and divides by the world size, so
every rank applies the optimizer step to gradients averaged over the global
batch. Since all ranks start from identical parameters/optimizer state and see
identical averaged gradients, the replicas stay in sync.

Two communication variants are supported:
  - flat_gradients=False (naive, Section 5.2): one all-reduce per parameter tensor.
  - flat_gradients=True (Section 5.3.1): all gradients are concatenated into a
    single flat tensor (torch._utils._flatten_dense_tensors) and communicated
    with one all-reduce, then written back, amortizing per-call overhead.
"""

import torch
import torch.distributed as dist


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, flat_gradients: bool = False):
        super().__init__()
        self.module = module
        self.flat_gradients = flat_gradients
        self._sync_initial_state()

    def _sync_initial_state(self) -> None:
        # Broadcast parameters and buffers from rank 0 (state_dict covers both,
        # and its tensors share storage with the module, so this updates the
        # module in place).
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        world_size = dist.get_world_size()
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]
        if self.flat_gradients:
            if not grads:
                return
            # Concatenate all gradients into one contiguous buffer, communicate
            # it with a single all-reduce, then copy the results back
            # (_unflatten_dense_tensors returns new tensors, so copy_ into the
            # original .grad tensors).
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= world_size
            for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                grad.copy_(synced)
        else:
            for grad in grads:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad /= world_size