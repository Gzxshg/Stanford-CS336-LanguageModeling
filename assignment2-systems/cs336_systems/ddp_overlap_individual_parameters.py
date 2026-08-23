"""DDP with gradient communication overlapped with the backward pass (Section 5.3.2).

Same contract as the naive DDP (cs336_systems/naive_ddp.py): broadcast the initial
state from rank 0 at construction, and finish_gradient_synchronization() must be
called after loss.backward() and before optimizer.step().

The difference is *when* communication happens. At construction we register a
post-accumulate-grad hook on every parameter that requires gradients. The hook
fires as soon as that parameter's gradient has been fully accumulated during the
backward pass, and immediately issues an *asynchronous* all-reduce
(async_op=True), so the communication runs concurrently with the backward
computation of the remaining (earlier) layers. finish_gradient_synchronization()
then only has to wait on the outstanding handles and average the gradients.

Correctness notes:
  - Autograd runs a single AccumulateGrad node per leaf parameter per backward
    pass, so the hook fires exactly once per parameter per backward -- even for
    tied weights, whose two gradient contributions are summed by autograd before
    the hook fires.
  - All ranks execute the same graph, so hooks fire in the same order on every
    rank and the asynchronous collectives match up.
  - Like the naive version, gradients are summed by all-reduce and divided by the
    world size afterwards (same numerics, just overlapped).
  - Limitation: multiple backward passes before finish_gradient_synchronization()
    (micro-batch gradient accumulation) would fire the hooks more than once per
    step and communicate partial gradients; not supported, same as PyTorch DDP
    without no_sync.
"""

import torch
import torch.distributed as dist


class DDPOverlap(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._comm_handles: list = []
        self._sync_initial_state()
        self._register_hooks()

    def _sync_initial_state(self) -> None:
        # Broadcast parameters and buffers from rank 0 (state_dict covers both,
        # and its tensors share storage with the module, so this updates the
        # module in place).
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def _register_hooks(self) -> None:
        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            # Bind p=param at definition time: the loop variable would otherwise
            # be captured late-bound and every hook would see the last parameter.
            def hook(p, _param=param):
                handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                self._comm_handles.append((handle, p))

            param.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        world_size = dist.get_world_size()
        for handle, param in self._comm_handles:
            handle.wait()
            param.grad /= world_size
        self._comm_handles.clear()
