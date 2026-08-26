"""Optimizer state sharding -- a simplified ZeRO stage 1 (ZeRO-DP P_os).

Each rank keeps a full replica of the parameters and gradients (data-parallel
training is unchanged), but the *optimizer* is sharded: every parameter is
assigned to exactly one owning rank, and only the owner keeps optimizer state
(e.g. AdamW's two moment buffers) for it. The owning rank's inner optimizer is
the only one that updates the parameter; after each step every rank broadcasts
its (freshly updated) shard so all replicas converge back to identical full
parameters.

Sharding policy: parameters are assigned greedily in arrival order to the rank
with the smallest total number of assigned elements, which balances shard sizes
across ranks while staying deterministic (all ranks run the same assignment over
the same parameter list, so they agree on ownership without communicating).

The post-step synchronization is done per owning rank with one flattened
broadcast per rank (torch._utils._flatten_dense_tensors), not one broadcast per
parameter, to avoid hundreds of tiny collectives per step.

Note: `add_param_group` does the assignment work because the
torch.optim.Optimizer super-class constructor calls it for the initial groups,
and training code may call it later (e.g. gradually unfreezing layers) -- new
parameters get owners either way.
"""

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._owner: dict = {}  # param -> owning rank
        self._shard_sizes = [0] * self.world_size
        self._inner: torch.optim.Optimizer | None = None
        self._pending_inner_groups: list[dict] = []
        # The super-class constructor calls self.add_param_group() for the
        # initial groups, which performs the sharding assignment.
        super().__init__(params, {})
        self._build_inner()

    def add_param_group(self, param_group: dict) -> None:
        params = list(param_group["params"])
        for p in params:
            if p not in self._owner:
                owner = min(range(self.world_size), key=lambda r: self._shard_sizes[r])
                self._owner[p] = owner
                self._shard_sizes[owner] += p.numel()
        super().add_param_group(param_group)
        # Forward only this rank's shard to the wrapped optimizer, keeping the
        # rest of the group spec (e.g. per-group hyperparameters) intact.
        owned = [p for p in params if self._owner[p] == self.rank]
        if not owned:
            return
        spec = {k: v for k, v in param_group.items() if k != "params"}
        spec["params"] = owned
        if self._inner is None:
            self._pending_inner_groups.append(spec)
        else:
            self._inner.add_param_group(spec)

    def _build_inner(self) -> None:
        if self._pending_inner_groups:
            self._inner = self.optimizer_cls(self._pending_inner_groups, **self.optimizer_kwargs)
            self._pending_inner_groups = []

    def step(self, closure=None, **kwargs):
        # The inner optimizer updates only this rank's shard; its states match a
        # non-sharded optimizer's exactly because every rank's gradients are
        # already synchronized (all-reduced) before the step.
        loss = self._inner.step(closure=closure, **kwargs) if self._inner is not None else None
        self._broadcast_updated_params()
        return loss

    def _broadcast_updated_params(self) -> None:
        for r in range(self.world_size):
            owned = [p for group in self.param_groups for p in group["params"] if self._owner[p] == r]
            if not owned:
                continue
            # One flat buffer per owner; non-owners' buffers are overwritten by
            # the broadcast, then copied back into their parameter replicas.
            data = [p.data for p in owned]
            flat = torch._utils._flatten_dense_tensors(data)
            dist.broadcast(flat, src=r)
            for p, synced in zip(owned, torch._utils._unflatten_dense_tensors(flat, data)):
                p.data.copy_(synced)
