import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    按作业 PDF Algorithm 1 实现的 AdamW
    与 PyTorch 内置 AdamW 同为解耦权重衰减, 偏差修正体现在 alpha_t 上
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # 惰性初始化状态: 步数 t 与一阶/二阶矩 (与参数同形状)
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # 带偏差修正的学习率: α_t = α·√(1-β2^t) / (1-β1^t)
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # 解耦权重衰减: θ ← θ - α·λ·θ
                p.mul_(1 - lr * weight_decay)

                # 一阶/二阶矩更新
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 矩修正后的参数更新 (v.sqrt() 生成新张量, 不会污染 v)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

        return loss
