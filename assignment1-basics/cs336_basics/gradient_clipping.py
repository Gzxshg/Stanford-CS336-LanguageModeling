import torch


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    全局梯度裁剪: 把所有参数的梯度拼起来的总 L2 范数裁剪到 max_l2_norm
    就地修改各参数的 .grad; 与 torch.nn.utils.clip_grad_norm_ 行为一致
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return

    total_norm = torch.norm(torch.stack([p.grad.norm() for p in params]))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.mul_(clip_coef)
