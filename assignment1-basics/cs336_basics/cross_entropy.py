import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    """
    平均交叉熵损失: -log softmax(logits)[target] 的 batch 均值
    数值稳定版本: 先减去每行最大值再算 logsumexp
    (利用平移不变性: logsumexp(x-m) = logsumexp(x) - m)
    """
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_logits
    return loss.mean()