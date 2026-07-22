import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 1D token 序列中随机采样一个 batch
    x: 输入序列 (batch_size, context_length)
    y: 对应标签, 即 x 右移一位 (预测下一个 token)
    起始下标均匀采样自 [0, len(dataset) - context_length - 1]
    """
    n = len(dataset)
    starts = np.random.randint(0, n - context_length, size=(batch_size,))
    offsets = np.arange(context_length)
    x_idx = starts[:, None] + offsets
    y_idx = x_idx + 1
    x = torch.tensor(dataset[x_idx], dtype=torch.long, device=device)
    y = torch.tensor(dataset[y_idx], dtype=torch.long, device=device)
    return x, y
