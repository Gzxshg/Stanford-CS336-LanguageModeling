import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .linear import Linear
from .silu import SiLU


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            w1_weight: Float[Tensor, " d_ff d_model"] = None,
            w2_weight: Float[Tensor, " d_model d_ff"] = None,
            w3_weight: Float[Tensor, " d_ff d_model"] = None,
    ):
        """
        SwiGLU 前馈网络: w2(SiLU(w1(x)) * w3(x))

        Args:
        - d_model: The dimension of the input and output features.
        - d_ff: The dimension of the hidden layer in the feedforward network.
        - w1_weight / w2_weight / w3_weight: 可选的外部权重, 传入时覆盖内部初始化。
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 三个 Linear 子模块, state_dict 键为 w1.weight / w2.weight / w3.weight
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

        if w1_weight is not None:
            self.w1.weight = nn.Parameter(w1_weight.clone().detach())
        if w2_weight is not None:
            self.w2.weight = nn.Parameter(w2_weight.clone().detach())
        if w3_weight is not None:
            self.w3.weight = nn.Parameter(w3_weight.clone().detach())

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        if in_features.shape[-1] != self.d_model:
            raise ValueError(f"last dim of in_features ({in_features.shape[-1]}) != d_model ({self.d_model})")

        a = self.w1(in_features)                     # (..., d_ff)
        b = self.w3(in_features)                     # (..., d_ff)
        silu_a_embodied = SiLU()                        
        silu_a=silu_a_embodied(a)                   # (..., d_ff)
                        # SiLU via sigmoid
        return self.w2(silu_a * b)                   # (..., d_model)
