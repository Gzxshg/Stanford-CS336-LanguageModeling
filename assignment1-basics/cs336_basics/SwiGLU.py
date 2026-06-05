import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            w1_weight: Float[Tensor, " d_ff d_model"],
            w2_weight: Float[Tensor, " d_model d_ff"],
            w3_weight: Float[Tensor, " d_ff d_model"],
    ) -> Float[Tensor, " ... d_model"]:
        """
        Run SwiGLU on the input `in_features` with the given weights.

        Args:
        - d_model: The dimension of the input and output features.
        - d_ff: The dimension of the hidden layer in the feedforward network.
        - w1_weight: The weight matrix for the first linear transformation, of shape (d_ff, d_model).
        - w2_weight: The weight matrix for the second linear transformation, of shape (d_model, d_ff).
        - w3_weight: The weight matrix for the third linear transformation, of shape (d_ff, d_model).
        - in_features: The input features, of shape (..., d_model).

        Returns:
        - Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        SwiGLU on the `in_features`.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        if w1_weight is None:
            w1=torch.empty(d_ff, d_model)
            nn.init.xavier_uniform_(w1)
            self.w1_weight = nn.Parameter(w1)
        else:
            self.w1_weight = nn.Parameter(w1_weight.clone().detach())
        
        if w2_weight is None:
            w2=torch.empty(d_model, d_ff)
            nn.init.xavier_uniform_(w2)
            self.w2_weight = nn.Parameter(w2)
        else:
            self.w2_weight = nn.Parameter(w2_weight.clone().detach())

        if w3_weight is None:
            w3=torch.empty(d_ff, d_model)
            nn.init.xavier_uniform_(w3)
            self.w3_weight = nn.Parameter(w3)
        else:
            self.w3_weight = nn.Parameter(w3_weight.clone().detach())

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        if in_features.shape[-1] != self.d_model:
            raise ValueError(f"last dim of in_features ({in_features.shape[-1]}) != d_model ({self.d_model})")

        a = torch.matmul(in_features, self.w1_weight.T)   # (..., d_ff)
        b = torch.matmul(in_features, self.w3_weight.T)   # (..., d_ff)
        silu_a = a * torch.sigmoid(a)                     # SiLU via sigmoid
        gated = silu_a * b                                # (..., d_ff)
        out = torch.matmul(gated, self.w2_weight.T)       # (..., d_model)
        return out