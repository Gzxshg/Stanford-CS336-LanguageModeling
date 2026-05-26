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
            in_features: Float[Tensor, " ... d_model"]
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
        self.w1_weight = w1_weight
        self.w2_weight = w2_weight
        self.w3_weight = w3_weight
        self.d_model = d_model
        self.d_ff = d_ff
        self.in_features = in_features