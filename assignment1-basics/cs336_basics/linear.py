import torch
import torch.nn as nn
import math
from torch._higher_order_ops.out_dtype import out_dtype_fake_tensor_mode

class Linear(nn.Module):
    def __init__(self,in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight=nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std=math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A tensor of shape (..., in_features)
        Returns:
            A tensor of shape (..., out_features) computed as x @ weight.T
        """
        return x @ self.weight.transpose(-1,-2)