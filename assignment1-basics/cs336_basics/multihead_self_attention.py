import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
from .scaled_dot_product_attention import scaled_dot_product_attention

def run_Multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # 获取序列长度和每个头的维度
    seq_len = in_features.shape[-2]
    d_head = d_model // num_heads

    # 1. 单次矩阵乘法进行 Q, K, V 投影
    # in_features: (..., seq_len, d_model)
    # proj_weight: (d_model, d_model)
    # 输出 Q, K, V: (..., seq_len, d_model)
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # 2. 重排张量，将 d_model 拆分为 num_heads 和 d_head
    # (..., seq_len, d_model) -> (..., seq_len, num_heads, d_head)
    # 交换维度 -> (..., num_heads, seq_len, d_head)
    Q = Q.view(*in_features.shape[:-1], num_heads, d_head).transpose(-3, -2)
    K = K.view(*in_features.shape[:-1], num_heads, d_head).transpose(-3, -2)
    V = V.view(*in_features.shape[:-1], num_heads, d_head).transpose(-3, -2)

    # 3. 计算缩放点积注意力
    # 使用 PyTorch 2.0+ 的 F.scaled_dot_product_attention
    # 它在底层会自动选择 FlashAttention 或 MemEfficientAttention，性能极佳
    # 注意：此处未使用 is_causal=True，因为原函数签名和文档说明未要求实现因果掩码
    casual_mask=torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attn_output = scaled_dot_product_attention(Q, K, V, casual_mask)

    # 4. 拼接所有头的输出
    # (..., num_heads, seq_len, d_head) -> (..., seq_len, num_heads, d_head)
    # -> (..., seq_len, d_model)
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*in_features.shape[:-1], d_model)

    # 5. 输出投影
    # (..., seq_len, d_model) @ (d_model, d_model).T -> (..., seq_len, d_model)
    out_features = attn_output @ o_proj_weight.T

    return out_features


class MultiheadSelfAttention(nn.Module):
    def __init__(
            d_model: int,
            num_heads: int,
            q_proj_weight: Float[Tensor, " d_model d_model"],
            k_proj_weight: Float[Tensor, " d_model d_model"],
            v_proj_weight: Float[Tensor, " d_model d_model"],
            o_proj_weight: Float[Tensor, " d_model d_model"],
            in_features: Float[Tensor, " ... sequence_length d_model"]):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 初始化 Q, K, V, O 投影权重
        # 注意：nn.Linear 的权重形状是，所以我们在 forward 中不需要转置
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, in_features: Tensor) -> Tensor:
        """
        Args:
            in_features: (..., sequence_length, d_model)

        Returns:
            output: (..., sequence_length, d_model)
        """
        # 提取权重矩阵，复用上面的优化函数
        # nn.Linear.weight 的形状是，转置后与函数要求的 一致
        return run_Multihead_self_attention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            o_proj_weight=self.o_proj.weight,
            in_features=in_features,
        )
