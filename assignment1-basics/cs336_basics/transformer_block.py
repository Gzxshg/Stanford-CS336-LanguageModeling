import torch
import torch.nn as nn
from .multihead_self_attention import MultiHeadSelfAttention
from .SwiGLU import SwiGLU
from .RMSNorm import RMSNorm
from .linear import Linear
from .silu import SiLU


class PlainFFN(nn.Module):
    """普通 SiLU 前馈网络: w2(SiLU(w1(x))), no_swiglu 消融用。

    隐藏维取 4*d_model, 与 SwiGLU(d_ff=8/3*d_model, 三个矩阵) 参数量持平:
    2 * 4d^2 = 8d^2  vs  3 * (8/3)d^2 = 8d^2
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.w1 = Linear(d_model, 4 * d_model)
        self.act = SiLU()
        self.w2 = Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


# ---------------------------------------------------------
# 核心 Transformer Block 实现
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    基于 Pre-Norm 架构的 Transformer Block
    根据说明，需接收 d_model, num_heads, d_ff 三个参数。

    消融开关(默认值 = 原始行为, 不影响既有测试):
    - use_layer_norm=False: 去掉全部 RMSNorm
    - pre_norm=False:       改为 Post-Norm(残差后归一化)
    - use_rope=False:       注意力不施加 RoPE
    - use_swiglu=False:     FFN 换成普通 SiLU MLP(隐藏维 4*d_model)
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            max_seq_len: int = None,
            use_layer_norm: bool = True,
            pre_norm: bool = True,
            use_rope: bool = True,
            use_swiglu: bool = True,
            ):
        super().__init__()

        # 验证参数合法性
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.pre_norm = pre_norm

        # 第一部分：多头自注意力子层 (MHA)
        self.ln1 = RMSNorm(d_model) if use_layer_norm else nn.Identity()
        # 替换为你自己实现的 MultiHeadSelfAttention
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            theta=theta)

        # 第二部分：前馈网络子层 (FFN)
        self.ln2 = RMSNorm(d_model) if use_layer_norm else nn.Identity()
        if use_swiglu:
            self.ffn = SwiGLU(
                d_model=d_model,
                d_ff=d_ff,
                w1_weight=None,
                w2_weight=None,
                w3_weight=None
                )
        else:
            self.ffn = PlainFFN(d_model)

    def forward(self, x):
        """
        前向传播函数
        x 的形状通常为: (batch_size, sequence_length, d_model)
        """
        if self.pre_norm:
            # Pre-Norm: y = x + Sublayer(RMSNorm(y))
            # use_layer_norm=False 时 ln 为 Identity, 自然退化为 y = x + Sublayer(y)
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            # Post-Norm: y = RMSNorm(x + Sublayer(y))
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))

        return x
