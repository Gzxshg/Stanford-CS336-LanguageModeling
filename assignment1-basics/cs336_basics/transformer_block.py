import torch
import torch.nn as nn
from .multihead_self_attention import MultiHeadSelfAttention
from .SwiGLU import SwiGLU
from .RMSNorm import RMSNorm
# ---------------------------------------------------------
# 核心 Transformer Block 实现
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    基于 Pre-Norm 架构的 Transformer Block
    根据说明，需接收 d_model, num_heads, d_ff 三个参数。
    """
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            d_ff: int,
            theta: float,
            max_seq_len: int = None,
            ):
        super().__init__()
        
        # 验证参数合法性
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        # 第一部分：多头自注意力子层 (MHA)
        self.ln1 = RMSNorm(d_model)
        # 替换为你自己实现的 MultiHeadSelfAttention
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads,
            max_seq_len=max_seq_len, 
            use_rope=True, 
            theta=theta) 

        # 第二部分：SwiGLU 前馈网络子层 (FFN)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(
            d_model=d_model, 
            d_ff=d_ff,
            w1_weight=None,
            w2_weight=None,
            w3_weight=None
            ) 

    def forward(self, x):
        """
        前向传播函数
        x 的形状通常为: (batch_size, sequence_length, d_model)
        """
        # 1. 第一个子层：Pre-norm -> Attention -> 残差连接
        # 对应公式: y = x + MultiHeadSelfAttention(RMSNorm(x))
        x = x + self.attn(self.ln1(x))
        
        # 2. 第二个子层：Pre-norm -> SwiGLU FFN -> 残差连接
        # 对应公式: z = y + SwiGLUFFN(RMSNorm(y))
        x = x + self.ffn(self.ln2(x))
        
        return x