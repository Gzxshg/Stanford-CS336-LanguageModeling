import torch
import torch.nn as nn

from .embedding import Embedding
from .linear import Linear
from .RMSNorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """
    完整的 Transformer 语言模型 (Pre-Norm + RoPE)
    数据流: token ids -> Embedding -> N x TransformerBlock -> 最终 RMSNorm -> LM Head -> logits
    属性命名与参考实现的 state_dict 键一致, 可直接 load_state_dict

    消融开关(默认值 = 原始行为): use_layer_norm / pre_norm / use_rope / use_swiglu,
    含义见 TransformerBlock。
    """

    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None,
            dtype=None,
            use_layer_norm: bool = True,
            pre_norm: bool = True,
            use_rope: bool = True,
            use_swiglu: bool = True,
            ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        # 词嵌入层: token id -> d_model 维向量
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
        )

        # 堆叠 num_layers 个 Transformer block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                use_layer_norm=use_layer_norm,
                pre_norm=pre_norm,
                use_rope=use_rope,
                use_swiglu=use_swiglu,
            )
            for _ in range(num_layers)
        ])

        # 输出前的最终归一化
        self.ln_final = (
            RMSNorm(d_model, device=device, dtype=dtype)
            if use_layer_norm else nn.Identity()
        )

        # 输出投影: d_model -> vocab_size, 返回 raw logits (不做 softmax)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: 形状 (batch_size, sequence_length) 的整数 token id
        Returns:
            形状 (batch_size, sequence_length, vocab_size) 的 raw logits
        """
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
