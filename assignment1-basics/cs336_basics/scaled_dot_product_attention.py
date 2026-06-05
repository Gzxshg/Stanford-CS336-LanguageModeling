import torch
import math

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    计算缩放点积注意力。
    
    参数:
    q: Query 张量，形状为 (batch_size, ..., seq_len, d_k)
    k: Key 张量，形状为 (batch_size, ..., seq_len, d_k)
    v: Value 张量，形状为 (batch_size, ..., seq_len, d_v)
    mask: 可选的布尔掩码，形状为 (seq_len, seq_len)。True 表示允许关注，False 表示屏蔽。
    
    返回:
    注意力计算后的输出张量，形状为 (batch_size, ..., seq_len, d_v)
    """
    # 提取 d_k 用于后续的缩放操作
    d_k = q.size(-1)
    
    # 1. 计算 Q 和 K^T 的点积打分 (Scores)
    # k 需要在最后两个维度上进行转置: (..., seq_len, d_k) 变成 (..., d_k, seq_len)
    # matmul 会自动在前面的 batch 维度上进行批量矩阵乘法
    # scores 的形状将会是: (batch_size, ..., seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    # 2. 进行缩放 (Scale)
    scores = scores / math.sqrt(d_k)
    
    # 3. 应用掩码 (Masking)
    if mask is not None:
        # 使用 masked_fill_ 方法，将 mask 中为 False 的位置填充为负无穷
        # mask 的形状是 (seq_len, seq_len)，PyTorch 会自动将其广播(broadcast)到 scores 的前面所有维度
        scores = scores.masked_fill(mask == False, float('-inf'))
        
    # 4. 计算 Softmax 得到注意力权重分布
    # (注：这里使用了原生自带的 softmax。如果你的作业框架要求强制调用上一题手写的 softmax，
    #  可以将此行替换为你引入的自定义 softmax 函数: custom_softmax(scores, dim=-1) )
    attn_probs = torch.softmax(scores, dim=-1)
    
    # 5. 权重分布乘以 Value 张量
    # attn_probs: (batch_size, ..., seq_len, seq_len)
    # v: (batch_size, ..., seq_len, d_v)
    # 结果 output 形状为: (batch_size, ..., seq_len, d_v)
    output = torch.matmul(attn_probs, v)
    
    return output