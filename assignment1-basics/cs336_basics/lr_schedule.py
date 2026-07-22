import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    带线性 warmup 的余弦退火学习率 (LLaMA 使用的 schedule)
    - it < T_w:        线性升到 α_max
    - T_w <= it <= T_c: 余弦从 α_max 退火到 α_min
    - it > T_c:        恒定 α_min
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate
