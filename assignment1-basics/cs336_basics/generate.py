import torch

from .softmax import softmax


def _sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> int:
    """
    从最后一个位置的 logits 采样下一个 token id
    temperature <= 0 时退化为 greedy (argmax)
    top_p < 1.0 时做 nucleus 截断: 保留累计概率刚跨过 p 的最小集合
    """
    if temperature <= 0:
        return int(logits.argmax().item())

    probs = softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # exclusive cumsum >= p 的位置全部屏蔽 (保留"刚好跨过"的那一个)
        mask = (cum_probs - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_pos = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[next_pos].item())

    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cuda",
    eot_token: str = "<|endoftext|>",
) -> str:
    """
    自回归采样: prompt 编码 -> 反复预测下一个 token -> 直到 <|endoftext|> 或达到 max_new_tokens
    输入超过 context_length 时从左侧截断 (只保留最近 context_length 个 token)
    返回 prompt + 生成内容的完整文本
    """
    model.eval()
    ids = list(tokenizer.encode(prompt))

    eot_ids = tokenizer.encode(eot_token)
    eot_id = eot_ids[0] if len(eot_ids) == 1 else None

    context_length = model.context_length

    for _ in range(max_new_tokens):
        window = ids[-context_length:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1]  # (vocab_size,) 只有最后一个位置预测下一个 token
        next_id = _sample_next(logits, temperature, top_p)
        ids.append(next_id)
        if eot_id is not None and next_id == eot_id:
            ids.pop()  # 不把 <|endoftext|> 本身放进输出
            break

    return tokenizer.decode(ids)
