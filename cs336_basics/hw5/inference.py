import torch

def top_p_sampling(probabilities, top_p=0.9):
    """
    Top-p 核采样（Nucleus Sampling）
    作用：从模型输出的概率分布中，只保留累积概率达到 top_p 的最小候选集合，
         然后在这个集合内重新归一化并随机采样一个 token。
    参数：
        probabilities: 形状通常为 [batch_size, vocab_size] 或 [vocab_size]，
                       表示每个 token 的预测概率，且最后一维已经过 softmax，和为 1。
        top_p: 累积概率阈值，默认 0.9。例如 top_p=0.9 表示保留概率质量累计到 90% 的 token。
    返回：
        next_token_idx: 采样得到的下一个 token 在原始词表中的索引，形状通常为 [batch_size, 1]。
    """
    sort_probabilities, idx = torch.sort(probabilities, dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sort_probabilities, dim=-1)
    threshold = top_p

    mask = cumulative_probabilities > threshold
    sort_probabilities[mask] = 0

    sort_probabilities.div_(sort_probabilities.sum(dim=-1, keepdim=True))

    next_token_idx = torch.multinomial(sort_probabilities, 1)
    next_token_idx = torch.gather(idx, dim=-1, index=next_token_idx)

    return next_token_idx

def temperature_scaling(logits, temperature=1.0):
    """
    温度缩放（Temperature Scaling）
    作用：通过除以温度参数来调整 softmax 的分布锐度。
         temperature < 1.0 会使分布更尖锐（更确定）；
         temperature > 1.0 会使分布更平坦（更随机）。
    参数：
        logits: 模型输出的原始分数，形状通常为 [batch_size, seq_len, vocab_size]。
        temperature: 温度系数，默认 1.0 表示不缩放。
    返回：
        probabilities: 最后一个时间步的 token 概率分布，形状为 [batch_size, vocab_size]。
    """
    # 取最后一个时间步的 logits：logits[:, -1, :] 表示所有 batch 的最后一个位置的所有词表分数。
    # 然后除以 temperature，再沿最后一维做 softmax，得到概率分布。
    # 这里 dim=-1 表示在词表维度上做 softmax。

    probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    return probabilities

def decode_token(input_tokens, model, max_tokens_to_generate, top_p=0.9, temperature=1.0):
    model.eval()
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_tokens_to_generate):
            if input_tokens[0, -1] == "<endoftext>":
                break

            logits = model(input_tokens)
            probabilities = temperature_scaling(logits, temperature)
            next_token_idx = top_p_sampling(probabilities, top_p)
            # torch.cat 沿着最后一维（序列维度）拼接，input_tokens 形状变为 [batch_size, seq_len+1]。
            input_tokens = torch.cat([input_tokens, next_token_idx], dim=-1)

    return input_tokens