import torch

@torch.no_grad()
def generate(
    model,
    token_ids,
    max_new_tokens,
    context_length,
    eos_token_id=None,
    temperature=1.0,
    top_p=1.0,
):
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须大于 0 且不超过 1")
    if temperature <= 0:
        raise ValueError("temperature 必须大于0")
    # 先支持一次生成一条序列。
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("token_ids 的形状应为 (1, 序列长度)")

    if token_ids.shape[1] == 0:
        raise ValueError("输入至少需要一个 token")

    # 记录模型原来的模式：
    # True 表示训练模式，False 表示评估模式。
    was_training = model.training

    # 切换到评估模式，准备生成。
    model.eval()

    # 用 generated 保存“原始输入 + 后续生成的 token”。
    # 最开始是 [[0, 1, 2]]。
    generated = token_ids

    # 最多生成 max_new_tokens 个新 token。
    # _ 表示我们不需要使用这个循环变量。
    for _ in range(max_new_tokens):

        # 从完整序列中，取最后 context_length 个 token。
        # 第一个 : 表示保留所有行。
        # -context_length: 表示取末尾这一段。
        # 序列不足 context_length 时，就取全部。
        model_input = generated[:, -context_length:]

        # 把当前输入交给模型，得到每个位置对下一个 token 的分数。
        # 假设输入形状是 (1, 3)，词表大小为 16：
        # logits 的形状就是 (1, 3, 16)。
        logits = model(model_input)

        # 我们要接在整个输入的后面，所以只取最后一个位置。
        # 例如根据 [0, 1, 2]，预测接下来的 token。
        # 形状从 (1, 3, 16) 变成 (1, 16)。
        next_logits = logits[:, -1, :]

        # 先根据温度调整概率。
        probs = torch.softmax(
            next_logits / temperature,
            dim=-1,
        )

        if top_p < 1.0:
            # 按概率从大到小排序，同时保留对应 token ID。
            sorted_probs, sorted_indices = torch.sort(
                probs,
                dim=-1,
                descending=True,
            )

            # 计算累计概率。
            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim=-1,
            )

            # 找到删除位置。
            remove_mask = cumulative_probs >= top_p

            # 右移，保留首次达到阈值的候选。
            remove_mask[..., 1:] = (
                remove_mask[..., :-1].clone()
            )
            remove_mask[..., 0] = False

            # 删除的候选概率设为 0。
            filtered_probs = sorted_probs.masked_fill(
                remove_mask,
                0.0,
            )

            # 重新归一化。
            filtered_probs = (
                filtered_probs
                / filtered_probs.sum(dim=-1, keepdim=True)
            )

            # 抽取排序位置，再找回真正的 token ID。
            sampled_position = torch.multinomial(
                filtered_probs,
                num_samples=1,
            )

            next_token = sorted_indices.gather(
                dim=-1,
                index=sampled_position,
            )

        else:
            # top_p=1 时，直接从原概率分布采样。
            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

        # 把抽到的 token 接到序列末尾。
        # dim=-1 表示沿最后一个维度，也就是序列长度方向拼接。
        # [[0, 1, 2]] 和 [[3]] → [[0, 1, 2, 3]]
        generated = torch.cat([generated, next_token], dim=-1)

        # 如果设置了结束符编号，就检查刚生成的 token。
        if eos_token_id is not None:

            # .item() 把只包含一个元素的张量转成 Python 数字。
            # 例如 tensor([[3]]).item() 得到 3。
            if next_token.item() == eos_token_id:

                # 如果生成了结束符，提前结束循环。
                break

    # 恢复生成之前的模式。
    # 原来是训练模式就恢复训练模式；
    # 原来是评估模式就保持评估模式。
    model.train(was_training)

    return generated


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    context_length,
    temperature=1.0,
    top_p=1.0,
    eos_token_id=None,
):
    # 1. 文本转成 token ID 列表。
    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        raise ValueError("prompt 编码后至少需要一个 token")

    # 2. 找到模型所在的设备。
    device = next(model.parameters()).device

    # 3. 转成张量，添加批次维度。
    token_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    # 4. 调用已经写好的生成函数。
    result = generate(
        model=model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )

    # 5. 取出唯一一条序列，转成 Python 列表。
    result_ids = result[0].tolist()

    # 6. token ID 转回文本。
    return tokenizer.decode(result_ids)