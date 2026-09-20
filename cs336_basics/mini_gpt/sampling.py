import torch

def top_k_sampling(
    logits,
    top_k = 50
):
    values, indices = torch.topk(
        logits,
        top_k,
        dim=-1
    )

    new_logits = torch.full_like(
        logits,
        float("-inf")
    )

    new_logits.scatter_(
        -1,
        indices,
        values
    )

    return new_logits


def top_p_sampling(
    logits,
    top_p=0.9
):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative_probs > top_p

    # 第一个超过p的位置保留
    mask[...,1:] = mask[...,:-1].clone()
    mask[...,0] = False

    sorted_probs[mask] = 0


    # 重新归一化

    sorted_probs = sorted_probs / sorted_probs.sum(
        dim=-1,
        keepdim=True
    )


    return sorted_probs, sorted_indices