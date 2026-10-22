import torch

def cross_entropy(logits, targets):
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))

    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1), ).squeeze(-1)
    losses = log_sum_exp - target_logits
    return losses.mean()
