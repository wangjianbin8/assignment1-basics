import torch

@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm):
    params = [p for p in parameters if p.grad is not None]

    if not params:
        return

    total_squared_norm = torch.stack([
        p.grad.square().sum() for p in params
    ]).sum()
    #所有梯度元素的平方和
    
    total_norm = total_squared_norm.sqrt()
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for p in params:
            p.grad.mul_(scale)