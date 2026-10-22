import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")

        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1

                t = state["t"]
                m = state["m"]
                v = state["v"]

                adjusted_lr = (
                    lr * math.sqrt(1 - beta2 ** t)
                    / (1 - beta1 ** t)
                )
                #原地修改
                p.mul_(1 - lr * weight_decay)

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # m ← beta1 * m + (1 - beta1) * grad

                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # v ← beta2 * v + (1 - beta2) * grad²

                p.addcdiv_(
                    m, 
                    v.sqrt() + eps,
                    value=-adjusted_lr,
                )
        return loss