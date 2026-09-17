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
        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
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
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # ==========================
                # 1. 更新一阶矩 m
                # ==========================

                # m = beta1*m + (1-beta1)*g

                exp_avg.mul_(beta1)

                exp_avg.add_(
                    grad,
                    alpha = 1 - beta1
                )

                # ==========================
                # 2. 更新二阶矩 v
                # ==========================

                # v = beta2*v + (1-beta2)*g^2

                exp_avg_sq.mul_(beta2)

                exp_avg_sq.addcmul_(
                    grad,
                    grad,
                    value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                denom = torch.sqrt(v_hat) + eps
                p.addcdiv_(m_hat, denom, value=-lr)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
            



