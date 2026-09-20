import torch


class AdamW:

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        self.params = list(params)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # 用 param_groups 结构，兼容 PyTorch 优化器接口
        # 训练脚本可以写：
        #   for group in optimizer.param_groups:
        #       group["lr"] = new_lr
        self.param_groups = [
            {
                "params": self.params,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        ]

        # 一阶、二阶矩缓存。用 id(p) 作 key，避免 tensor 作 key 的问题
        self.m = {id(p): torch.zeros_like(p) for p in self.params}
        self.v = {id(p): torch.zeros_like(p) for p in self.params}

    def step(self):
        self.step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # 一阶矩
                self.m[id(p)] = beta1 * self.m[id(p)] + (1 - beta1) * grad
                # 二阶矩
                self.v[id(p)] = beta2 * self.v[id(p)] + (1 - beta2) * grad ** 2

                # 偏差校正
                m_hat = self.m[id(p)] / (1 - beta1 ** self.step_count)
                v_hat = self.v[id(p)] / (1 - beta2 ** self.step_count)

                # 更新方向
                update = m_hat / (torch.sqrt(v_hat) + eps)

                # 解耦 weight decay
                p.data -= lr * weight_decay * p.data
                # 梯度更新
                p.data -= lr * update

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "m": self.m,
            "v": self.v,
            "param_groups": self.param_groups,
        }