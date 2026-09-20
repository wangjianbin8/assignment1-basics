import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(
        self, 
        d_model: int,
        eps: float = 1e-5
    ):
        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(d_model)
        )

    def forward(self, x):
        rms = torch.mean(
            x ** 2,
            dim = -1,
            keepdim=True
        )

        rms = torch.sqrt(rms + self.eps)
        x_norm = x / rms
        # 加入可学习缩放参数
        #
        # weight:
        # (d_model,)
        #
        # 会自动broadcast
        return x_norm * self.weight