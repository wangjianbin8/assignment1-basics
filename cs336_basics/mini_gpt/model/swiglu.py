import torch
import torch.nn as nn

from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff
    ):
        super().__init__()

        # 第一条路径
        #
        # x -> W1
        #
        # 产生候选信息
        self.w1 = Linear(
            d_model,
            d_ff
        )


        # 第二条路径
        #
        # x -> W3
        #
        # 产生gate
        self.w3 = Linear(
            d_model,
            d_ff
        )

        self.w2 = Linear(
            d_ff,
            d_model
        )

    def forward(self, x):
        # 第一条路径
        x1 = self.w1(x)


        # SiLU激活
        x1 = torch.nn.functional.silu(
            x1
        )


        # 第二条路径
        x2 = self.w3(x)


        # 门控相乘
        x = x1 * x2


        # 投影回d_model
        x = self.w2(x)


        return x