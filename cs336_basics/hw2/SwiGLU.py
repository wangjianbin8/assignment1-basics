import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    """
    SwiGLU 是激活函数，它通过将输入乘以sigmoid函数，然后乘以一个线性变换来得到输出。
    公式是：
    out = w2(w1(x) * sigmoid(w1(x)) * w3(x))

    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))