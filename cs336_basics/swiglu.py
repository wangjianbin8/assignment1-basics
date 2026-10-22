import torch
from cs336_basics.linear import Linear

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(
            self,
            d_model,
            d_ff,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        gate = silu(self.w1(x))
        value = self.w3(x)

        return self.w2(gate * value)

