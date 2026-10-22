import torch


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model,
        eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.eps = eps

        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x):
        #先提高计算精度，尤其防止低精度输入在平方时溢出。
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = x.square().mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)

        result = (x / rms) * self.weight

        return result.to(in_dtype)