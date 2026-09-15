import torch
from torch import nn

class RMSnorm(nn.Module):
    '''
    x_norm = x / sqrt(mean(x^2) + eps) * weight
    其中：
    '''
    def __init__(self, d_model, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = x.pow(2).mean(-1, keepdim=True)
        '''
        x.pow(2)：逐元素平方。
        .mean(-1, keepdim=True)：沿最后一维（d_model）求平均，保持维度。
        结果 variance 形状为 (batch_size, seq_len, 1)。
        '''
        x = x * torch.rsqrt(mean_square + self.eps)

        return (self.weight * x).to(input_dtype)

        '''
        self.weight * x：乘以可学习的缩放参数 weight。
        weight 形状 (d_model,)，会自动广播到 (batch_size, seq_len, d_model)。
        '''
        #对每个 token 的 d_model 维向量，每个维度都乘以一个独立的、可学习的缩放因子。