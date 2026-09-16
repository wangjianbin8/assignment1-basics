import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            Q,
            K,
            V,
            mask
    ):
        d_k = Q.shape[-1]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim = -1)

        return torch.matmul(attn_weights, V)