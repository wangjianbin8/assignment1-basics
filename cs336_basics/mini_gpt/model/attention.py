import torch
import torch.nn as nn

from .linear import Linear
from .rope import RoPE

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len,
        theta
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (d_model % num_heads == 0)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        self.rope = RoPE(head_dim=self.head_dim, max_seq_len=max_seq_len, theta=theta)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        positions = torch.arange(seq_len, device = x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        Q = self.rope(Q, positions)
        K = self.rope(K, positions)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(mask == 1, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, self.d_model)

        out = self.out_proj(out)
        return out