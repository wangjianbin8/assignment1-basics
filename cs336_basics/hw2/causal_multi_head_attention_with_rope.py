import torch
import torch.nn as nn
from .rope import RoPE
from einops import rearrange

class CausalMultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, theta, device=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.theta = theta
        self.head_dim = d_model // n_heads
        self.rope = RoPE(theta, self.head_dim, seq_len, device)

    def attention(self, Q, K, V, mask):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim = -1)

        return torch.matmul(attn_weights, V)

    def forward(self, x, wq, wk, wv, wo, token_positions):

        seq_len = x.shape[-2]

        # Q K V
        q = x @ wq.T
        k = x @ wk.T
        v = x @ wv.T

        # [ ..., T, D ]
        # ↓
        # [ ..., H, T, d_head ]
        q = rearrange(
            q,
            "... seq (head d_head) -> ... head seq d_head",
            head=self.n_heads,
        )

        k = rearrange(
            k,
            "... seq (head d_head) -> ... head seq d_head",
            head=self.n_heads,
        )

        v = rearrange(
            v,
            "... seq (head d_head) -> ... head seq d_head",
            head=self.n_heads,
        )

        # RoPE
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # causal mask
        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        # Attention
        out = self.attention(q, k, v, mask)

        # [ ..., H, T, d_head ]
        # ↓
        # [ ..., T, D ]
        out = rearrange(
            out,
            "... head seq d_head -> ... seq (head d_head)"
        )

        # output projection
        out = out @ wo.T

        return out