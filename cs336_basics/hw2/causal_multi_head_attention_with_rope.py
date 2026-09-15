import torch
import torch.nn as nn
from rope import RoPE

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
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(scores, dim = -1)

        return torch.matmul(attn_weights, V)

    def forward(self, x, wq, wk, wv, wo, token_positions):
        batch_size, seq_len, d_model = x.shape

        q = x @ wq.T
        k = x @ wk.T
        v = x @ wv.T

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = self.attention(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = out @ wo.T

        return out