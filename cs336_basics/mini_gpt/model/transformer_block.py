import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import MultiHeadAttention
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta
    ):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x):
        norm_x = self.attn_norm(x)
        attn_out = self.attention(norm_x)
        x = x + attn_out

        norm_x = self.ffn_norm(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out

        return x