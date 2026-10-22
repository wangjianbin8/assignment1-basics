import torch
from einops import rearrange
from cs336_basics.linear import Linear
from cs336_basics.attention import scaled_dot_product_attention


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        device=None,
        dtype=None,
        rope=None,
    ):
        super().__init__()

        self.rope = rope

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, x):
        return rearrange(
            x, 
            "... t (h d) -> ... h t d",
            h=self.num_heads,
        )

    def _merge_heads(self, x):
        return rearrange(
            x,
            "... h t d -> ... t (h d)"
        )

    def forward(self, x, token_positions=None):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        seq_len = x.shape[-2]
        indices = torch.arange(seq_len, device=x.device)

        if self.rope is not None:
            if token_positions is None:
                token_positions = indices

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = indices[:, None] >= indices[None, :]

        head_outputs = scaled_dot_product_attention(q, k, v, mask=mask)

        merged = self._merge_heads(head_outputs)
        return self.o_proj(merged)

