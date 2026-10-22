import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
            self,
            theta,
            d_k,
            max_seq_len,
            device=None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k 必须是偶数")

        exponents = (
            torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
            / d_k
        )

        inv_freq = theta ** (-exponents)

        #位置: 0, 1, ..., max_seq_len - 1
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer(
            "cos_cache",
            torch.cos(angles),
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            torch.sin(angles),
            persistent=False,
        )


    def forward(self, x, token_positions):
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 必要时补充广播维度，例如注意力头维度。
        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        result = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(start_dim=-2)

        return result.to(x.dtype)

