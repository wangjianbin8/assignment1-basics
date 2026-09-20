import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(
        self,
        head_dim,
        max_seq_len,
        theta = 10000
    ):
        super().__init__()
        self.head_dim = head_dim

        freqs = 1.0 / (
            theta ** (
                torch.arange(
                    0, 
                    head_dim,
                    2
                ).float()
                / head_dim
            )
        )

        positions = torch.arange(max_seq_len)

        # position和frequency相乘
        #
        # positions:
        # (max_seq_len,)
        #
        # freqs:
        # (head_dim/2,)
        #
        # 输出:
        # (max_seq_len, head_dim/2)
        angles = torch.outer(
            positions,
            freqs
        )

        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x, positions):
        """
        x:
        (batch, heads, seq_len, head_dim)

        positions:
        (batch, seq_len)

        """

        # 根据当前位置取cos
        #
        # cos:
        # (batch, seq_len, head_dim/2)

        cos = self.cos[
            positions
        ]

        sin = self.cos[
            positions
        ]

        # 原:
        # (batch,seq,dim/2)
        #
        # 变:
        # (batch,1,seq,dim/2)
        #
        # 方便和x广播    
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        x_out = torch.stack([rotated_x1, rotated_x2], dim=-1)

        return x_out.flatten(-2)

