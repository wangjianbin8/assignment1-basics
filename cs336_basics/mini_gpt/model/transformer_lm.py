import torch
import torch.nn as nn

from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        num_layers,
        num_heads,
        d_ff, 
        theta = 10000,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=theta
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

        