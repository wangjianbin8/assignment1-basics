import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                vocab_size,
                d_model
            )
        )

        nn.init.normal_(
            self.weight,
            mean=0,
            std=d_model ** -0.5
        )

    def forward(self, x):
        #输入的x维度是[batch_size, seq_len]
        return self.weight[x]