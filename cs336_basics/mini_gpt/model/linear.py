import torch
import torch.nn as nn

class Linear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        bias=None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features
                )
            )
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.weight,
            std=0.02
        )

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y += self.bias
        return y