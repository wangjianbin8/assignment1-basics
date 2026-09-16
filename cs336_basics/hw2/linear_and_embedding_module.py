import torch
import torch.nn as nn

"""在讲义中，线性层在最后一步将d_model维度映射到vocab_size维度"""
class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))

        std = 2 / (self.in_features + self.out_features) ** 0.5

        torch.nn.init.trunc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)
        # 用截断正态分布初始化 W
        # 均值默认为 0，标准差为 std
        # 截断范围是 [a, b]，即 [-3*std, 3*std]
        # 这样避免初始权重出现极端值

    def forward(self, x: torch.Tensor):
        return x @ self.W.T

"""在讲义中，embedding层在第一步将token_ids映射到d_model维度"""
class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, d_model, device: torch.device | None = None,  dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.embedding_matrix = nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=self.device, dtype=self.dtype))
        std = 1
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.Tensor):
        return self.embedding_matrix[token_ids]
        # 例如 token_ids 形状 (2, 3)，则输出形状 (2, 3, embedding_dim)