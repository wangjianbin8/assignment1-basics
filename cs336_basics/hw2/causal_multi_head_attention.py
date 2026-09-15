import torch
import torch.nn as nn
import math
import torch.nn.functional as F

'''
以后看到多头注意力，先写这几个：

B = batch_size
T = seq_len
D = d_model
H = n_heads
d = head_dim = D / H

x
[B,T,D]
  │
  ├──────── Wq ────────→ Q [B,T,D]
  │
  ├──────── Wk ────────→ K [B,T,D]
  │
  └──────── Wv ────────→ V [B,T,D]
                            │
                            ↓
                      view(B,T,H,d)
                            │
                            ↓
                       [B,T,H,d]
                            │
                     transpose(1,2)
                            │
                            ↓
                       [B,H,T,d]
                            │
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  Q                   K
                  │                   │
                  └────── QKᵀ ───────┘
                            │
                            ↓
                       [B,H,T,T]
                            │
                          Mask
                            │
                         Softmax
                            │
                            ↓
                       [B,H,T,T]
                            │
                            × V
                            ↓
                       [B,H,T,d]
                            │
                     transpose(1,2)
                            │
                            ↓
                       [B,T,H,d]
                            │
                       reshape
                            │
                            ↓
                         [B,T,D]
                            │
                           Wo
                            │
                            ↓
                         [B,T,D]
'''

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def attention(self, Q, K, V, mask):
        d_k = Q.shape[-1]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        '''

        所以 softmax 的意思是：

        对于一个 Query，把它对所有 Key 的关注程度转换成概率，并且加起来等于 1
        '''

        return torch.matmul(attn_weights, V)

    def forward(self, x, wq, wk, wv, wo):
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

        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = self.attention(q, k, v, mask)

        out = out.transpose(1, 2)
        #拼回来
        out = out.contiguous().view(batch_size, seq_len, d_model)
        #view的时候，要求逻辑与物理存储一致
        #transpose之后，只是改变了逻辑上的顺序，实际在内存中顺序没有改变
        #即修改了Tensor对内存的访问方式（stride）
        #矩阵相乘也是看逻辑上的排列的（能够通过逻辑顺序找到数据）
        #contiguous() 会复制一份数据，使内存排列和逻辑形状一致，之后 view 就可以自由变形。
        out = out @ wo.T                     
        return out