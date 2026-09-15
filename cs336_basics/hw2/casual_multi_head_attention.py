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

例如：

B = 2
T = 4
D = 8
H = 2
d = 4

那么整个过程就是：

[B,T,D]
   ↓
[B,T,D] Q/K/V
   ↓
[B,T,H,d]
   ↓ transpose
[B,H,T,d]
   ↓
QKᵀ
[B,H,T,T]
   ↓
softmax
[B,H,T,T]
   ↓ × V
[B,H,T,d]
   ↓ transpose
[B,T,H,d]
   ↓ reshape
[B,T,D]

这条线你一定要记住。
'''