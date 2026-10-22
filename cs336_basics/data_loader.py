'''
一长串 token ID
    ↓
随机选择 batch_size 个起点
    ↓
每个起点取输入，以及错开一位的目标
    ↓
返回 x、y，放到指定设备
'''

import numpy as np
import torch

def get_batch(data, batch_size, context_length, device):
    starts = np.random.randint(
        low=0,
        high=len(data) - context_length,
        size=batch_size,
    )

    x = np.stack([
        data[start:start + context_length]
        for start in starts
    ])

    y = np.stack([
        data[start + 1:start + context_length + 1]
        for start in starts
    ])

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return x, y

