import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        # 计算频率：freqs = 1 / (theta^(2i/d_k))，其中 i = 0,1,...,d_k/2-1
        
        # 生成位置索引：0, 1, 2, ..., max_seq_len-1
        positions = torch.arange(self.max_seq_len)

        # outer 计算外积，得到形状 [max_seq_len, d_k//2]
        '''
        假设：

            positions = [0, 1, 2]

            freqs = [f0, f1, f2, f3]

            那么：

                            f0      f1      f2      f3

            position 0       0       0       0       0
            position 1       f0      f1      f2      f3
            position 2       2f0     2f1     2f2     2f3
        '''
        sinusoids = torch.outer(positions, freqs) #保存旋转角度

        # 将余弦和正弦值注册为缓冲区（buffer），它们不是可学习参数，
        # persistent=False 表示不会保存到 state_dict 中，节省内存
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False)
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x, token_positions):
        # x: [..., seq_len, d_k]
        # token_positions: [..., seq_len]
        # cos_cache 形状 [max_seq_len, d_k//2]，索引后得到 [..., seq_len, d_k//2]
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        #如果是多头，则cos比x少一个维度
        if cos.ndim == x.ndim - 1:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_part1 = x[..., 0::2]
        x_part2 = x[..., 1::2]

        output1 = x_part1 * cos - x_part2 * sin  # 偶数位置的新值
        output2 = x_part1 * sin + x_part2 * cos  # 奇数位置的新值

        out = torch.stack([output1, output2], dim=-1)  # [batch, seq_len, d_k//2, 2]
        # stack对于每一个位置，把 output1 和 output2 的对应元素打包在一起
        '''
        原来：

            output1 = [x0', x2', x4', x6']

            output2 = [x1', x3', x5', x7']

            执行：

            torch.stack([output1, output2], dim=-1)

            得到：

            [
                [x0', x1'],
                [x2', x3'],
                [x4', x5'],
                [x6', x7']
            ]
        '''
        # 将最后两维展平，恢复成 [batch, seq_len, d_k]
        # flatten(-2) 从倒数第二维开始展平，即把 d_k//2 和 2 合并成 d_k
        out = out.flatten(-2)  # [batch, seq_len, d_k]

        return out