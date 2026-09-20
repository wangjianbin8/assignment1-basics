import numpy as np
import torch

def load_bin(path):
    """
    加载token二进制文件
    """
    # 用内存映射打开 bin 文件
    # 不会把整个文件一次性加载到 RAM
    data = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )

    return data


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
):
    # dataset 是一维 token 序列
    # 例如：[10, 25, 38, 91, 72, ...]
    
    # 能够作为起点的最大数量
    # 例如：
    # dataset 长度 = 100
    # context_length = 7
    # 合法起点 = 0 ~ 92
    num_possible_starts = len(dataset) - context_length

    # 随机生成 batch_size 个起点
    # np.random.randint 左闭右开，所以这里写 num_possible_starts
    start_indices = np.random.randint(
        0,
        num_possible_starts,
        size=batch_size,
    )

    # 根据随机起点取出 x
    x = np.stack([
        dataset[i:i + context_length]
        for i in start_indices
    ])

    # y 比 x 向后移动一个 token
    y = np.stack([
        dataset[i + 1:i + context_length + 1]
        for i in start_indices
    ])

    # 转成 PyTorch Tensor
    # token ID 必须使用 long/int64
    x = torch.from_numpy(x).long()

    # y 同样转换成 long/int64
    y = torch.from_numpy(y).long()

    # 把数据移动到 GPU 或 CPU
    x = x.to(device)
    y = y.to(device)

    # 返回输入和标签
    return x, y



