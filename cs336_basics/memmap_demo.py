import numpy as np

from cs336_basics.data_loader import get_batch
'''完整数据留在磁盘上，每次只准备训练需要的一小批。'''

# 生成一个小文件，用于演示。
np.save("tokens_demo.npy", np.arange(100, dtype=np.uint16))

# 以内存映射方式打开。
data = np.load("tokens_demo.npy", mmap_mode="r")

print(type(data))  # <class 'numpy.memmap'>
print(data[:10])

x, y = get_batch(
    data,
    batch_size=2,
    context_length=5,
    device="cpu",
)

print("输入：", x)
print("目标：", y)