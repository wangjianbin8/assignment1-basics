"""讲义 §5.1：演示内存映射；输出文件放在 artifacts/memmap_demo/。"""

from pathlib import Path

import numpy as np

from cs336_basics.data_loader import get_batch


def main():
    # 完整数据留在磁盘上，每次只准备训练需要的一小批。
    output_dir = Path("artifacts/memmap_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "tokens.npy"

    np.save(data_path, np.arange(100, dtype=np.uint16))
    data = np.load(data_path, mmap_mode="r")

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


if __name__ == "__main__":
    main()
