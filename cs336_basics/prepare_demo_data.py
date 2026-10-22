from pathlib import Path

import numpy as np


def main():
    output_dir = Path("artifacts/demo_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.tile(
        np.arange(16, dtype=np.uint16),
        100,
    )

    val_data = np.tile(
        np.arange(16, dtype=np.uint16),
        20,
    )

    np.save(output_dir / "train.npy", train_data)
    np.save(output_dir / "val.npy", val_data)

    print("演示数据已保存到：", output_dir)


if __name__ == "__main__":
    main()