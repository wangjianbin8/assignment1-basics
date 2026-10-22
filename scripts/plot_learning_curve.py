import argparse
import csv
from pathlib import Path

import matplotlib

# 直接保存图片，不依赖桌面窗口，适合 WSL。
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)

    # 从 CSV 中读取每一行。
    with log_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("日志没有数据，请先运行训练")

    # CSV 读出的内容是字符串，需要转换成数字。
    steps = [int(row["step"]) for row in rows]
    seconds = [float(row["elapsed_seconds"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    val_losses = [float(row["val_loss"]) for row in rows]

    # 创建并排的两张图。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, x, label in [
        (axes[0], steps, "Training steps"),
        (axes[1], seconds, "Elapsed seconds"),
    ]:
        ax.plot(x, train_losses, marker="o", label="Train")
        ax.plot(x, val_losses, marker="o", label="Validation")

        ax.set_xlabel(label)
        ax.set_ylabel("Cross-entropy loss")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()

    output_path = log_path.with_suffix(".png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("学习曲线已保存：", output_path)


if __name__ == "__main__":
    main()