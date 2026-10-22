"""讲义 §4.2 learning_rate_tuning：比较三个 SGD 学习率。"""

import torch

from cs336_basics.sgd import SGD


def main():
    torch.manual_seed(42)

    # 准备同一份初始值，方便公平比较不同学习率。
    initial_weights = 5 * torch.randn(10, 10)

    for lr in [10, 100, 1000]:
        weights = torch.nn.Parameter(initial_weights.clone())
        optimizer = SGD([weights], lr=lr)

        print(f"\n学习率：{lr}")

        for t in range(10):
            optimizer.zero_grad()
            loss = weights.square().mean()
            print(f"第 {t + 1} 轮，loss={loss.item():.6e}")
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
