import torch

from cs336_basics.sgd import SGD


torch.manual_seed(42)

# 准备同一份初始值，方便公平比较不同学习率。
initial_weights = 5 * torch.randn(10, 10)

for lr in [10, 100, 1000]:
    # 每次实验都从相同的初始参数重新开始。
    weights = torch.nn.Parameter(initial_weights.clone())
    optimizer = SGD([weights], lr=lr)

    print(f"\n学习率：{lr}")

    for t in range(10):
        optimizer.zero_grad()

        loss = weights.square().mean()

        print(f"第 {t + 1} 轮，loss={loss.item():.6e}")

        loss.backward()
        optimizer.step()