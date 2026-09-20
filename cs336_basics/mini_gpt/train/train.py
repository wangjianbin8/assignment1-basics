import torch

from cs336_basics.mini_gpt.data.data_loader import load_bin,get_batch
from cs336_basics.mini_gpt.model.transformer_lm import TransformerLM
from .loss import cross_entropy
from .adamw import AdamW

from cs336_basics.mini_gpt import config
from cs336_basics.mini_gpt.train.scheduler import cosine_lr
from cs336_basics.mini_gpt.train.checkpoint import save_checkpoint

import os

os.makedirs(
    "checkpoints",
    exist_ok=True
)
# =====================
# 配置
# =====================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# =====================
# 创建模型
# =====================

model = TransformerLM(
    vocab_size=config.vocab_size,
    max_seq_len=config.context_length,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
    theta=config.theta,
)
model = model.to(device)

# =====================
# 创建优化器
# =====================

optimizer = AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,

)

print("model initialized")

# =====================
# 加载数据
# =====================

train_data = load_bin(
    "cs336_basics/mini_gpt/data/train.bin"
)

valid_data = load_bin(
    "cs336_basics/mini_gpt/data/valid.bin"
)

print("train tokens:",len(train_data))
print("valid tokens:",len(valid_data))

def evaluate(
    model,
    data,
    batch_size,
    context_length,
    device,
    eval_steps=20,
):
    model.eval()
    losses = []

    # 不计算梯度
    with torch.no_grad():
        for _ in range(eval_steps):
            # 随机取验证batch
            x, y = get_batch(
                data,
                batch_size,
                context_length,
                device
            )
            # forward
            logits = model(x)

            # loss
            loss = cross_entropy(
                logits,
                y
            )

            losses.append(
                loss.item()
            )
    # 回到训练模式
    model.train()
    # 求平均loss
    return sum(losses) / len(losses)


# =====================
# 开始训练
# =====================

model.train()

for step in range(config.max_steps):
    # 1. 获取一个batch

    x,y = get_batch(
        train_data,
        config.batch_size,
        config.context_length,
        device,
    )

    # 2. forward
    logits = model(x)

    # logits:
    # (batch, seq_len, vocab_size)

    # 3. 计算loss
    loss = cross_entropy(
        logits,
        y,
    )

    # 4. 清空旧梯度
    optimizer.zero_grad()

    # 5. 反向传播
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    lr = cosine_lr(step, config.max_steps, warmup_steps=500, max_lr=config.learning_rate)

    for group in optimizer.param_groups:
        group["lr"] = lr

    # 6. 更新参数
    optimizer.step()

    if step % 100 == 0:

        train_loss = loss.item()
        valid_loss = evaluate(
            model,
            valid_data,
            config.batch_size,
            config.context_length,
            device,
        )

        print(
            f"""
            step: {step}
            train loss: {train_loss:.4f}
            valid loss: {valid_loss:.4f}
            """
        )

        save_checkpoint(
            model,
            optimizer,
            step,
            f"checkpoints/step_{step}.pt"
        )