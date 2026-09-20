import torch

#相对导入
from ..data.data_loader import load_bin,get_batch 

#绝对导入
from cs336_basics.mini_gpt.model.transformer_lm import TransformerLM

from cs336_basics.mini_gpt.train.loss import cross_entropy
from cs336_basics.mini_gpt.train.adamw import AdamW

# =====================
# 配置
# =====================

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 8

context_length = 256

max_steps = 1000

# =====================
# 加载数据
# =====================

train_data = load_bin(
    "cs336_basics/mini_gpt/data/train.bin"
)

# =====================
# 创建模型
# =====================

model = TransformerLM(
    vocab_size=10000,
    max_seq_len=context_length,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
)
model = model.to(device)

# =====================
# 创建优化器
# =====================

optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
)


# =====================
# 开始训练
# =====================

model.train()

# 固定一个batch

x,y = get_batch(
    train_data,
    batch_size=8,
    context_length=256,
    device=device,
)

for step in range(200):


    # forward

    logits = model(x)


    # loss

    loss = cross_entropy(
        logits,
        y,
    )


    optimizer.zero_grad()


    loss.backward()


    optimizer.step()



    if step % 10 == 0:

        print(
            step,
            loss.item()
        )