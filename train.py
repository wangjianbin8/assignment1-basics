import numpy as np
import torch
import argparse

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.data_loader import get_batch
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device):
    model.eval()

    losses = []
    for _ in range(10):
        x, y = get_batch(
            data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    # 模型设置。
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # 优化器设置。
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # 学习率调度设置。
    parser.add_argument("--max-lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=0.0001)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--cosine-cycle-iters", type=int, default=99)
    
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = args.device

    data = np.load(
        args.train_data,
        mmap_mode="r",
    )

    val_data = np.load(
        args.val_data,
        mmap_mode="r",
    )

    batch_size = args.batch_size
    context_length = args.context_length

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    total_steps = args.total_steps

    start_step = 0

    if args.resume is not None:
        start_step = load_checkpoint(
            src=args.resume,
            model=model,
            optimizer=optimizer,
        )

        print(f"已恢复存档，之前完成了 {start_step} 次更新")

    model.train()

    for step in range(start_step, total_steps):
        # 1. 设置本轮学习率。
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. 取一批输入和目标。
        x, y = get_batch(
            data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # 3. 清除旧梯度。
        optimizer.zero_grad()

        # 4. 前向传播，计算损失。
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 5. 反向传播，计算梯度。
        loss.backward()

        # 6. 裁剪梯度，再更新参数。
        gradient_clipping(
            model.parameters(),
            max_l2_norm=1.0,
        )
        optimizer.step()

        # 7. 打印训练情况。
        if (step + 1) % 10 == 0:
            val_loss = evaluate(
                model=model,
                data=val_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )

            print(
                f"已完成 {step + 1} 次更新，"
                f"train_loss={loss.item():.4f}，"
                f"val_loss={val_loss:.4f}，"
                f"lr={lr:.6f}"
            )

    # 训练结束后保存。
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_steps,
        out="checkpoint_demo.pt",
    )


if __name__ == "__main__":
    main()