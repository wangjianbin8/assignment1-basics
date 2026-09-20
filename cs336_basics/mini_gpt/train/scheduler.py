import math

def cosine_lr(
        step,
        max_steps,
        warmup_steps,
        max_lr,
        min_lr=0
):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step >= max_steps:
        return min_lr


    # cosine decay

    progress = (
        step - warmup_steps
    ) / (
        max_steps - warmup_steps
    )


    cosine = 0.5 * (
        1 + math.cos(
            math.pi * progress
        )
    )


    return min_lr + (
        max_lr-min_lr
    ) * cosine