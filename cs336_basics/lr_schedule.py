import math

def get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    if it <= cosine_cycle_iters:

        progress = (
            (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )

        return min_learning_rate + (
            0.5 * (1 + math.cos(math.pi * progress))
            * (max_learning_rate - min_learning_rate)
        )

    return min_learning_rate
