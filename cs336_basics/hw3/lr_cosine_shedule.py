import math
'''
阶段1:
it < warmup_iters

小 lr → max_lr
(线性增加)


阶段2:
warmup结束

max_lr → min_lr
(余弦下降)


阶段3:
训练结束

保持 min_lr
'''
class CosineSchedule:
    def __init__(
            self,
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            cosine_cycle_iters
    ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it):
        if it < self.warmup_iters:
            return self.max_learning_rate * it / self.warmup_iters

        elif it > self.cosine_cycle_iters:
            return self.min_learning_rate

        return (
            self.min_learning_rate
            +
            (self.max_learning_rate - self.min_learning_rate)
            * 
            (
                1 + math.cos(math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters))
            )
            / 2
        )

