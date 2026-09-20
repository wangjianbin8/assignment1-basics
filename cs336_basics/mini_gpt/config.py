# 模型词表大小
vocab_size = 10000

# 最大上下文长度
context_length = 256

# Transformer隐藏维度
d_model = 512

# Transformer层数
num_layers = 4

# attention头数量
num_heads = 16

# FFN隐藏层大小
d_ff = 1344

# RoPE参数
theta = 10000


# ===================
# 训练参数
# ===================

# batch大小
batch_size = 32

# 总训练步数
max_steps = 5000


# 学习率
learning_rate = 3e-4

# weight decay
weight_decay = 0.01