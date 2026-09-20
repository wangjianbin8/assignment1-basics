import torch

from cs336_basics.mini_gpt.model.transformer_lm import TransformerLM
from cs336_basics.mini_gpt.data.load_tokenizer import load_vocab, load_merges
from cs336_basics.mini_gpt.data.tokenizer_v3 import Tokenizer
from cs336_basics.mini_gpt import config
from .sampling import top_k_sampling, top_p_sampling

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 加载 tokenizer
# ======================
vocab = load_vocab(
    "cs336_basics/mini_gpt/data/vocab.json"
)

merges = load_merges(
    "cs336_basics/mini_gpt/data/merges.txt"
)

tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=[
        "<|endoftext|>"
    ],
)

# ======================
# 创建模型
# ======================
model = TransformerLM(
    vocab_size=config.vocab_size,
    max_seq_len=config.context_length,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
    theta=config.theta,
)

# ======================
# 加载checkpoint
# ======================
checkpoint = torch.load(
    "checkpoints/step_5000.pt",
    map_location=device,
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.to(device)
model.eval()

def generate_one_token(
    model,
    tokens,
    temperature=1.0
):
    """
    根据已有tokens生成下一个token
    """
    # tokens:
    # [batch, seq_len]
    with torch.no_grad():
        # forward
        logits = model(tokens)

        # 只取最后一个位置
        # 因为我们只预测下一个token
        logits = logits[:, -1, :]

        # temperature控制随机程度
        logits = logits / temperature

        logits = top_k_sampling(
            logits,
            top_k=50
        )

        # 转概率
        probs = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

    return next_token

def generate(
    model,
    input_ids,
    max_new_tokens=200,
    temperature=1.0
):
    model.eval()
    for _ in range(max_new_tokens):
        next_token = generate_one_token(model, input_ids, temperature)
        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )
    return input_ids

prompt = "I love him, but"

# 编码prompt

tokens = tokenizer.encode(
    prompt
)

input_ids = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device,
)

# 生成

output_ids = generate(
    model,
    input_ids,
    max_new_tokens=200,
    temperature=0.8,
)

# tensor -> list
output_ids = output_ids[0].tolist()

# decode
text = tokenizer.decode(
    output_ids
)
print(text)

