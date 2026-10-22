"""使用 run1 中的 16 词表模型续写数字序列，不涉及文本分词。"""

import torch

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.decoding import generate


def main():
    torch.manual_seed(42)

    # 先创建与训练时结构相同的模型。
    model = TransformerLM(
        vocab_size=16,
        context_length=16,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=192,
        rope_theta=10000,
        device="cpu",
    )

    # 读取存档，只恢复模型状态。
    checkpoint = torch.load(
        "artifacts/checkpoints/run1/final.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])

    # 输入一条序列，形状为 (1, 3)。
    prompt = torch.tensor(
        [[0, 1, 2]],
        dtype=torch.long,
        device="cpu",
    )

    # 最多额外生成 20 个 token。
    result = generate(
        model=model,
        token_ids=prompt,
        max_new_tokens=20,
        context_length=16,
        temperature=0.7,
        top_p=0.9,
    )

    print("输入：", prompt[0].tolist())
    print("完整结果：", result[0].tolist())

    # 去掉原始输入，只显示新生成的部分。
    new_tokens = result[:, prompt.shape[1]:]
    print("新生成：", new_tokens[0].tolist())


if __name__ == "__main__":
    main()
