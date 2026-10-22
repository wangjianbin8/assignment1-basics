"""讲义 §6：加载 text_demo 模型和配套 BPE，续写 The cat。"""

import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.decoding import generate_text


def main():
    torch.manual_seed(42)

    # 1. 加载训练数据时使用的同一个分词器。
    tokenizer = Tokenizer.from_files(
        vocab_filepath="artifacts/bpe_debug/vocab.pkl",
        merges_filepath="artifacts/bpe_debug/merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    # 2. 创建与训练时结构相同的模型。
    model = TransformerLM(
        vocab_size=max(tokenizer.vocab) + 1,
        context_length=16,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=192,
        rope_theta=10000,
        device="cpu",
    )

    # 3. 恢复文本模型的权重。
    checkpoint = torch.load(
        "artifacts/checkpoints/text_demo/final.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])

    # 4. 查询结束符的 token ID。
    eos_token_id = tokenizer.token_to_id[
        b"<|endoftext|>"
    ]

    # 5. 输入文本，进行续写。
    prompt = "The cat"

    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=40,
        context_length=16,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=eos_token_id,
    )

    print("输入：", prompt)
    print("完整结果：", result)


if __name__ == "__main__":
    main()
