"""将几句演示文本用已有 BPE 编码，保存到 artifacts/text_demo/。"""

from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def main():
    tokenizer = Tokenizer.from_files(
        vocab_filepath="artifacts/bpe_debug/vocab.pkl",
        merges_filepath="artifacts/bpe_debug/merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    output_dir = Path("artifacts/text_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 演示用的小文本。
    train_documents = [
        "The cat is small.",
        "The dog is happy.",
    ]

    val_documents = [
        "The cat is happy.",
    ]

    def encode_documents(documents):
        ids = []

        for text in documents:
            # 每篇文档后添加结束符。
            ids.extend(
                tokenizer.encode(text + "<|endoftext|>")
            )

        return ids

    train_ids = encode_documents(train_documents)
    val_ids = encode_documents(val_documents)

    # 重复数据，方便多次采样练习。
    train_ids = train_ids * 100
    val_ids = val_ids * 20

    # 确认这些 ID 可以使用 uint16 保存。
    assert max(train_ids + val_ids) < 65536

    np.save(
        output_dir / "train.npy",
        np.array(train_ids, dtype=np.uint16),
    )

    np.save(
        output_dir / "val.npy",
        np.array(val_ids, dtype=np.uint16),
    )

    # 模型需要能够覆盖词表里的所有 ID。
    vocab_size = max(tokenizer.vocab) + 1

    print("模型需要的 vocab_size：", vocab_size)
    print("训练 token 数：", len(train_ids))
    print("验证 token 数：", len(val_ids))
    print("训练文本解码检查：")
    print(tokenizer.decode(train_ids[:30]))


if __name__ == "__main__":
    main()
