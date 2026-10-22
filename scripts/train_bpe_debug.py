"""讲义 §2 的 BPE 小语料演示；此脚本没有训练完整 TinyStories 数据集。"""

import pickle
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def main():
    output_dir = Path("artifacts/bpe_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建小语料，先检查实验流程。
    input_path = output_dir / "sample.txt"
    input_path.write_text(
        "ab ab ab abc abc bc bc bc bc",
        encoding="utf-8",
    )

    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=270,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start

    print(f"训练耗时：{elapsed:.4f} 秒")
    print(f"词表大小：{len(vocab)}")
    print(f"合并次数：{len(merges)}")

    # 保存结果。
    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"

    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)

    with merges_path.open("wb") as f:
        pickle.dump(merges, f)

    # 读回检查。
    with vocab_path.open("rb") as f:
        loaded_vocab = pickle.load(f)

    with merges_path.open("rb") as f:
        loaded_merges = pickle.load(f)

    assert loaded_vocab == vocab
    assert loaded_merges == merges

    print(f"保存和读取检查通过：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
