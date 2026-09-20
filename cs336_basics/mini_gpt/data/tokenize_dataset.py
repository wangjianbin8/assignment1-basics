import numpy as np  # 导入 NumPy，用于把一批 token id 转成 uint16 二进制

from .bpe import train_bpe                # 从 bpe 模块导入 BPE 训练函数
from .tokenizer_v3 import Tokenizer       # 从 tokenizer_v3 模块导入 Tokenizer 类
from .load_tokenizer import load_merges, load_vocab

def tokenize_dataset(
    input_path,     # 输入：原始文本文件路径（.txt）
    output_path,    # 输出：二进制 token 文件路径（.bin）
    tokenizer,      # 已训练好的 tokenizer，需实现 encode_iterable(f)
):
    """
    把文本文件流式编码成 token id，用 10000 个 token 一批的方式
    写入 .bin 文件（uint16 小端字节流）。
    优点：省内存 + 减少磁盘写入次数。
    """
    # 每次积累 1 万个 token
    buffer = []     # 空的 Python list，用来暂存待写入的 token id

    # 打开原始文本
    with open(
        input_path,         # 文本路径
        "r",                # 文本模式读取
        encoding="utf-8",   # UTF-8 解码
    ) as f:
        # 外层 with 结束自动关闭输入文件

        # 以二进制方式打开输出文件
        with open(
            output_path,    # 输出 .bin 路径
            "wb",           # 二进制写模式
        ) as out:
            # 内层 with 结束自动关闭输出文件

            # 一行一行进行 BPE
            for token_id in tokenizer.encode_iterable(f):
                # encode_iterable 是生成器：
                # - 从文件对象逐行（或分块）读取
                # - 每次 yield 出一个整数 token id
                # 这样不用一次性把整个文件读入内存

                # 把 token 暂时放进 buffer
                buffer.append(token_id)     # O(1) 追加到 list 尾部

                # buffer 满了以后再一次性写入
                if len(buffer) >= 10000:
                    # 当缓存达到 1 万 token 时触发批量写入

                    # 转成 uint16 数组
                    tokens = np.array(
                        buffer,             # 从 list 构造数组
                        dtype=np.uint16,    
                    )
                    # 每个 token 占 2 字节，1 万个 token 就是 20 KB

                    # 一次性写入 10000 个 token
                    out.write(
                        tokens.tobytes()    # 把 numpy 数组转成 raw bytes
                    )
                    # 只调用一次 write，比逐 token 写快得多

                    # 清空 buffer
                    buffer.clear()          # 原地清空 list，复用同一个对象

            # 文件结束后，把剩余 token 写进去
            if buffer:
                # 如果文件读完时 buffer 里还有没写出的 token
                # （总数不是 10000 的整数倍）

                tokens = np.array(
                    buffer,
                    dtype=np.uint16,
                )

                out.write(
                    tokens.tobytes()        # 把最后一批写出去
                )
                # 注意：这里没再 buffer.clear()，因为函数就快结束了

    print(f"保存完成: {output_path}")       # 打印完成信息


# ============ 训练 BPE tokenizer ============
from save_tokenizer import save_vocab, save_merges

# vocab, merges = train_bpe(
#     input_path="TinyStoriesV2-GPT4-train.txt",   # 用训练集训练 BPE
#     vocab_size=10000,                            # 词表大小 10000，故 uint16 安全
#     special_tokens=["<|endoftext|>"],            # 特殊 token
#     max_bytes=100*1024*1024
# )
# # 返回：
# # - vocab: token(str/bytes) -> id 的映射
# # - merges: BPE 合并规则（按学习顺序）

# save_vocab(
#     vocab,
#     "vocab.json",
# )

# save_merges(
#     merges,
#     "merges.txt",
# )

vocab = load_vocab("vocab.json")
merges = load_merges("merges.txt")

tokenizer = Tokenizer(
    vocab=vocab,                                  # 传词表
    merges=merges,                                # 传合并规则
    special_tokens=["<|endoftext|>"],             # 特殊 token，编码时不被切碎
)
# 构造 Tokenizer 实例

# ============ 编码训练集和验证集 ============

# tokenize_dataset(
#     "TinyStoriesV2-GPT4-train.txt",   # 训练集文本,太大了哩哩啦啦
#     "train.bin",                       # 训练集输出
#     tokenizer,                         # 同一个 tokenizer
# )

# tokenize_dataset(
#     "sample_300mb.txt",   # 这里就先取这么多训练集文本
#     "train_300mb.bin",                       # 训练集输出
#     tokenizer,                         # 同一个 tokenizer
# )

tokenize_dataset(
    "TinyStoriesV2-GPT4-valid.txt",   # 验证集文本
    "valid.bin",                       # 验证集输出
    tokenizer,                         # 必须和训练集用同一个 tokenizer
)
