import re
import regex

'''
最朴素的方法
encode里对每一个预分词后的token都进行merge
'''

def split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    # 以特殊词为划分边界，把原始文本划分为多个子文本，返回子文本列表
    # 例如：text = "Hello, world! This is a test.", special_tokens = ["Hello", "test"]
    # 返回 ["Hello", ", world! This is a ", "test", "."]
    if not special_tokens:
        return [text]

    # 构建正则表达式模式
    special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )
    pattern = '|'.join(re.escape(token) for token in special_tokens)
    regex = re.compile(f'({pattern})')

    # 使用正则表达式进行分割
    parts = regex.split(text)

    # 去除空字符串
    return [part for part in parts if part]

def merge(token_tuple: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_token = pair[0] + pair[1]
    res = []
    i = 0
    while i < len(token_tuple):
        if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == pair:
            res.append(new_token)
            i += 2
        else:
            res.append(token_tuple[i])
            i += 1
    return tuple(res)

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        text = split_special_tokens(text, self.special_tokens)

        invert_vocab = {v: k for k, v in self.vocab.items()}

        token_id = []

        for token in text:
            if token in self.special_tokens:
                token_id.append(invert_vocab[token.encode("utf-8")])
            else:
                PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
                matches = PAT.finditer(token)
                for match in matches:
                    pre_token = match.group(0) #取出预分词结果
                    pre_token_bytes = pre_token.encode('utf-8') 
                    pre_token_tuple = tuple(bytes([b]) for b in pre_token_bytes) #离散byte化
                    # 还要进行merge
                    for pair in self.merges:
                        pre_token_tuple = merge(pre_token_tuple, pair)
                    for t in pre_token_tuple:
                        if t in invert_vocab:
                            token_id.append(invert_vocab[t])
                        else :
                            raise ValueError(f"Token {t} not in vocabulary.")

        return token_id

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        res = b""
        for token_id in token_ids:
            if token_id in self.vocab:
                token_bytes = self.vocab[token_id]
                res += token_bytes
            else:
                raise ValueError(f"Token ID {token_id} not in vocabulary.")
        return res.decode("utf-8", errors="replace")
 
    def encode_iterable(self, iterable):
        for text in iterable:
            token_ids = self.encode(text)

            for token_id in token_ids:
                yield token_id