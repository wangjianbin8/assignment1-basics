import regex
import pickle
from cs336_basics.bpe import PAT, merge_tokens, pretokenize

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # 复制输入，避免修改调用者的词表和列表。
        self.vocab = dict(vocab)
        self.merges = list(merges)

        self.merge_ranks = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        self.special_tokens = special_tokens or []

        # 编码时，需要从字节串查询 token ID。
        self.token_to_id = {
            token: token_id
            for token_id, token in self.vocab.items()
        }

        #讲义要求：特殊 token 不在词表中时，将它加入。
        next_id = max(self.vocab, default=-1) + 1

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                self.vocab[next_id] = token_bytes
                self.token_to_id[token_bytes] = next_id
                next_id += 1

    def decode(self, ids: list[int]):
        token_bytes = [self.vocab[token_id] for token_id in ids]
        full_bytes = b"".join(token_bytes)
        return full_bytes.decode("utf-8", errors="replace")

    def _encode_piece(self, piece: str) -> list[int]:
        tokens = tuple(bytes([value]) for value in piece.encode("utf-8"))
        while len(tokens) > 1:
            candidates = [(tokens[i], tokens[i + 1]) 
                          for i in range(len(tokens) - 1)
                          if (tokens[i], tokens[i + 1]) in self.merge_ranks]
            if not candidates:
                break
            best_pair = min(candidates, key=lambda pair: self.merge_ranks[pair])

            tokens = merge_tokens(tokens, best_pair)

        return [self.token_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        ids = []
        if self.special_tokens:
            # 较长的特殊 token 优先，处理相互重叠的情况。
            specials = sorted(
                self.special_tokens, key=len, reverse=True,
            )
            #这里和前面bpe的划分不同，会把特殊词保留下来
            pattern = "(" + "|".join(
                regex.escape(token) for token in specials
            ) + ")"
            parts = regex.split(pattern, text)
        else:
            parts = [text]

        special_set = set(self.special_tokens)

        for part in parts:
            if part in special_set:
                token_bytes = part.encode("utf-8")
                ids.append(self.token_to_id[token_bytes])
            else:
                for piece in pretokenize([part]):
                    ids.extend(self._encode_piece(piece))

        return ids

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens=None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )


    def _safe_prefix_end(self, text):
        spans = []

        def add_normal_spans(start, end):
            for match in regex.finditer(PAT, text[start:end]):
                piece = match.group(0)
                spans.append((
                    start + match.end(),
                    not piece[-1].isspace(),
                ))

        specials = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        offset = 0

        if specials:
            pattern = "|".join(
                regex.escape(token) for token in specials
            )

            for match in regex.finditer(pattern, text):
                add_normal_spans(offset, match.start())

                # 完整特殊 token 的结束位置。
                spans.append((match.end(), True))
                offset = match.end()

        add_normal_spans(offset, len(text))

        # 保留足够的尾部，防止特殊 token 尚未收完整。
        longest_special = max(
            (len(token) for token in specials),
            default=0,
        )
        limit = len(text) - longest_special

        # 额外保留末尾两个片段，保护可能变化的普通匹配。
        candidates = [
            end
            for end, safe in spans[:-2]
            if safe and end <= limit
        ]

        return max(candidates, default=0)

    def encode_iterable(self, iterable):
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            cut = self._safe_prefix_end(buffer)

            if cut > 0:
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]

        # 输入结束，剩余内容不再需要等待。
        yield from self.encode(buffer)

'''
为什么需要流式编码？
大文件不能一次全部读入内存，需要分块读取、逐步输出 token ID。

为什么不能每块独立编码？
分块位置可能落在单词或特殊 token 中间。例如 ["hel", "lo"] 应当作为 "hello" 处理，否则可能改变编码结果。

缓冲区解决什么问题？
已经确定边界的前缀先处理；尚未收完整的尾部暂存，等下一块到来。输入结束后，再处理全部剩余内容。

'''