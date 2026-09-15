from collections import defaultdict
import re
import regex

'''
最终算法

bytes
  ↓
Node(byte0) <-> Node(byte1) <-> Node(byte2) ...
  ↓
建立 heap，push (rank, seq, left, right)
  ↓
pop 最小的 rank
  ↓
检查 left.alive / right.alive / left.next is right
  ↓
不是 → lazy deletion，继续 pop
  ↓
是 → 合并，更新 left.value，断开 right
  ↓
只 push：
  (prev_node, left)
  (left, next_node)
  ↓
继续，直到 heap 空
  ↓
从头遍历链表，收集 alive 节点
'''

import heapq

class Node:
    __slots__ = ("value", "prev", "next", "alive")
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None
        self.alive = True   # 用于 lazy deletion


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

def bpe_merge_with_heap(pre_token_bytes: bytes, merge_ranks: dict) -> list[bytes]:
    if len(pre_token_bytes) < 2:
        return [bytes([b]) for b in pre_token_bytes]

    # 1. 建立双向链表
    nodes = [Node(bytes([b])) for b in pre_token_bytes]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
        nodes[i + 1].prev = nodes[i]

    # 2. 建立 heap
    heap = []
    for i in range(len(nodes) - 1):
        pair = (nodes[i].value, nodes[i + 1].value)
        rank = merge_ranks.get(pair)
        if rank is not None:
            heapq.heappush(heap, (rank, id(nodes[i]), nodes[i], nodes[i + 1]))

    # 3. 不断 pop 并 merge
    while heap:
        rank, _, left, right = heapq.heappop(heap)

        # lazy deletion 检查
        if not left.alive or not right.alive:
            continue
        if left.next is not right:
            continue
        # 检查这对 pair 的 rank 是否仍然匹配（防止旧记录误导）
        if merge_ranks.get((left.value, right.value)) != rank:
            continue

        # 4. 合并 left 和 right
        new_value = left.value + right.value
        left.value = new_value
        left.next = right.next
        if right.next is not None:
            right.next.prev = left
        right.alive = False

        # 5. 只更新左右邻居产生的新 pair
        prev_node = left.prev
        next_node = left.next

        if prev_node is not None:
            pair = (prev_node.value, left.value)
            r = merge_ranks.get(pair)
            if r is not None:
                heapq.heappush(heap, (r, id(prev_node), prev_node, left))

        if next_node is not None:
            pair = (left.value, next_node.value)
            r = merge_ranks.get(pair)
            if r is not None:
                heapq.heappush(heap, (r, id(left), left, next_node))

    # 6. 收集结果
    result = []
    cur = nodes[0]
    # 找到链表头
    while cur.prev is not None:
        cur = cur.prev
    while cur is not None:
        if cur.alive:
            result.append(cur.value)
        cur = cur.next
    return result

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        text = split_special_tokens(text, self.special_tokens)

        invert_vocab = {v: k for k, v in self.vocab.items()}

        token_id = []

        #用来找到最优的pair
        merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        for token in text:
            if token in self.special_tokens:
                token_id.append(invert_vocab[token.encode("utf-8")])
            else:
               
                matches = self.PAT.finditer(token)
                for match in matches:
                    pre_token = match.group(0) #取出预分词结果
                    pre_token_bytes = pre_token.encode('utf-8') 
                    tokens = tuple(bytes([b]) for b in pre_token_bytes) #离散byte化
                    # 还要进行merge

                    pre_token_bytes = pre_token.encode('utf-8')
                    tokens = bpe_merge_with_heap(pre_token_bytes, merge_ranks)  

                    for t in tokens:
                        if t in invert_vocab:
                            token_id.append(invert_vocab[t])
                        else:
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


'''
bytes
  ↓
Node(byte0) <-> Node(byte1) <-> Node(byte2) ...
  ↓
建立 heap
  ↓
heap 中保存：
(rank, left_node, right_node)
  ↓
不断 pop
  ↓
检查：
left_node.next is right_node ?
  ↓
不是 → lazy deletion，继续 pop
是 → merge
  ↓
只处理 merge 后的：
(left_neighbor, new_node)
(new_node, right_neighbor)
  ↓
继续
'''