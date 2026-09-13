import re
import regex
from collections import defaultdict

#这里最新采用了倒排索引的机制
'''
初始化完成后，三张表形成关系

例如：

token_frequency_table
─────────────────────
(h,e,l,l,o) → 10
(l,l,l)     → 3


pair_to_tokens
─────────────────────
(l,l)
  ↓
{
  (h,e,l,l,o),
  (l,l,l)
}


pair_counts
─────────────────────
(l,l) → 13
'''
def build_pair_to_tokens(
        token_frequency_table: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
    pair_to_tokens = defaultdict(set)

    for token_tuple in token_frequency_table.keys():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_to_tokens[pair].add(token_tuple)

    return pair_to_tokens

def update_pair_to_tokens(
        pair_to_tokens, 
        old_token: tuple[bytes, ...],
        new_token: tuple[bytes, ...]
    ) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
        for i in range(len(old_token) - 1):
            pair = (old_token[i], old_token[i + 1])
            if old_token in pair_to_tokens[pair]:
                pair_to_tokens[pair].remove(old_token)
            if not pair_to_tokens[pair]:
                del pair_to_tokens[pair]
        for i in range(len(new_token) - 1):
            pair = (new_token[i], new_token[i + 1])
            pair_to_tokens[pair].add(new_token)

        return pair_to_tokens

        
def build_vocab():
    vocab = {i : bytes([i]) for i in range(256)}    
    return vocab

#以特殊词为划分边界，把原始文本划分为多个子文本，返回子文本列表
def split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split the input text into a list of tokens, ensuring that special tokens are treated as separate tokens.
    """
    if not special_tokens:
        return [text]
    # Sort special tokens by length in descending order to match longer tokens first
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    
    # Create a regex pattern to match special tokens
    special_tokens_pattern = '|'.join(re.escape(token) for token in sorted_special_tokens)
    
    tokens = re.split(f'({special_tokens_pattern})', text)
    
    return [token for token in tokens if token]


#将划分后的子文本逐个进行预分词，跳过特殊词，并用一个字典记录，
#这里还要将每个分词的每个预分词结果bytes化一下，再逐个bytes化，作为Key,对应value加一即可，最后返回统计字典
#返回的结果类似于：{
#    (b'h', b'e', b'l', b'l', b'o'): 2,
#    (b'w', b'o', b'r', b'l', b'd'): 1
#}
def pre_tokenize_and_count_pretoken(chunks: list[str], special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize the input text, skipping special tokens, and return a dictionary of token counts.
    """
    token_counts = defaultdict(int)  
    PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    for chunk in chunks:
        # Skip special tokens
        if chunk in special_tokens:
            continue

        matches = PAT.finditer(chunk)
        for match in matches:
            token = match.group(0)
            token_bytes = token.encode('utf-8')  # Convert token to bytes
            ans = tuple(bytes([b]) for b in token_bytes)  # Convert each byte to bytes

            token_counts[ans] += 1

    return token_counts

#统计 pair frequency
def count_pair_frequencies(token_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts = defaultdict(int)

    for token_tuple, count in token_counts.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] += count

    return pair_counts

def update_token(old_token,
                 pair_to_merge: tuple[bytes, bytes]
            ) -> tuple[bytes, ...]:
    new_token = []
    i = 0
    while i < len(old_token):
        if i < len(old_token) - 1 and old_token[i] == pair_to_merge[0] and old_token[i + 1] == pair_to_merge[1]:
            new_token.append(pair_to_merge[0] + pair_to_merge[1]) 
            i += 2
        else:
            new_token.append(old_token[i])
            i += 1
    return tuple(new_token)


def run_bpe(
        vocab: dict[int, bytes], 
        token_counts: dict[tuple[bytes, ...], int], 
        special_tokens: list[str], 
        vocab_size: int
    ):
    idx = len(vocab)
    for special_token in special_tokens:
        vocab[idx] = special_token.encode("utf-8")
        idx += 1

    pair_to_tokens = build_pair_to_tokens(token_counts) #建立倒排索引表

    merges = []

    while len(vocab) < vocab_size:
        pair_counts = count_pair_frequencies(token_counts)

        best_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))  

        #不能一边遍历原来的 set 一边修改它。所以要复制一份
        affected_tokens = pair_to_tokens[best_pair].copy()

        for old_token in affected_tokens: #直接找到所有包含 best_pair 的 token
            for i in range(len(old_token) - 1):
                pair = (old_token[i], old_token[i + 1])
                pair_counts[pair] -= token_counts[old_token]
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            new_token = update_token(old_token, best_pair)

            for i in range(len(new_token) - 1):
                pair = (new_token[i], new_token[i + 1])
                pair_counts[pair] += token_counts[old_token]

            token_counts[new_token] += token_counts[old_token]
            del token_counts[old_token]

            update_pair_to_tokens(pair_to_tokens, old_token, new_token)

        merges.append(best_pair)
 
        vocab[idx] = best_pair[0] + best_pair[1]
        idx += 1

        

    return vocab, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = build_vocab()

    chunks = split_special_tokens(text, special_tokens)

    token_counts = pre_tokenize_and_count_pretoken(chunks, special_tokens)

    vocab, merges = run_bpe(vocab, token_counts, special_tokens, vocab_size)

    return vocab, merges


'''
① 找 affected_tokens

② 对每个 old_token：
      ├── 找 freq
      ├── pair_counts 减去 old_token 的所有 pair
      ├── 得到 new_token
      ├── pair_counts 加上 new_token 的所有 pair
      ├── token_frequency_table 给 new_token 增加 freq
      ├── token_frequency_table 删除 old_token
      └── update_pair_to_tokens()


                 best_pair
                     │
                     ↓
              pair_to_tokens
                     │
                     ↓
              affected_tokens
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
      old_token              old_token
          │                     │
          ↓                     ↓
   pair_counts 减旧贡献    token_frequency
          │                     │
          └──────────┬──────────┘
                     ↓
                  merge
                     ↓
                new_token
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
     pair_counts  token_freq  pair_to_tokens
       加新贡献     更新频率      更新关系
'''