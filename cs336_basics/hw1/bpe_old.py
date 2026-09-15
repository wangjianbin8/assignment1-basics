import re
import regex
from collections import defaultdict

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
    
    # Split the text using the regex pattern
    tokens = re.split(f'({special_tokens_pattern})', text)
    
    # Filter out empty strings and return the list of tokens
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

def update_token_counts(token_counts: dict[tuple[bytes, ...], int], pair_to_merge: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    new_token_counts = defaultdict(int)
    merged_token = pair_to_merge[0] + pair_to_merge[1]

    for token_tuple, count in token_counts.items():
        new_tuple = []
        i = 0
        while i < len(token_tuple):
            if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == pair_to_merge:
                new_tuple.append(merged_token)
                i += 2
            else:
                new_tuple.append(token_tuple[i])
                i += 1
        new_token_counts[tuple(new_tuple)] += count
    return new_token_counts

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

    merges = []

    while len(vocab) < vocab_size:
        pair_counts = count_pair_frequencies(token_counts)

        best_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))  

        merges.append(best_pair)
 
        vocab[idx] = best_pair[0] + best_pair[1]
        idx += 1

        token_counts = update_token_counts(token_counts, best_pair)

    return vocab, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = build_vocab()

    chunks = split_special_tokens(text, special_tokens)

    token_counts = pre_tokenize_and_count_pretoken(chunks, special_tokens)

    vocab, merges = run_bpe(vocab, token_counts, special_tokens, vocab_size)
    
    return vocab, merges