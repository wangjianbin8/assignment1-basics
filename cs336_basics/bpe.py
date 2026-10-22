import regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#按特殊词划分后再预分词
def pretokenize(parts):
    for part in parts:
        for match in regex.finditer(PAT, part):
            yield match.group(0)


#统计预分词后片段数目
def count_word_counts(pre_tokens):
    word_counts = {}
    for s in pre_tokens:
        #每个片段是字符串，先通过utf-8编码变成字节序列,
        #然后再遍历每个字节（int类型）,
        #将其变成单独bytes对象，组合成元组（才能成为字典的键）
        #最开始的一个一个token都是一个字节大小
        tokens = tuple(bytes([b]) for b in s.encode("utf-8"))
        word_counts[tokens] = word_counts.get(tokens, 0) + 1
    return word_counts


#统计相邻对数量
def count_pairs(word_counts):
    pair_counts = {}
    for tokens, count in word_counts.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


#找到最大频次，频次相同则比较元组
def find_max_frq(pair_counts):
    # best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
    key = tuple((pair_counts[p], p) for p in pair_counts)
    best_pair = max(key)[1]
    return best_pair


#根据最大频次的pair来更新tokens
def merge_tokens(tokens, best_pair):
    merged = []
    i = 0
    while i < len(tokens):
        if (i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair):
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return tuple(merged)

#合并更新word_counts
def update_word_counts(word_counts, best_pair):
    new_word_counts = {}
    for tokens, count in word_counts.items():
        merged_tokens = merge_tokens(tokens, best_pair)
        if merged_tokens in new_word_counts:
            new_word_counts[merged_tokens] += count
        else:
            new_word_counts[merged_tokens] = count
    return new_word_counts


def train_bpe(input_path, vocab_size, special_tokens):
    #要返回的两个东西
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    
    #先将特殊词放到vocab中
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    #按特殊token划分
    if special_tokens:
        special_pattern = "|".join(
            regex.escape(token) for token in special_tokens
        )
        parts = regex.split(special_pattern, text)
    else:
        parts = [text]

    #预分词 + 片段计数
    word_counts = count_word_counts(pretokenize(parts))

    #有两种停止条件：词表达到上限或者已经没有可合并的相邻对
    while len(vocab) < vocab_size:
        pair_counts = count_pairs(word_counts)
        if len(pair_counts) == 0:
            break
        best_pair = find_max_frq(pair_counts)
        #加到vocab和merges里
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        word_counts = update_word_counts(word_counts, best_pair)

    return vocab, merges




        
        

    
        