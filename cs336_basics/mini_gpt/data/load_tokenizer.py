import json

def load_vocab(
    path="vocab.json",
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        vocab_json = json.load(f)

    vocab = {}

    for token_id, byte_list in vocab_json.items():

        # list[int]
        # 转回 bytes

        vocab[int(token_id)] = bytes(
            byte_list
        )


    return vocab



def load_merges(
    path="merges.json",
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)


    merges = []

    for left, right in data:

        merges.append(
            (
                bytes(left),
                bytes(right),
            )
        )

    return merges