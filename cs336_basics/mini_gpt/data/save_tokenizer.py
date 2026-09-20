import json


def save_vocab(
    vocab,
    path="vocab.json",
):
    """
    保存byte级vocab
    """

    vocab_to_save = {}

    for token_id, token_bytes in vocab.items():

        # bytes 转成整数列表
        # 例如:
        # b"hello"
        # ->
        # [104,101,108,108,111]

        vocab_to_save[str(token_id)] = list(
            token_bytes
        )


    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            vocab_to_save,
            f,
            indent=2,
        )

def save_merges(
    merges,
    path="merges.json",
):

    data = []

    for left, right in merges:

        data.append(
            [
                list(left),
                list(right),
            ]
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )