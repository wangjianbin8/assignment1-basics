import torch

def cross_entropy(
    logits, targets
):
    """
    logits:
        (batch, seq_len, vocab_size)


    targets:
        (batch, seq_len)


    return:
        scalar loss
    """
    vocab_size = logits.shape[-1]
    # -----------------------
    # 展平
    # -----------------------
    
    # 原来:
    #
    # (batch,seq,vocab)
    #
    # 例如:
    #
    # (2,4,10000)

    logits = logits.reshape(
        -1,
        vocab_size
    )


    # 变成:
    #
    # (8,10000)
    #
    # 每一行代表一个token预测

    # targets:

    targets = targets.reshape(
        -1
    )

    # 变成:
    #
    # (8,)



    # -----------------------
    # log softmax
    # -----------------------

    log_probs = torch.log_softmax(
        logits,
        dim=-1
    )
    #注意用了稳定化技巧
    #torch.log_softmax(logits,dim=-1) == logits - torch.logsumexp(logits,dim=-1,keepdim=True)

    loss = -log_probs[
        torch.arange(
            targets.shape[0],
            device=targets.device
        ),
        targets
    ]

    return loss.mean()
