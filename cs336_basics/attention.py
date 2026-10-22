import math
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)

    return weights @ v