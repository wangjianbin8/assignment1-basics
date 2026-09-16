import torch
import torch.nn as nn
from .RMSnorm import RMSnorm
from .SwiGLU import SwiGLU
from .causal_multi_head_attention_with_rope import CausalMultiHeadAttentionWithRoPE

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, theta,
            attn_q_proj_weight, attn_k_proj_weight, attn_v_proj_weight,
            attn_o_proj_weight, ln1_weight, ln2_weight, ffn_w1_weight, 
            ffn_w2_weight, ffn_w3_weight, device=None
    ):
        super().__init__()

        self.attn_q_proj_weight = attn_q_proj_weight
        self.attn_k_proj_weight = attn_k_proj_weight
        self.attn_v_proj_weight = attn_v_proj_weight
        self.attn_o_proj_weight = attn_o_proj_weight

        self.ln1_weight = ln1_weight
        self.ln2_weight = ln2_weight

        self.ffn_w1_weight = ffn_w1_weight
        self.ffn_w2_weight = ffn_w2_weight
        self.ffn_w3_weight = ffn_w3_weight

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device


        self.rms_norm1 = RMSnorm(d_model, eps=1e-5, device=device)
        self.rms_norm1.load_state_dict({"weight": self.ln1_weight})

        self.rms_norm2 = RMSnorm(d_model, eps=1e-5, device=device)
        self.rms_norm2.load_state_dict({"weight": self.ln2_weight})

        self.swiglu = SwiGLU(d_model, d_ff)
        self.swiglu.load_state_dict({
            "w1.weight": self.ffn_w1_weight,
            "w2.weight": self.ffn_w2_weight,
            "w3.weight": self.ffn_w3_weight
        })

        self.causal_multi_head_attention = CausalMultiHeadAttentionWithRoPE(
            d_model, n_heads, max_seq_len, theta, device
        )

    def forward(self, in_features):
        token_positions = torch.arange(in_features.shape[1], device=in_features.device)
        x1 = self.rms_norm1(in_features)

        x1 = self.causal_multi_head_attention(
            x1,
            self.attn_q_proj_weight,
            self.attn_k_proj_weight,
            self.attn_v_proj_weight,
            self.attn_o_proj_weight,
            token_positions
        )

        x1 = x1 + in_features

        x2 = self.rms_norm2(x1)

        x2 = self.swiglu(x2)

        out = x2 + x1

        return out



