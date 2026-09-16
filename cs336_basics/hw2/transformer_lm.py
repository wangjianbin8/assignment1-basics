import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .linear_and_embedding_module import LinearModule, EmbeddingModule
from .RMSnorm import RMSnorm


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta,
            weights
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

    def forward(self, in_put):
        print(self.vocab_size)
        embedding_module = EmbeddingModule(self.vocab_size, self.d_model, device=in_put.device)
        embedding_module.load_state_dict({"embedding_matrix":self.weights["token_embeddings.weight"]})
        x = embedding_module(in_put)

        for layer in range(self.num_layers):
            attn_q_proj_weight = self.weights[f"layers.{layer}.attn.q_proj.weight"]
            attn_k_proj_weight = self.weights[f"layers.{layer}.attn.k_proj.weight"]
            attn_v_proj_weight = self.weights[f"layers.{layer}.attn.v_proj.weight"]
            attn_o_proj_weight = self.weights[f"layers.{layer}.attn.output_proj.weight"]
            ln1_weight = self.weights[f"layers.{layer}.ln1.weight"]
            ln2_weight = self.weights[f"layers.{layer}.ln2.weight"]
            ffn_w1_weight = self.weights[f"layers.{layer}.ffn.w1.weight"]
            ffn_w2_weight = self.weights[f"layers.{layer}.ffn.w2.weight"]
            ffn_w3_weight = self.weights[f"layers.{layer}.ffn.w3.weight"] 

            transformer_block = TransformerBlock(self.d_model,self.num_heads,self.d_ff,self.context_length,self.rope_theta,attn_q_proj_weight,attn_k_proj_weight,attn_v_proj_weight,attn_o_proj_weight,ln1_weight,ln2_weight,ffn_w1_weight,ffn_w2_weight,ffn_w3_weight,device=None)
            x  = transformer_block(x)

        transformer_block_out = x

        rms_norm = RMSnorm(self.d_model, eps = 1e-5, device = None)
        rms_norm.load_state_dict({"weight":self.weights["ln_final.weight"]})
        out_norm = rms_norm(transformer_block_out)

        linear_module = LinearModule(self.d_model, self.vocab_size, device=None)
        linear_module.load_state_dict({"W":self.weights["lm_head.weight"]})
        out_linear = linear_module(out_norm)
        # out = softmax(out_linear,dim=-1) #注意最后作业不需要softmax归一化
        return out_linear