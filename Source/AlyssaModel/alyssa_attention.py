import torch
import torch.nn as nn
import torch.nn.functional as F

class AlyssaMultiHeadAttention(nn.Module):
    def __init__(self, embed_dims, n_attn_heads, attn_dropout, res_attn_dropout):
        super().__init__()
        assert embed_dims % n_attn_heads == 0

        self.num_attn_heads = n_attn_heads
        self.attn_head_dim = embed_dims // n_attn_heads

        self.attn_dropout_prob = attn_dropout
        self.res_attn_dropout_prob = res_attn_dropout

        self.res_attn_dropout = nn.Dropout(res_attn_dropout)

        self.Wq = nn.Linear(embed_dims, embed_dims)
        self.Wk = nn.Linear(embed_dims, embed_dims)
        self.Wv = nn.Linear(embed_dims, embed_dims)

        self.out_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, in_embeds):
        B, T, C = in_embeds.shape

        q = self.Wq(in_embeds).reshape(B, T, self.num_attn_heads, self.attn_head_dim).transpose(1, 2)
        k = self.Wk(in_embeds).reshape(B, T, self.num_attn_heads, self.attn_head_dim).transpose(1, 2)
        v = self.Wv(in_embeds).reshape(B, T, self.num_attn_heads, self.attn_head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_prob, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, C)

        attn_out = self.out_proj(attn_out)
        attn_out = self.res_attn_dropout(attn_out)

        return attn_out
