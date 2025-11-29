import torch
import torch.nn as nn

from Source.AlyssaModel.alyssa_attention import AlyssaMultiHeadAttention

class AlyssaTransformerBlock(nn.Module):
    def __init__(self, embed_dims, n_attn_heads, attn_dropout, res_attn_dropout, mlp_dropout):
        super().__init__()

        self.in_embeds_norm = nn.LayerNorm(embed_dims)
        self.mul_head_attn_norm = nn.LayerNorm(embed_dims)

        self.mul_head_attn_layer = AlyssaMultiHeadAttention(embed_dims, n_attn_heads,
                                                            attn_dropout, res_attn_dropout)
        self.mlp_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.GELU(),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(mlp_dropout)
        )

    def forward(self, in_embeds):
        mul_head_attn_out = self.mul_head_attn_layer(self.in_embeds_norm(in_embeds))
        mul_head_attn_out = torch.add(mul_head_attn_out, in_embeds)

        mlp_out = self.mlp_layer(self.mul_head_attn_norm(mul_head_attn_out))

        trans_block_out = torch.add(mlp_out, mul_head_attn_out)

        return trans_block_out