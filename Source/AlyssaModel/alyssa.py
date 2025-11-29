import torch
import torch.nn as nn

from Source.AlyssaModel.alyssa_transformer_block import AlyssaTransformerBlock

class Alyssa(nn.Module):
    def __init__(self, context_window, vocab_size, embed_dims, n_trans_blocks,
                 n_attn_heads, attn_dropout, res_attn_dropout, mlp_dropout):
        super().__init__()

        self.context_window = context_window

        self.vocab_size = vocab_size
        self.embed_dims = embed_dims

        self.n_trans_blocks = n_trans_blocks

        self.n_attn_heads = n_attn_heads

        self.attn_dropout = attn_dropout
        self.res_attn_dropout = res_attn_dropout
        self.mlp_dropout = mlp_dropout

        self.trans_blocks_norm = nn.LayerNorm(embed_dims)

        self.token_embed_layer = nn.Embedding(vocab_size, embed_dims)
        self.pos_embed_layer = nn.Embedding(context_window, embed_dims)

        self.trans_blocks = nn.ModuleList([
            AlyssaTransformerBlock(embed_dims, n_attn_heads, attn_dropout, res_attn_dropout,
                                   mlp_dropout) for _ in range(n_trans_blocks)
        ])

        self.lm_head = nn.Linear(embed_dims, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed_layer.weight

    def forward(self, in_token_ids):
        token_embeds = self.token_embed_layer(in_token_ids)
        pos_embeds = self.pos_embed_layer(torch.arange(in_token_ids.size(1),
                                    device=in_token_ids.device).unsqueeze(0))

        in_embeds = torch.add(token_embeds, pos_embeds)

        trans_blocks_out = in_embeds
        for block in self.trans_blocks:
            trans_blocks_out = block(trans_blocks_out)

        lm_head_out = self.lm_head(self.trans_blocks_norm(trans_blocks_out))

        return lm_head_out
