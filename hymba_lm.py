"""Hymba-style multi-layer LM."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from hymba_block import HymbaBlock


class HymbaLM(nn.Module):
    def __init__(self, vocab_size, n_layers, dim, num_heads, num_kv_heads,
                 max_seq_len=2048, mlp_ratio=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        head_dim = dim // num_heads
        self.layers = nn.ModuleList([
            HymbaBlock(dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                       head_dim=head_dim, mlp_ratio=mlp_ratio,
                       max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=1.0 / (dim ** 0.5))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, labels=None):
        h = self.embed(ids)
        for layer in self.layers:
            h, _, _ = layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            sl = logits[..., :-1, :].contiguous()
            sb = labels[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sb.view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "aux_loss": logits.new_zeros(())}
