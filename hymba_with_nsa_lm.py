"""LM wrapper for hymba-with-nsa ablation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from hymba_with_nsa_block import HymbaWithNSABlock


class HymbaWithNSALM(nn.Module):
    def __init__(self, vocab_size, n_layers, dim, num_heads, num_kv_heads,
                 max_seq_len=2048, mlp_ratio=4, nsa_window_size=256,
                 random_gate=False, attn_mode="nsa", use_attn_gate=False,
                 gate_mode="linear", pc_temperature=1.0,
                 nsa_select_from_compress=False, nsa_n_select_blocks=8,
                 stream_decor=False, decor_coef=0.0):
        super().__init__()
        self.decor_coef = float(decor_coef)
        self.embed = nn.Embedding(vocab_size, dim)
        head_dim = dim // num_heads
        self.layers = nn.ModuleList([
            HymbaWithNSABlock(dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                              head_dim=head_dim, mlp_ratio=mlp_ratio,
                              max_seq_len=max_seq_len,
                              nsa_window_size=nsa_window_size,
                              layer_idx=i,
                              random_gate=random_gate,
                              attn_mode=attn_mode,
                              use_attn_gate=use_attn_gate,
                              gate_mode=gate_mode,
                              pc_temperature=pc_temperature,
                              nsa_select_from_compress=nsa_select_from_compress,
                              nsa_n_select_blocks=nsa_n_select_blocks,
                              stream_decor=stream_decor)
            for i in range(n_layers)
        ])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=1.0 / (dim ** 0.5))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, labels=None):
        h = self.embed(ids)
        aux_total = h.new_zeros(())
        for layer in self.layers:
            h, aux, _ = layer(h)
            aux_total = aux_total + aux
        h = self.norm(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            sl = logits[..., :-1, :].contiguous()
            sb = labels[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sb.view(-1), ignore_index=-100)
            if self.decor_coef > 0.0:
                loss = loss + self.decor_coef * aux_total / max(1, len(self.layers))
        return {"logits": logits, "loss": loss, "aux_loss": aux_total.detach()}
