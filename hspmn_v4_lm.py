"""HSPMN v4.0 multi-layer LM.

Stack: embedding -> N x HSPMNBlockV4 -> RMSNorm -> tied lm_head.

Sized for RTX 5090 single-block research scale: configurations from 50M to
350M params. Above 350M the contextual-stream KV cache + reflexive state
saturate 24 GB during training even at S=2048.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hspmn_v4_0 import HSPMNBlockV4
from utils_v3_0 import HSPMNConfig


@dataclass
class HSPMNv4LMConfig:
    vocab_size: int = 32000
    n_layers: int = 12
    dim: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    mlp_ratio: int = 4
    max_seq_len: int = 2048
    rope_base: int = 10000
    target_sparsity: float = 0.25
    num_sink_tokens: int = 8
    reflexive: str = "gdn"  # 'elu1' | 'gdn'
    attention: str = "nsa"  # 'sqsk' | 'nsa'
    nsa_compress_block_size: int = 32
    nsa_compress_stride: int = 16
    nsa_window_size: int = 512
    router_local_window: int = 64
    router_l1_coef_init: float = 1e-5
    router_z_loss_coef: float = 1e-5
    aux_loss_coef: float = 0.01
    tie_embeddings: bool = True
    init_std_factor: float = 1.0
    random_gate: bool = False  # Aquino-Michaels protocol: freeze gate at N(0,1)


def _scaled_init_(weight: torch.Tensor, factor: float = 1.0):
    """Std = factor / sqrt(in_features). Standard GPT-2/Llama style."""
    if weight.dim() < 2:
        nn.init.zeros_(weight)
        return
    in_dim = weight.shape[-1]
    nn.init.normal_(weight, mean=0.0, std=factor / (in_dim**0.5))


class HSPMNv4LM(nn.Module):
    """Multi-layer HSPMN v4.0 language model."""

    def __init__(self, config: HSPMNv4LMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)

        block_cfg = HSPMNConfig(
            dim=config.dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            mlp_ratio=config.mlp_ratio,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            sparsity_k=config.target_sparsity,
        )
        self.layers = nn.ModuleList(
            [
                HSPMNBlockV4(
                    block_cfg,
                    num_sink_tokens=config.num_sink_tokens,
                    router_local_window=config.router_local_window,
                    router_target_sparsity=config.target_sparsity,
                    router_l1_coef_init=config.router_l1_coef_init,
                    router_z_loss_coef=config.router_z_loss_coef,
                    reflexive=config.reflexive,
                    attention=config.attention,
                    nsa_compress_block_size=config.nsa_compress_block_size,
                    nsa_compress_stride=config.nsa_compress_stride,
                    nsa_window_size=config.nsa_window_size,
                    layer_idx=i,
                )
                for i in range(config.n_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._init_weights()
        if config.random_gate:
            self._freeze_gates_random()

    def _freeze_gates_random(self):
        """Aquino-Michaels random-gate baseline: freeze every router's gate at
        a fresh N(0, 1) init so gradients cannot adapt the gating function.
        Q/K/V remain learnable. Used to measure absorption ratio at scale."""
        with torch.no_grad():
            for layer in self.layers:
                gate_lin = layer.router.gate_proj  # ReMoE attr name
                gate_lin.weight.normal_(mean=0.0, std=1.0)
                if gate_lin.bias is not None:
                    gate_lin.bias.zero_()
                gate_lin.weight.requires_grad_(False)
                if gate_lin.bias is not None:
                    gate_lin.bias.requires_grad_(False)

    def _init_weights(self):
        nn.init.normal_(
            self.embed_tokens.weight,
            std=self.config.init_std_factor / (self.config.dim**0.5),
        )
        if not self.config.tie_embeddings:
            _scaled_init_(self.lm_head.weight, self.config.init_std_factor)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_active_params(self):
        """Active = parameters that participate in every forward pass.
        For HSPMN this is everything (no MoE conditional routing); identical
        to total. Reported separately for compatibility with MoE comparisons.
        """
        return self.num_params()

    def forward(
        self,
        input_ids,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[tuple]] = None,
        return_aux: bool = True,
    ):
        h = self.embed_tokens(input_ids)
        aux_total = h.new_zeros(())
        new_pkvs = []
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h, aux, pkv = layer(h, past_key_values=past_key_values[i])
            aux_total = aux_total + aux
            new_pkvs.append(pkv)

        h = self.norm(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce + self.config.aux_loss_coef * aux_total
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_total,
            "past_key_values": new_pkvs if return_aux else None,
        }


def estimate_param_counts():
    """Print a few size points for planning."""
    configs = [
        ("50M", HSPMNv4LMConfig(n_layers=8, dim=512, num_heads=8, num_kv_heads=2)),
        ("100M", HSPMNv4LMConfig(n_layers=12, dim=640, num_heads=10, num_kv_heads=2)),
        ("160M", HSPMNv4LMConfig(n_layers=12, dim=768, num_heads=12, num_kv_heads=4)),
        ("350M", HSPMNv4LMConfig(n_layers=24, dim=1024, num_heads=16, num_kv_heads=4)),
    ]
    for name, cfg in configs:
        m = HSPMNv4LM(cfg)
        n = m.num_params()
        print(
            f"{name:>8s}  layers={cfg.n_layers:3d} dim={cfg.dim:4d}  params={n / 1e6:.1f}M"
        )
        del m


if __name__ == "__main__":
    estimate_param_counts()
