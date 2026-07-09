"""HSPMN v4.0 block - ReMoE router + decoupled stream-specific QKV projections.

Two structural changes versus v3.0:

1. **ReMoE router** (`ReMoERouter`) replaces sigmoid+0.05-threshold+dual-entropy.
   Gradient flows through the gate at every nonzero ReLU value, defeating the
   "zero gradient through top-k" failure identified in arXiv:2603.02227.

2. **Decoupled QKV per stream**. Aquino-Michaels (arXiv:2603.02227) shows that
   shared Q/K/V projections across stream pathways enable cross-layer
   compensation that absorbs the routing signal - at 31M params, learned and
   random gates converge in perplexity. v4.0 gives the Reflexive Stream and
   the Contextual Stream separate Q/K/V projections so neither stream can
   silently rotate to compensate for a bad mask in the other.

Design notes:
- Sink tokens reduced from 128 (v3.0) to 8 - matches StreamingLLM evidence
  (Xiao et al., ICLR 2024) and drops a no-op RoPE branch.
- Continuous gate value (ReLU magnitudes, not sigmoid probs) multiplies the
  contextual stream output. ReMoE-faithful: dead tokens contribute zero,
  active tokens scale linearly with their gate strength.
- Reflexive stream is *unchanged* in this commit - still ELU+1 chunked linear
  attention from `LinearStateSpaceStream`. Phase 2b replaces it with Gated
  DeltaNet. We isolate the absorption-defense change here so the ablation
  v3-block-with-v4-router-only is clean.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    from kernels_v3_0 import sparse_query_sparse_key_attention
    HAS_TRITON_KERNELS = True
except ImportError:
    HAS_TRITON_KERNELS = False

from hspmn_v3_0 import RotaryEmbedding, apply_rotary_pos_emb, LinearStateSpaceStream
from gated_deltanet import GatedDeltaNetReflexive
from mamba3_block import Mamba3SISOReflexive
from nsa_attention import NSAAttention
from router_v4_0 import ReMoERouter
from utils_v3_0 import HSPMNConfig

__all__ = ["HSPMNBlockV4"]


class HSPMNBlockV4(nn.Module):
    """v4.0 dual-stream block with decoupled QKV and ReMoE routing."""

    def __init__(self, config: HSPMNConfig, num_sink_tokens: int = 8,
                 router_local_window: int = 64,
                 router_target_sparsity: float = None,
                 router_l1_coef_init: float = 1e-3,
                 router_z_loss_coef: float = 1e-3,
                 reflexive: str = "elu1",
                 attention: str = "sqsk",
                 nsa_compress_block_size: int = 32,
                 nsa_compress_stride: int = 16,
                 nsa_window_size: int = 512,
                 layer_idx: int = 0):
        """`reflexive`: 'elu1' (v3 ELU+1) or 'gdn' (Gated DeltaNet).
        `attention`:   'sqsk' (v3-style top-k SQSK + sinks)
                       or 'nsa' (NSA triple-branch: compress + select + window).
        """
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.num_sink_tokens = num_sink_tokens

        target = float(router_target_sparsity if router_target_sparsity is not None else config.sparsity_k)
        self.router = ReMoERouter(
            config.dim,
            target_sparsity=target,
            local_window=router_local_window,
            l1_coef_init=router_l1_coef_init,
            z_loss_coef=router_z_loss_coef,
        )

        self.norm = nn.RMSNorm(config.dim)

        # ---- DECOUPLED QKV: separate projections per stream ----
        # Reflexive stream sees every token; uses its own Q/K/V.
        self.q_proj_refl = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj_refl = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.v_proj_refl = nn.Linear(config.dim, self.kv_dim, bias=False)
        # Contextual stream sees only routed tokens; uses its own Q/K/V.
        self.q_proj_ctx = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj_ctx = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.v_proj_ctx = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)
        if reflexive == "elu1":
            self.reflexive = LinearStateSpaceStream(
                config.dim, config.num_heads, config.num_kv_heads,
                config.head_dim, config.mlp_ratio,
            )
        elif reflexive == "gdn":
            self.reflexive = GatedDeltaNetReflexive(
                config.dim, config.num_heads, config.num_kv_heads,
                config.head_dim, config.mlp_ratio,
            )
        elif reflexive == "mamba3":
            self.reflexive = Mamba3SISOReflexive(
                config.dim, config.num_heads, config.num_kv_heads,
                config.head_dim, config.mlp_ratio,
            )
        elif reflexive == "rwkv7":
            from rwkv7_block import RWKV7Reflexive
            self.reflexive = RWKV7Reflexive(
                config.dim, config.num_heads, config.num_kv_heads,
                config.head_dim, config.mlp_ratio,
            )
        else:
            raise ValueError(f"reflexive must be 'elu1'|'gdn'|'mamba3'|'rwkv7', got {reflexive!r}")
        self.reflexive_kind = reflexive

        # 8 sink tokens (down from 128). Separate learnable embedding.
        self.sink_tokens = nn.Parameter(torch.zeros(1, self.num_sink_tokens, config.dim))

        # Optional NSA triple-branch / ASA two-branch contextual attention.
        self.attention_kind = attention
        if attention in ("nsa", "asa"):
            self.nsa = NSAAttention(
                num_heads=self.num_heads, head_dim=self.head_dim,
                compress_block_size=nsa_compress_block_size,
                compress_stride=nsa_compress_stride,
                window_size=nsa_window_size,
                mode=attention,
                layer_idx=layer_idx,
            )
        elif attention != "sqsk":
            raise ValueError(f"attention must be 'sqsk'|'nsa'|'asa', got {attention!r}")

        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.dim)
        for mod in [self.q_proj_refl, self.k_proj_refl, self.v_proj_refl,
                    self.q_proj_ctx, self.k_proj_ctx, self.v_proj_ctx, self.o_proj]:
            nn.init.xavier_uniform_(mod.weight, gain=scale)
        nn.init.normal_(self.sink_tokens, std=0.02)

    def _attention_triton(self, q, k, v, q_indices, kv_indices_full, B, S, D, seq_len_offset):
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        idx_exp = q_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)
        q_selected = torch.gather(q_t, 1, idx_exp).contiguous()

        real_q_indices = (q_indices + seq_len_offset).to(torch.int32)
        real_sink_pos = torch.zeros(B, self.num_sink_tokens, dtype=torch.int32, device=q.device)
        real_kv_pos = kv_indices_full.to(torch.int32)
        real_full_kv_indices = torch.cat([real_sink_pos, real_kv_pos], dim=1)

        attn_out = sparse_query_sparse_key_attention(
            q_selected, k_t.contiguous(), v_t.contiguous(),
            real_q_indices, real_full_kv_indices,
        )
        out = torch.zeros_like(q_t)
        out.scatter_(1, idx_exp, attn_out)
        return self.o_proj(out.view(B, S, D))

    def _attention_flex(self, q, k, v, q_mask, kv_indices_full, B, S, D, S_kv_full, seq_len_offset):
        def mask_mod(b, h, q_idx, kv_idx):
            real_q_pos = q_idx + seq_len_offset
            real_kv_pos = kv_indices_full[b, torch.clamp(kv_idx - self.num_sink_tokens, min=0)]
            causal = real_q_pos >= real_kv_pos
            q_ok = q_mask[b, q_idx]
            return causal & q_ok

        block_mask = create_block_mask(
            mask_mod, B=B, H=1, Q_LEN=S,
            KV_LEN=S_kv_full + self.num_sink_tokens, device=q.device,
        )
        out = flex_attention(q, k, v, block_mask=block_mask,
                             enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))

    def _contextual(self, x_norm, router_out, past_key_values):
        """Run the contextual stream using its OWN Q/K/V projections."""
        B, S, _ = x_norm.shape
        D = self.dim

        q_raw = self.q_proj_ctx(x_norm)
        k_raw = self.k_proj_ctx(x_norm)
        v_raw = self.v_proj_ctx(x_norm)

        q = q_raw.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_raw.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_raw.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        seq_len_offset = 0
        if past_key_values is not None:
            past_k, _, _ = past_key_values
            seq_len_offset = past_k.shape[2]

        cos, sin = self.rope(q, S + seq_len_offset)
        cos_curr = cos[:, :, seq_len_offset:seq_len_offset + S, :]
        sin_curr = sin[:, :, seq_len_offset:seq_len_offset + S, :]
        q, k = apply_rotary_pos_emb(q, k, cos_curr, sin_curr)

        # ---- NSA triple-branch / ASA two-branch path ----
        if self.attention_kind in ("nsa", "asa"):
            # Expand GQA to full head count for the simple NSA reference.
            k_full = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v_full = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            nsa_out = self.nsa(q, k_full, v_full, select_indices=router_out.indices)
            # Project back to dim.
            attn_out = nsa_out.out.transpose(1, 2).contiguous().view(B, S, D)
            attn_out = self.o_proj(attn_out)
            # NSA does not maintain the v3-style sparse KV cache. For now we
            # return a no-op past-state tuple compatible with the sqsk path.
            return attn_out, (k, v, torch.arange(S, device=q.device).unsqueeze(0).expand(B, -1))

        kv_idx_exp = router_out.kv_indices.unsqueeze(1).unsqueeze(-1)\
            .expand(-1, self.num_kv_heads, -1, self.head_dim)
        k_sparse = torch.gather(k, 2, kv_idx_exp)
        v_sparse = torch.gather(v, 2, kv_idx_exp)

        if past_key_values is not None:
            past_k, past_v, past_kv_indices = past_key_values
            k_sparse_full = torch.cat([past_k, k_sparse], dim=2)
            v_sparse_full = torch.cat([past_v, v_sparse], dim=2)
            kv_indices_full = torch.cat(
                [past_kv_indices, router_out.kv_indices + seq_len_offset], dim=1)
        else:
            k_sparse_full = k_sparse
            v_sparse_full = v_sparse
            kv_indices_full = router_out.kv_indices

        new_past_key_values = (k_sparse_full, v_sparse_full, kv_indices_full)
        S_kv_full = k_sparse_full.shape[2]

        # Sinks use ctx K/V projections too - they are part of the contextual
        # attention apparatus, not a separate identity.
        sink = self.sink_tokens.expand(B, -1, -1)
        sink_norm = self.norm(sink)
        k_sink = self.k_proj_ctx(sink_norm).view(
            B, self.num_sink_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_sink = self.v_proj_ctx(sink_norm).view(
            B, self.num_sink_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k_full = torch.cat([k_sink, k_sparse_full], dim=2)
        v_full = torch.cat([v_sink, v_sparse_full], dim=2)

        # Build q_mask from router gate: any token with gate>0 is active.
        q_mask = router_out.gate > 0

        if HAS_TRITON_KERNELS and not self.training and q.is_cuda:
            attn = self._attention_triton(
                q, k_full, v_full, router_out.indices, kv_indices_full,
                B, S, D, seq_len_offset)
        else:
            attn = self._attention_flex(
                q, k_full, v_full, q_mask, kv_indices_full,
                B, S, D, S_kv_full, seq_len_offset)

        return attn, new_past_key_values

    def forward(self, x, past_key_values=None) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        router_out = self.router(x)
        x_norm = self.norm(x)

        # Reflexive stream: own Q/K/V over every token.
        q_refl = self.q_proj_refl(x_norm)
        k_refl = self.k_proj_refl(x_norm)
        v_refl = self.v_proj_refl(x_norm)
        reflexive_out = self.reflexive(q_refl, k_refl, v_refl)

        # Contextual stream: own Q/K/V over routed tokens.
        attn_out, new_past = self._contextual(x_norm, router_out, past_key_values)

        # Gate the contextual output by ReLU magnitudes - gradient flows back
        # to router via the gate values themselves, not through top-k.
        gate = router_out.gate.unsqueeze(-1).to(attn_out.dtype)
        attn_out = attn_out * gate

        return x + reflexive_out + attn_out, router_out.aux_loss, new_past
