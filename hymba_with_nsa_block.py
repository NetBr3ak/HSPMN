"""Hymba block with NSA triple-branch in place of dense causal attention.

Ablation rationale (P2-screen, 2026-05): isolate fusion mechanism from
sparse-attention contribution. Stock Hymba uses dense SDPA in its attention
heads. Stock HSPMN v4-gdn-nsa uses NSA + GDN with block-level dual-stream
fusion. This variant uses NSA + GDN with Hymba's head-level (mean-fusion)
design. Difference vs v4-gdn-nsa = fusion mechanism only. Difference vs
hymba = attention primitive only.

Note on NSA select branch: stock NSAAttention takes external select_indices
from the v4 router. Hymba has no router, so the select branch is left off
(passed select_indices=None → returns zero). The active branches are
compress + window. This is documented as 'NSA-no-select' in the run report.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gated_deltanet import GatedDeltaNetReflexive
from hspmn_v3_0 import RotaryEmbedding, apply_rotary_pos_emb
from nsa_attention import NSAAttention


class HymbaWithNSABlock(nn.Module):
    """Parallel NSA + GDN-SSM heads, mean-fused + SwiGLU MLP tail."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mlp_ratio: int = 4,
        max_seq_len: int = 2048,
        rope_base: int = 10000,
        attn_frac: float = 0.5,
        num_meta_tokens: int = 64,
        nsa_window_size: int = 256,
        nsa_compress_block: int = 32,
        nsa_compress_stride: int = 16,
        layer_idx: int = 0,
        random_gate: bool = False,
        attn_mode: str = "nsa",
        use_attn_gate: bool = False,
        gate_mode: str = "linear",
        pc_temperature: float = 1.0,
        nsa_select_from_compress: bool = False,
        nsa_n_select_blocks: int = 8,
        stream_decor: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.kv_groups = num_heads // num_kv_heads
        if attn_mode not in ("nsa", "asa"):
            raise ValueError(f"attn_mode must be 'nsa'|'asa', got {attn_mode!r}")
        if gate_mode not in ("linear", "predictive_coding"):
            raise ValueError(
                f"gate_mode must be 'linear'|'predictive_coding', got {gate_mode!r}"
            )
        self.attn_mode = attn_mode
        self.random_gate = bool(random_gate)
        self.gate_mode = gate_mode
        self.pc_temperature = float(pc_temperature)
        # v6: stream-decorrelation regularizer. The gate-ceiling finding says
        # routing fails because the two streams are substitutable per token;
        # this penalty forces them to carry decorrelated features so that
        # "which stream helps" can become predictable (raises the ceiling).
        self.stream_decor = bool(stream_decor)
        # random_gate / use_attn_gate / pc gate_mode all imply the attn gate is present.
        # Bare hymba-with-nsa (P2 winner) is gate-less for backward compatibility
        # with the May-2026 sweep checkpoints.
        self.use_attn_gate = bool(
            use_attn_gate or random_gate or gate_mode == "predictive_coding"
        )

        self.n_attn_heads = max(1, int(num_heads * attn_frac))
        self.n_ssm_heads = num_heads - self.n_attn_heads

        self.norm = nn.RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_base)

        self.nsa = NSAAttention(
            num_heads=self.n_attn_heads,
            head_dim=head_dim,
            compress_block_size=nsa_compress_block,
            compress_stride=nsa_compress_stride,
            window_size=nsa_window_size,
            mode=attn_mode,
            layer_idx=layer_idx,
            select_from_compress=nsa_select_from_compress,
            n_select_blocks=nsa_n_select_blocks,
        )

        # Per-token sigmoid gate over the NSA attention output, analogous to
        # the v4 ReMoE-gate-times-ctx mechanism. Bare hymba-with-nsa (P2 winner)
        # is gate-less; gate present only when use_attn_gate or random_gate or
        # gate_mode == 'predictive_coding'.
        if self.use_attn_gate and self.gate_mode == "linear":
            self.attn_gate_proj = nn.Linear(dim, self.n_attn_heads, bias=True)
            nn.init.xavier_uniform_(self.attn_gate_proj.weight, gain=0.02)
            nn.init.zeros_(self.attn_gate_proj.bias)
            if self.random_gate:
                # Aquino-Michaels random-gate baseline (arXiv:2603.02227).
                # Freeze at N(0, std=0.02) init - sigmoid keeps mid-range mean.
                nn.init.normal_(self.attn_gate_proj.weight, std=0.02)
                self.attn_gate_proj.weight.requires_grad_(False)
                self.attn_gate_proj.bias.requires_grad_(False)
        else:
            self.attn_gate_proj = None
        # Predictive-coding gate is parameter-free; only stores the temperature
        # buffer (immutable). The gate signal is derived in forward() from the
        # SSM head's per-token output magnitude.

        if self.n_ssm_heads > 0:
            ssm_dim = self.n_ssm_heads * head_dim
            ssm_kv_heads = max(1, num_kv_heads * self.n_ssm_heads // num_heads)
            self.ssm = GatedDeltaNetReflexive(
                dim=ssm_dim,
                num_heads=self.n_ssm_heads,
                num_kv_heads=ssm_kv_heads,
                head_dim=head_dim,
                mlp_ratio=1,
                use_fla=None,
            )
            self.ssm_in_proj = nn.Linear(dim, ssm_dim, bias=False)
            self.ssm_k_proj = nn.Linear(dim, ssm_kv_heads * head_dim, bias=False)
            self.ssm_v_proj = nn.Linear(dim, ssm_kv_heads * head_dim, bias=False)
        else:
            self.ssm = None

        self.num_meta_tokens = num_meta_tokens
        self.meta_tokens = nn.Parameter(torch.zeros(1, num_meta_tokens, dim))
        nn.init.normal_(self.meta_tokens, std=0.02)

        self.norm2 = nn.RMSNorm(dim)
        hidden = dim * mlp_ratio
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.dim)
        for m in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ]:
            nn.init.xavier_uniform_(m.weight, gain=scale)
        if self.ssm is not None:
            for m in [self.ssm_in_proj, self.ssm_k_proj, self.ssm_v_proj]:
                nn.init.xavier_uniform_(m.weight, gain=scale)

    def forward(
        self, x: torch.Tensor, past_key_values=None
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        B, S, D = x.shape
        x_norm = self.norm(x)

        q = (
            self.q_proj(x_norm)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x_norm)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x_norm)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos, sin = self.rope(q, S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k_exp = k.repeat_interleave(self.kv_groups, dim=1)
        v_exp = v.repeat_interleave(self.kv_groups, dim=1)

        q_a = q[:, : self.n_attn_heads]
        k_a = k_exp[:, : self.n_attn_heads]
        v_a = v_exp[:, : self.n_attn_heads]
        nsa_out = self.nsa(q_a, k_a, v_a, select_indices=None)
        attn_out = nsa_out.out  # [B, n_attn_heads, S, head_dim]

        # Compute SSM output FIRST when present - needed both for the
        # head-concat and for the predictive-coding gate (when in use).
        ssm_out_view = None
        if self.ssm is not None:
            ssm_q = self.ssm_in_proj(x_norm)
            ssm_k = self.ssm_k_proj(x_norm)
            ssm_v = self.ssm_v_proj(x_norm)
            ssm_out = self.ssm(ssm_q, ssm_k, ssm_v)
            ssm_out_view = ssm_out.view(
                B, S, self.n_ssm_heads, self.head_dim
            ).transpose(1, 2)

        # Optional per-token sigmoid gate over the NSA contribution.
        if self.attn_gate_proj is not None:
            # Linear (learned or random-frozen) gate path.
            x_for_gate = x_norm.to(self.attn_gate_proj.weight.dtype)
            gate_logits = self.attn_gate_proj(x_for_gate)
            g = torch.sigmoid(gate_logits).to(attn_out.dtype)
            g = g.transpose(1, 2).unsqueeze(-1)  # [B, n_attn, S, 1]
            attn_out = attn_out * g
            self._last_gate = g.detach()
        elif self.gate_mode == "predictive_coding" and ssm_out_view is not None:
            # Predictive-coding gate: parameter-free, derives from the SSM
            # reflexive stream's per-token output magnitude. Low norm ≈
            # uncertain reflexive prediction → gate NSA ON; high norm ≈
            # confident → gate OFF. We z-score the per-token SSM norm
            # against the batch's own (mean, std), so the gate signal is
            # scale-invariant regardless of pre-/post-training magnitude.
            ssm_norm = ssm_out_view.norm(dim=-1)  # [B, n_ssm_heads, S]
            ssm_norm_pooled = ssm_norm.mean(dim=1, keepdim=True)  # [B, 1, S]
            tau = ssm_norm_pooled.mean(dim=-1, keepdim=True)  # [B, 1, 1]
            sigma = ssm_norm_pooled.std(dim=-1, keepdim=True).clamp_min(1e-6)
            pc_signal = (tau - ssm_norm_pooled) / (self.pc_temperature * sigma)
            g = torch.sigmoid(pc_signal).to(attn_out.dtype)
            g = g.expand(-1, self.n_attn_heads, -1).unsqueeze(-1)
            attn_out = attn_out * g
            self._last_gate = g.detach()

        attn_full = torch.zeros(
            B, self.num_heads, S, self.head_dim, device=x.device, dtype=x.dtype
        )
        attn_full[:, : self.n_attn_heads] = attn_out
        if ssm_out_view is not None:
            attn_full[:, self.n_attn_heads :] = ssm_out_view

        # v6 stream decorrelation: normalized cross-covariance between the
        # token-level features of the two head groups (Barlow-Twins-style
        # cross term). Computed in fp32; scalar per layer; the LM wrapper
        # scales it by decor_coef and adds it to the loss.
        aux = x.new_zeros(())
        if self.stream_decor and self.training and ssm_out_view is not None:
            A = attn_out.transpose(1, 2).reshape(B * S, -1).float()
            R = ssm_out_view.transpose(1, 2).reshape(B * S, -1).float()
            A = A - A.mean(0, keepdim=True)
            R = R - R.mean(0, keepdim=True)
            A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-6)
            R = R / R.norm(dim=0, keepdim=True).clamp_min(1e-6)
            cross = A.t() @ R  # [Da, Dr] correlations
            aux = cross.pow(2).mean().to(x.dtype)

        merged = attn_full.transpose(1, 2).contiguous().view(B, S, D)
        h = x + self.o_proj(merged)

        n2 = self.norm2(h)
        out = h + self.down_proj(F.silu(self.gate_proj(n2)) * self.up_proj(n2))

        return out, aux, (None, None, None)
