"""Gated DeltaNet (Yang/Kautz/Hatamizadeh, ICLR 2025, arXiv:2412.06464).

Pure-PyTorch implementation of the chunkwise parallel form. Slower than the
official FLA Triton kernels (`fla-org/flash-linear-attention`), but works
without any external dependency - ships in a single file.

Recurrence:
  S_t = (α_t · I  −  β_t · k_t k_t^T) · S_{t-1}  +  β_t · v_t k_t^T
  o_t = q_t · S_t

This unifies Mamba-2's gating (α: forget rate per timestep) with the delta
rule (β: targeted update). At α=1, β=0 it's identity recurrence; at α=1,
β=1 it's the unmodified delta rule; with both data-dependent it's the full
Gated DeltaNet.

For HSPMN v4.0 it replaces ELU+1 chunked linear attention in the Reflexive
Stream. Theoretical wins (vs ELU+1):
- recognizes regular languages beyond TC^0 (Yang et al. §3, RWKV-7 §K);
- no 1/Σ normalization → no instability;
- gradient flows through the gates α and β cleanly;
- chunkwise parallel form keeps GPU utilization.

This file ALSO ships a thin `GatedDeltaNetReflexive` wrapper that drops in
where `LinearStateSpaceStream` lives in the v4 block - same input/output
contract: `forward(q_raw, k_raw, v_raw) -> [B, S, dim]`, with shared QKV
projections from the surrounding block. SwiGLU MLP is kept identical.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    HAS_FLA = True
except ImportError:
    HAS_FLA = False


def _chunked_gated_delta(
    q: torch.Tensor,   # [B, S, H, D]
    k: torch.Tensor,   # [B, S, H, D]
    v: torch.Tensor,   # [B, S, H, D]
    alpha: torch.Tensor,  # [B, S, H] - forget gate, in (0, 1]
    beta: torch.Tensor,   # [B, S, H] - write rate, in (0, 1]
    chunk_size: int = 64,
) -> torch.Tensor:
    """Chunkwise parallel Gated DeltaNet.

    Inside each chunk we run the recurrence as a Python loop (cheap, chunk_size
    is small). Across chunks we keep a running matrix state in fp32 and pass
    it forward. This matches the FLA implementation's mathematical form.
    """
    B, S, H, D = q.shape
    state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    out = torch.empty_like(q, dtype=q.dtype)

    # Promote gates to fp32 for stability on long sequences.
    alpha_f32 = alpha.float()
    beta_f32 = beta.float()

    for cs in range(0, S, chunk_size):
        ce = min(cs + chunk_size, S)
        for t in range(cs, ce):
            q_t = q[:, t].float()        # [B, H, D]
            k_t = k[:, t].float()        # [B, H, D]
            v_t = v[:, t].float()        # [B, H, D]
            a_t = alpha_f32[:, t]        # [B, H]
            b_t = beta_f32[:, t]         # [B, H]

            # state shape: [B, H, D, D]; broadcast a_t and b_t over (D, D).
            a_view = a_t.unsqueeze(-1).unsqueeze(-1)        # [B, H, 1, 1]
            b_view = b_t.unsqueeze(-1).unsqueeze(-1)        # [B, H, 1, 1]

            # k_t outer k_t - only one row of state needs adjusting per token.
            kkT = torch.einsum("bhd,bhe->bhde", k_t, k_t)   # [B, H, D, D]
            update = b_view * torch.einsum("bhde,bhef->bhdf", kkT, state)
            forget = a_view * state
            state = forget - update + b_view * torch.einsum("bhd,bhe->bhde", v_t, k_t)

            # Output: q_t · state, with last D the value-output dim.
            o_t = torch.einsum("bhd,bhde->bhe", q_t, state)
            out[:, t] = o_t.to(out.dtype)

    return out


class GatedDeltaNetReflexive(nn.Module):
    """Drop-in reflexive stream using Gated DeltaNet + SwiGLU.

    Same forward contract as `LinearStateSpaceStream`:
        forward(q_raw, k_raw, v_raw) -> output of shape [B, S, dim]

    Internally:
    - Reshape Q/K/V to [B, S, H, D].
    - Compute α, β gates from a small per-timestep linear projection
      from the input residual (or in our case, K - cheaper, no extra
      input plumbing). Gates are scalar per (B, S, H) - one number per
      head per timestep.
    - Run chunkwise Gated DeltaNet recurrence to get attention output.
    - Apply RMSNorm + SwiGLU MLP.
    - Return [B, S, dim].
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 mlp_ratio: int = 4, chunk_size: int = 64,
                 use_fla: Optional[bool] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.chunk_size = chunk_size
        # Auto-pick FLA kernel when available; fall back to PyTorch reference.
        # `use_fla=False` forces the reference path (used in unit tests).
        self.use_fla = HAS_FLA if use_fla is None else (use_fla and HAS_FLA)

        # Gates (α, β) per query head, computed from K (one number per token per head).
        # Sigmoid → (0, 1). Initialized so α≈1, β≈small to start near identity recurrence.
        self.gate_alpha = nn.Linear(num_kv_heads * head_dim, num_heads, bias=True)
        self.gate_beta = nn.Linear(num_kv_heads * head_dim, num_heads, bias=True)
        nn.init.xavier_uniform_(self.gate_alpha.weight, gain=0.02)
        nn.init.xavier_uniform_(self.gate_beta.weight, gain=0.02)
        nn.init.constant_(self.gate_alpha.bias, 4.0)   # sigmoid(4) ≈ 0.98
        nn.init.constant_(self.gate_beta.bias, -2.0)   # sigmoid(-2) ≈ 0.12

        self.out_norm = nn.RMSNorm(dim)
        hidden = dim * mlp_ratio
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        for w in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(w.weight, gain=0.02)

    def forward(self, q_raw: torch.Tensor, k_raw: torch.Tensor, v_raw: torch.Tensor) -> torch.Tensor:
        B, S, _ = q_raw.shape
        H, D = self.num_heads, self.head_dim

        # Compute gates from K (avoid plumbing extra inputs).
        alpha = torch.sigmoid(self.gate_alpha(k_raw))   # [B, S, H], in (0, 1)
        beta = torch.sigmoid(self.gate_beta(k_raw))     # [B, S, H]

        # Reshape Q/K/V to [B, S, H, D]; expand KV heads to match query heads (GQA).
        q = q_raw.view(B, S, H, D)
        k = k_raw.view(B, S, self.num_kv_heads, D)
        v = v_raw.view(B, S, self.num_kv_heads, D)
        k = k.repeat_interleave(self.kv_groups, dim=2)   # [B, S, H, D]
        v = v.repeat_interleave(self.kv_groups, dim=2)   # [B, S, H, D]

        if self.use_fla and q.is_cuda:
            # FLA expects gates as log(forget rate); we compute alpha as
            # sigmoid → in (0,1), so log(alpha) ∈ (-∞, 0].
            g_log = torch.log(alpha.float().clamp(min=1e-6))
            # FLA kernel handles SiLU+L2 norm internally when requested.
            attn, _ = chunk_gated_delta_rule(
                q, k, v, g_log, beta,
                use_qk_l2norm_in_kernel=True,
                output_final_state=False,
            )
        else:
            # Reference path: SiLU+L2 norm in PyTorch, then chunked recurrence.
            k = F.silu(k)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            attn = _chunked_gated_delta(q, k, v, alpha, beta, chunk_size=self.chunk_size)
        attn_flat = attn.reshape(B, S, H * D)             # [B, S, dim]

        # SwiGLU MLP, identical to the v3 reflexive stream's tail.
        n = self.out_norm(attn_flat)
        gate = self.gate_proj(n)
        gate = torch.nan_to_num(gate, nan=0.0).clamp(min=-1e4)  # F.silu NaN guard
        return self.down_proj(F.silu(gate) * self.up_proj(n))
