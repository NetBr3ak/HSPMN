"""Mamba-3 SISO reflexive stream.

# variant-of: arXiv:2603.15569 (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu;
#             ICLR 2026 oral; Apache-2.0 reference release Mar 17 2026).

Mamba-3 over Mamba-2:
  1. Trapezoidal discretization (instead of ZOH) - second-order accurate in Δ.
  2. Complex-valued state via 2x2 block-diagonal A - recovers state-tracking
     expressivity (Proposition 4 of the paper); Mamba-2 was provably TC^0.
  3. SISO variant: one input channel per state head (this file). MIMO variant
     has higher prefill cost, deferred.

Recurrence (per head, per token t):
  Δ_t   = softplus(W_Δ x_t + b_Δ)            ∈ R_+
  A_t   = blkdiag2x2( -|a_re| ± i*a_im )      complex eigenvalue per pair
  Ā_t   = exp(Δ_t · A_t)                       trapezoidal: cleaner, see below
  B̄_t   = (Ā_t - I) / A_t  · B_t               trapezoidal mid-point
  S_t   = Ā_t · S_{t-1} + B̄_t · x_t
  y_t   = C_t · S_t + D · x_t

We implement complex pairs as real (re, im) tuples. Each "head" is N_state
2x2 rotation+scale blocks. State shape: [B, S, H, N_state, 2, D_head].

Drop-in contract for HSPMN v4 block:
  forward(q_raw, k_raw, v_raw) -> [B, S, dim]
  We treat v_raw as the input x (Mamba does not use Q/K). q_raw / k_raw are
  ignored - the dual-stream block still uses its own decoupled QKV for the
  contextual stream; the reflexive stream gets its own input here.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mamba3_siso_recurrence(
    x: torch.Tensor,        # [B, S, H, D_head]
    delta: torch.Tensor,    # [B, S, H]
    a_re: torch.Tensor,     # [H, N_state]   negative real part (decay)
    a_im: torch.Tensor,     # [H, N_state]   imaginary part (rotation)
    B_proj: torch.Tensor,   # [B, S, H, N_state]
    C_proj: torch.Tensor,   # [B, S, H, N_state]
    D_skip: torch.Tensor,   # [H, D_head]
) -> torch.Tensor:
    """Reference SISO recurrence with trapezoidal discretization.

    Each (re, im) eigenvalue pair drives a 2D state: state = (s_re, s_im).
    Update via complex multiplication unfolded into reals:
      Ā_re = exp(Δ * a_re) * cos(Δ * a_im)
      Ā_im = exp(Δ * a_re) * sin(Δ * a_im)
      [s_re_t]   [Ā_re  -Ā_im] [s_re_{t-1}]   [B̄_re]
      [s_im_t] = [Ā_im   Ā_re] [s_im_{t-1}] + [B̄_im] · x_t (broadcast over D)
    Output is C_proj · s_re (project complex state back to real).

    Slow: O(S) sequential. Fall back to chunked FLA Mamba-3 kernel if available.
    Used for diagnostics; production path goes through fla.ops.mamba3.
    """
    B, S, H, D_head = x.shape
    N_state = a_re.shape[1]
    device, dtype = x.device, torch.float32

    # State: [B, H, N_state, D_head, 2]
    s = torch.zeros(B, H, N_state, D_head, 2, device=device, dtype=dtype)
    out = torch.empty_like(x, dtype=x.dtype)

    a_re_neg = -F.softplus(a_re).float()  # ensure decay (negative real part)
    a_im_f = a_im.float()

    delta_f = delta.float()
    B_f = B_proj.float()
    C_f = C_proj.float()

    for t in range(S):
        d_t = delta_f[:, t]                           # [B, H]
        # Discretized eigenvalue contributions per head/state.
        # Δ_t · a_re ∈ [B, H, N_state]
        d_a_re = d_t.unsqueeze(-1) * a_re_neg.unsqueeze(0)
        d_a_im = d_t.unsqueeze(-1) * a_im_f.unsqueeze(0)
        amp = torch.exp(d_a_re)                       # [B, H, N_state]
        co = torch.cos(d_a_im)
        si = torch.sin(d_a_im)
        Abar_re = amp * co                             # [B, H, N_state]
        Abar_im = amp * si
        # Trapezoidal B̄ = (Ā - I) / A ·  in real form, complex divide.
        # |A|^2 = a_re^2 + a_im^2; (Ā - I)/A = ((Abar_re-1)+iAbar_im) / (a_re+ia_im)
        a_norm2 = (a_re_neg ** 2 + a_im_f ** 2).clamp(min=1e-6)  # [H, N_state]
        Bbar_re = ((Abar_re - 1.0) * a_re_neg.unsqueeze(0)
                   + Abar_im * a_im_f.unsqueeze(0)) / a_norm2.unsqueeze(0)
        Bbar_im = (Abar_im * a_re_neg.unsqueeze(0)
                   - (Abar_re - 1.0) * a_im_f.unsqueeze(0)) / a_norm2.unsqueeze(0)

        # Apply input projection B per (B, H, N_state).
        b_t = B_f[:, t]                                # [B, H, N_state]
        x_t = x[:, t].float()                          # [B, H, D_head]

        # state = Abar · state + Bbar · b_t · x_t
        # Broadcast: Abar_re [B, H, N_state, 1] over D_head.
        s_re_old = s[..., 0]                           # [B, H, N_state, D_head]
        s_im_old = s[..., 1]
        Abar_re_b = Abar_re.unsqueeze(-1)
        Abar_im_b = Abar_im.unsqueeze(-1)
        s_re_new = Abar_re_b * s_re_old - Abar_im_b * s_im_old
        s_im_new = Abar_im_b * s_re_old + Abar_re_b * s_im_old

        # Input drive: (Bbar_re + i Bbar_im) · b_t · x_t (real input).
        bx = b_t.unsqueeze(-1) * x_t.unsqueeze(2)      # [B, H, N_state, D_head]
        s_re_new = s_re_new + Bbar_re.unsqueeze(-1) * bx
        s_im_new = s_im_new + Bbar_im.unsqueeze(-1) * bx
        s = torch.stack([s_re_new, s_im_new], dim=-1)

        # Output: y = C · s_re + D · x  (project complex state to real via real part).
        c_t = C_f[:, t]                                # [B, H, N_state]
        y = (c_t.unsqueeze(-1) * s_re_new).sum(dim=2)  # [B, H, D_head]
        y = y + D_skip.unsqueeze(0) * x_t              # skip connection
        out[:, t] = y.to(out.dtype)

    return out


class Mamba3SISOReflexive(nn.Module):
    """Drop-in reflexive stream using Mamba-3 SISO + SwiGLU.

    Same contract as `GatedDeltaNetReflexive`:
      forward(q_raw, k_raw, v_raw) -> [B, S, dim]
    q_raw and k_raw are unused (Mamba is not attention). v_raw is treated as
    the input sequence; HSPMN block calls this with v_proj_refl(x_norm) so the
    decoupled-QKV defense still applies - Mamba-3 head sees its own projection.
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 mlp_ratio: int = 4, n_state: int = 16,
                 use_fla: Optional[bool] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_state = n_state

        # Attempt FLA Mamba-3 import (lands with FLA 0.5.3+).
        try:
            from fla.ops.mamba3 import chunk_mamba3_siso  # noqa: F401
            self.has_fla_mamba3 = True
        except ImportError:
            self.has_fla_mamba3 = False
        self.use_fla = (self.has_fla_mamba3 if use_fla is None
                        else (use_fla and self.has_fla_mamba3))

        # Per-head learned eigenvalues (re, im).
        self.a_re = nn.Parameter(torch.zeros(num_heads, n_state))
        self.a_im = nn.Parameter(torch.zeros(num_heads, n_state))
        nn.init.uniform_(self.a_re, 0.5, 2.0)   # softplus → positive decay
        nn.init.uniform_(self.a_im, -math.pi / 2, math.pi / 2)

        # Per-token Δ, B, C projections from input.
        self.delta_proj = nn.Linear(dim, num_heads, bias=True)
        self.B_proj = nn.Linear(dim, num_heads * n_state, bias=False)
        self.C_proj = nn.Linear(dim, num_heads * n_state, bias=False)

        # Skip term D (per head per channel).
        self.D_skip = nn.Parameter(torch.ones(num_heads, head_dim))

        # Δ bias init so softplus(Δ) starts ~1.0 (paper recommendation).
        nn.init.constant_(self.delta_proj.bias, math.log(math.e - 1))

        # SwiGLU MLP tail (matches GDN/ELU+1 reflexive variants).
        self.out_norm = nn.RMSNorm(dim)
        hidden = dim * mlp_ratio
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        for w in [self.B_proj, self.C_proj, self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(w.weight, gain=0.02)

    def forward(self, q_raw: torch.Tensor, k_raw: torch.Tensor, v_raw: torch.Tensor) -> torch.Tensor:
        # k_raw unused - Mamba is recurrent, not attentional.
        # q_raw is the full-dim projection (W_Q^refl x_norm) - used as the SSM
        #   input "x" and as input for Δ/B/C projections.
        # v_raw is GQA kv_dim (W_V^refl x_norm) - kept for API parity, ignored.
        # Decoupled-QKV defense: q_proj_refl is independent of contextual stream.
        B, S, _ = q_raw.shape
        H, D = self.num_heads, self.head_dim

        # Per-head input channels.
        x = q_raw.view(B, S, H, D)

        # Δ/B/C projected from the full-dim residual (q_raw == W_Q^refl x_norm).
        delta = F.softplus(self.delta_proj(q_raw))                    # [B, S, H]
        B_proj = self.B_proj(q_raw).view(B, S, H, self.n_state)
        C_proj = self.C_proj(q_raw).view(B, S, H, self.n_state)

        if self.use_fla and x.is_cuda:
            from fla.ops.mamba3 import chunk_mamba3_siso
            attn_flat = chunk_mamba3_siso(
                x, delta, self.a_re, self.a_im, B_proj, C_proj, self.D_skip,
            ).reshape(B, S, H * D)
        else:
            attn = _mamba3_siso_recurrence(
                x, delta, self.a_re, self.a_im, B_proj, C_proj, self.D_skip,
            )
            attn_flat = attn.reshape(B, S, H * D)

        # SwiGLU tail.
        n = self.out_norm(attn_flat)
        gate = self.gate_proj(n)
        gate = torch.nan_to_num(gate, nan=0.0).clamp(min=-1e4)
        return self.down_proj(F.silu(gate) * self.up_proj(n))
