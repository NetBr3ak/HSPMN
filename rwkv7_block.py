"""RWKV-7 "Goose" reflexive stream.

# variant-of: arXiv:2503.14456 (Peng et al., March 2025).

RWKV-7 over RWKV-6 / Gated DeltaNet:
  1. Generalized delta rule: state update is a learnable rank-2 (vs rank-1 in
     GDN) modification, S_t = S_{t-1} (I − k_t a_t^T) + v_t k_t^T.
     a_t and k_t are independent ⇒ removal vector ≠ write key.
  2. Vector-valued gating w_t (per-head), not scalar - enables higher-rank state
     transitions.
  3. In-context learning rate: explicit data-dependent η_t scales the write.
  4. Provably recognizes all regular languages (Theorem 1, Appendix K) ⇒
     strictly beyond TC^0 expressivity.

Recurrence (per head, per token t):
  w_t  = data-dependent gate vector (per channel decay), in (0, 1]^D
  k_t  = key (write address)
  a_t  = removal-from-state vector (independent of k_t)  - NEW vs GDN
  v_t  = value
  η_t  = scalar in-context learning rate (per-head)
  S_t  = S_{t-1} ⊙ w_t  −  η_t · S_{t-1} (k_t a_t^T)  +  η_t · v_t k_t^T
  o_t  = S_t · q_t                       (q_t is the read query)

Drop-in for HSPMN v4 reflexive stream: forward(q_raw, k_raw, v_raw) → [B, S, dim].
We construct k, a, v, q from the q_raw / k_raw / v_raw inputs (decoupled-QKV
defense preserved - these are W_Q^refl, W_K^refl, W_V^refl already).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.rwkv7 import chunk_rwkv7

    HAS_FLA_RWKV7 = True
except ImportError:
    HAS_FLA_RWKV7 = False


def _rwkv7_recurrence(
    q: torch.Tensor,  # [B, S, H, D]   read query
    k: torch.Tensor,  # [B, S, H, D]   write key
    a: torch.Tensor,  # [B, S, H, D]   removal-from-state vector (NEW vs GDN)
    v: torch.Tensor,  # [B, S, H, D]   value
    w: torch.Tensor,  # [B, S, H, D]   per-channel decay in (0, 1]
    eta: torch.Tensor,  # [B, S, H]      per-head in-context learning rate
) -> torch.Tensor:
    """Reference generalized-delta-rule recurrence. Slow O(S); diagnostic only.

    state shape: [B, H, D, D]. Output: [B, S, H, D].
    """
    B, S, H, D = q.shape
    state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    out = torch.empty_like(q, dtype=q.dtype)

    for t in range(S):
        q_t = q[:, t].float()  # [B, H, D]
        k_t = k[:, t].float()
        a_t = a[:, t].float()
        v_t = v[:, t].float()
        w_t = w[:, t].float()
        e_t = eta[:, t].float()  # [B, H]

        # Per-channel decay: state ⊙ w_t (broadcast over second D axis).
        state = state * w_t.unsqueeze(-1)  # [B, H, D, D]

        # Generalized delta: state -= η · state · (k_t a_t^T)
        # = η · einsum("bhde,bhe,bhf->bhdf", state, k_t, a_t)
        # First state_k = state · k_t  : [B, H, D]
        state_k = torch.einsum("bhde,bhe->bhd", state, k_t)
        # Outer with a_t scaled by η.
        e_view = e_t.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        state = state - e_view * torch.einsum("bhd,bhe->bhde", state_k, a_t)

        # Add new write: η · v_t · k_t^T
        state = state + e_view * torch.einsum("bhd,bhe->bhde", v_t, k_t)

        # Read: o_t = state · q_t (last D axis).
        o_t = torch.einsum("bhde,bhe->bhd", state, q_t)
        out[:, t] = o_t.to(out.dtype)

    return out


class RWKV7Reflexive(nn.Module):
    """Drop-in reflexive stream using RWKV-7 generalized delta rule.

    Forward contract (matches GatedDeltaNetReflexive):
      forward(q_raw, k_raw, v_raw) -> [B, S, dim]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mlp_ratio: int = 4,
        use_fla: Optional[bool] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.use_fla = HAS_FLA_RWKV7 if use_fla is None else (use_fla and HAS_FLA_RWKV7)

        # Per-token gates: w (per-channel log decay) and η (per-head ICLR scalar).
        # w_proj output → -softplus → log decay ∈ (-∞, 0] (FLA convention).
        self.w_proj = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.eta_proj = nn.Linear(dim, num_heads, bias=True)
        # Removal direction a: separate projection (NEW in RWKV-7 vs GDN).
        self.a_proj = nn.Linear(dim, num_heads * head_dim, bias=False)

        # Init w_proj bias = -3.0 → softplus(-3.0) ≈ 0.049 → log_decay ≈ -0.049
        # → decay ≈ 0.95 (slow forgetting, matches FLA RWKV-7 default).
        # η near 0.5 (moderate write rate), a small (start near identity).
        nn.init.constant_(self.w_proj.bias, -3.0)
        nn.init.constant_(self.eta_proj.bias, 0.0)  # sigmoid(0) = 0.5
        nn.init.xavier_uniform_(self.w_proj.weight, gain=0.02)
        nn.init.xavier_uniform_(self.eta_proj.weight, gain=0.02)
        nn.init.xavier_uniform_(self.a_proj.weight, gain=0.02)

        # SwiGLU MLP tail (matches other reflexive variants).
        self.out_norm = nn.RMSNorm(dim)
        hidden = dim * mlp_ratio
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        for p in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(p.weight, gain=0.02)

    def forward(
        self, q_raw: torch.Tensor, k_raw: torch.Tensor, v_raw: torch.Tensor
    ) -> torch.Tensor:
        B, S, _ = q_raw.shape
        H, D = self.num_heads, self.head_dim

        # Reshape Q to [B, S, H, D]; expand GQA K/V to full head count.
        q = q_raw.view(B, S, H, D)
        k = k_raw.view(B, S, self.num_kv_heads, D).repeat_interleave(
            self.kv_groups, dim=2
        )
        v = v_raw.view(B, S, self.num_kv_heads, D).repeat_interleave(
            self.kv_groups, dim=2
        )

        # Project gates + removal vector from full-dim residual.
        # FLA convention: w = log decay ∈ (-∞, 0], realized via -softplus.
        w_log = -F.softplus(self.w_proj(q_raw)).view(B, S, H, D)
        eta = torch.sigmoid(self.eta_proj(q_raw))  # [B, S, H]
        a_dir = self.a_proj(q_raw).view(B, S, H, D)
        a_dir = F.normalize(a_dir, dim=-1)  # unit removal direction

        if self.use_fla and q.is_cuda:
            # FLA chunk_rwkv7 expects DPLR delta-rule form:
            #   r=q, w=log_decay, k, v, a (rank-1 left), b (rank-1 right).
            # Convention from fla.layers.rwkv7: a = -kk_dir, b = kk_dir * iclr.
            # eta (per-head) broadcasts to per-channel via unsqueeze(-1).
            eta_bcast = eta.unsqueeze(-1)  # [B, S, H, 1]
            a_arg = -a_dir
            b_arg = a_dir * eta_bcast
            # FLA chunk kernel rejects fp32 inputs on some platforms; cast bf16.
            target_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
            r_in = q.to(target_dtype)
            k_in = k.to(target_dtype)
            v_in = v.to(target_dtype)
            w_in = w_log.to(target_dtype)
            a_in = a_arg.to(target_dtype)
            b_in = b_arg.to(target_dtype)
            attn, _ = chunk_rwkv7(r=r_in, w=w_in, k=k_in, v=v_in, a=a_in, b=b_in)
            attn = attn.to(q.dtype)
        else:
            # Reference path: scalar-eta, sigmoid-decay, original recurrence form.
            w_decay = torch.exp(w_log)  # ∈ (0, 1]
            attn = _rwkv7_recurrence(q, k, a_dir, v, w_decay, eta)
        attn_flat = attn.reshape(B, S, H * D)

        # SwiGLU tail.
        n = self.out_norm(attn_flat)
        gate = self.gate_proj(n)
        gate = torch.nan_to_num(gate, nan=0.0).clamp(min=-1e4)
        return self.down_proj(F.silu(gate) * self.up_proj(n))
