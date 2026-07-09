"""
HSPMN v4.0 router - ReMoE-style ReLU gate with adaptive L1 sparsity control.

Replaces v3.0's sigmoid + 0.05-hard-threshold + dual-entropy router. Two
mechanisms work together:

1. **ReLU activation gate** (Wang/Zhu/Chen, ICLR 2025, arXiv:2412.14711).
   `g(x) = ReLU(W_g x + b)` - fully differentiable; tokens with gate=0 are
   inactive, gate>0 is the contextual-stream activation strength. Gradients
   flow through the gate at every nonzero value, which directly counters the
   "zero gradient through top-k" failure mode that the routing-absorption
   paper (Aquino-Michaels, arXiv:2603.02227) identified in v3.0.

2. **Adaptive L1 sparsity coefficient** that tracks a target activation
   fraction. The coefficient grows when too many tokens are active and
   shrinks when too few are, driving the natural ReLU sparsity toward the
   target without requiring a hyperparameter sweep.

3. **ALF-LB bias update** (DeepSeek-V3, arXiv:2412.19437). A registered
   buffer `route_bias` is added to the gate logits *only for selection*,
   not for the gate weight. This balances load across the batch without an
   auxiliary entropy loss.

4. **Router z-loss** (Zoph et al., ST-MoE, arXiv:2202.08906). Penalizes
   `mean(logsumexp(logits, dim=-1)^2)` to keep router logits from drifting
   to numerical extremes. Computed in fp32 before any nonlinearity.

The router exposes `top_k_indices()` for the contextual stream's static-shape
selection path (compatible with torch.compile fullgraph + CUDAGraphs), but
gradient flow is through the *gate values themselves*, not through the top-k
which has no gradient.
"""
from typing import NamedTuple

import torch
import torch.nn as nn


class RouterV4Output(NamedTuple):
    gate: torch.Tensor          # [B, S] - ReLU(W_g x + b), nonneg, gradient-carrying
    indices: torch.Tensor       # [B, K] - top-k token indices for contextual stream
    kv_indices: torch.Tensor    # [B, K_kv] - top-k + local-window KV indices
    aux_loss: torch.Tensor      # scalar - L1 penalty + z-loss + (optional) seq-balance
    active_fraction: torch.Tensor  # scalar - fraction of tokens with gate>0


class ReMoERouter(nn.Module):
    def __init__(
        self,
        dim: int,
        target_sparsity: float = 0.20,
        local_window: int = 64,
        l1_coef_init: float = 0.0,
        l1_adapt_rate: float = 0.0,
        z_loss_coef: float = 1e-4,
        bias_update_rate: float = 1e-3,
        seq_balance_coef: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(dim, 1, bias=True)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.02)
        nn.init.zeros_(self.gate_proj.bias)

        self.target_sparsity = float(target_sparsity)
        self.local_window = int(local_window)
        self.z_loss_coef = float(z_loss_coef)
        self.bias_update_rate = float(bias_update_rate)
        self.seq_balance_coef = float(seq_balance_coef)
        self.l1_adapt_rate = float(l1_adapt_rate)

        # ALF-LB bias: not in autograd graph, updated post-step.
        self.register_buffer("route_bias", torch.zeros(1))
        # Adaptive L1 coefficient - buffer so it persists across steps.
        self.register_buffer("l1_coef", torch.tensor(float(l1_coef_init)))

    @torch.no_grad()
    def _update_bias_and_l1(self, active_fraction: torch.Tensor):
        """Out-of-graph load-balance updates. Called only during training."""
        # ALF-LB: push selection toward target sparsity via bias sign.
        err = self.target_sparsity - active_fraction
        self.route_bias.add_(self.bias_update_rate * torch.sign(err))

        # Adaptive L1: if active fraction > target, raise the L1 penalty;
        # if below target, lower it. Multiplicative update keeps it positive.
        # `relative_excess` is in [-1, +inf): zero when on target, positive
        # when too active, negative when too sparse.
        relative_excess = (active_fraction - self.target_sparsity) / max(self.target_sparsity, 1e-6)
        self.l1_coef.mul_(torch.exp(self.l1_adapt_rate * relative_excess).clamp_(1e-6, 1e3))

    def forward(self, x: torch.Tensor) -> RouterV4Output:
        B, S, _ = x.shape

        # All router math in fp32 - sm_120 TF32 sigmoid pathology, plus
        # z-loss math wants high-precision logsumexp.
        x_fp32 = x.float()
        w = self.gate_proj.weight.float()
        b = self.gate_proj.bias.float() if self.gate_proj.bias is not None else None
        logits = torch.nn.functional.linear(x_fp32, w, b).squeeze(-1)  # [B, S]

        # ReLU gate: nonnegative, fully differentiable above zero.
        gate = torch.relu(logits)  # [B, S]
        active_mask = gate > 0
        active_fraction = active_mask.float().mean()

        # ---- Aux loss components (training only; trivial cost in eval) ----
        # L1 sparsity penalty on the gate magnitudes.
        l1_pen = gate.abs().mean()
        # Router z-loss (squared logsumexp). For a per-token scalar logit, this
        # reduces to mean(log(1 + exp(logit))^2 + ... ) - we emulate the ST-MoE
        # form by treating the per-token logit + 0 as a 2-class routing.
        # logsumexp([logit, 0]) = log(1 + exp(logit)) = softplus(logit).
        z = torch.nn.functional.softplus(logits).pow(2).mean()
        # Optional sequence-wise balance loss (DeepSeek-V3 §2.1.2 form).
        if self.seq_balance_coef > 0.0:
            per_seq_active = active_mask.float().mean(dim=1)  # [B]
            seq_balance = (per_seq_active - self.target_sparsity).pow(2).mean()
        else:
            seq_balance = logits.new_zeros(())

        # `l1_coef` is a buffer mutated out-of-graph by `_update_bias_and_l1`.
        # Read as a Python scalar (not a detached tensor) so the post-step
        # `mul_` cannot invalidate any saved tensor in the autograd graph.
        l1_scale = float(self.l1_coef)
        aux_loss = (
            l1_scale * l1_pen
            + self.z_loss_coef * z
            + self.seq_balance_coef * seq_balance
        )

        # ---- Selection (static shape, ALF-LB biased) ----
        # Bias is added ONLY to the selection score, never to the gate weight.
        # Read as a Python scalar (mutated out-of-graph after backward).
        select_score = logits + float(self.route_bias)

        k = max(1, int(S * self.target_sparsity))
        _, indices = torch.topk(select_score, k, dim=1, sorted=False)
        indices, _ = torch.sort(indices, dim=-1)
        indices = indices.contiguous()

        # KV indices: top-k + local window, with logit boost to guarantee
        # the most-recent W tokens are included (preserves locality).
        K_kv = min(S, k + self.local_window)
        kv_score = select_score.clone() if self.training else select_score
        if self.local_window > 0 and S >= self.local_window:
            kv_score = kv_score.clone() if not self.training else kv_score
            kv_score[:, -self.local_window:] = kv_score[:, -self.local_window:] + 1e4
        _, kv_indices = torch.topk(kv_score, K_kv, dim=1, sorted=False)
        kv_indices, _ = torch.sort(kv_indices, dim=-1)
        kv_indices = kv_indices.contiguous()

        if self.training:
            self._update_bias_and_l1(active_fraction.detach())

        return RouterV4Output(
            gate=gate,
            indices=indices,
            kv_indices=kv_indices,
            aux_loss=aux_loss,
            active_fraction=active_fraction.detach(),
        )
