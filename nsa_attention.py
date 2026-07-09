"""Native Sparse Attention (Yuan et al., DeepSeek, arXiv:2502.11089).

Three branches over the same Q/K/V, mixed by per-token sigmoid gates:

  1. Compress branch  - block-pooled K/V (block_size=32, stride=16). Cheap
     global summary; every query attends to every compressed block.
  2. Select branch    - top-N learned blocks of size `select_block_size=64`,
     selected per-query by importance scores derived from the compressed
     branch attention. This is the *hardware-aligned* sparse-attention path.
  3. Window branch    - sliding causal window of size `window_size=512`.
     Local context fine detail.

For HSPMN v4.0 the *select* branch reuses the ReMoE router's top-k indices
(Phase 1 reuse) - no separate selection scoring needed. The compress and
window branches are added as cheap globalish/localish coverage.

The three branch outputs are mixed by independent sigmoid gates (not a
3-way softmax - sigmoids let branches activate independently, matching the
ReMoE philosophy of "every gate is an on/off decision").

This is a PyTorch reference implementation. The DeepSeek paper ships a
Triton kernel for the select branch with arithmetic-intensity-balanced
tile shapes; we leave that as a Phase 5 optimization. At small/medium S
the PyTorch path is already reasonable because compress and window are
cuDNN-friendly, and select reuses indices from the existing top-k path.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NSAOutput(NamedTuple):
    out: torch.Tensor  # [B, S, H, D]
    gate_compress: torch.Tensor  # [B, S, H] mean over heads, or [B, S]
    gate_select: torch.Tensor
    gate_window: torch.Tensor


def _sliding_window_attention(q, k, v, window_size, scale):
    """Causal sliding-window attention on [B, H, S, D] tensors.

    For S * window_size memory budget, compute scores then mask to a
    causal window. PyTorch's SDPA does not expose sliding window directly,
    so we build the score matrix manually. At window=512 and S=4096 that
    is 4096 × 4096 = ~16M entries per head - fine for medium S.
    """
    B, H, S, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]
    pos = torch.arange(S, device=q.device)
    causal = pos[:, None] >= pos[None, :]
    in_window = (pos[:, None] - pos[None, :]) < window_size
    mask = causal & in_window  # [S, S]
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)  # [B, H, S, D]


def _compress_branch(
    q, k, v, block_size: int, stride: int, scale: float, return_scores: bool = False
):
    """Compressed K/V via mean-pooling blocks of size `block_size` with
    stride `stride`. Every query attends to the (causal subset of) compressed
    K/V positions.

    When return_scores=True, also returns a tuple of (out, attn_scores,
    block_starts, block_size) so callers can derive per-query block selection
    (used by NSA-select-from-compress).
    """
    B, H, S, D = q.shape
    if S < block_size:
        # Sequence too short to compress - return zero contribution.
        zero = torch.zeros_like(q)
        if return_scores:
            return zero, None, None, block_size
        return zero

    n_blocks = max(0, (S - block_size) // stride + 1)
    if n_blocks <= 0:
        zero = torch.zeros_like(q)
        if return_scores:
            return zero, None, None, block_size
        return zero

    starts = torch.arange(n_blocks, device=q.device) * stride
    block_idx = starts[:, None] + torch.arange(block_size, device=q.device)[None, :]
    block_idx = block_idx.clamp(max=S - 1)

    k_blocks = k[:, :, block_idx, :]
    v_blocks = v[:, :, block_idx, :]
    k_compressed = k_blocks.mean(dim=3)
    v_compressed = v_blocks.mean(dim=3)

    block_ends = starts + block_size - 1
    pos_q = torch.arange(S, device=q.device)
    can_see = pos_q[:, None] >= block_ends[None, :]

    scores = torch.matmul(q, k_compressed.transpose(-2, -1)) * scale
    scores_masked = scores.masked_fill(~can_see, float("-inf"))

    row_visible = can_see.any(dim=-1, keepdim=True)
    scores_masked = torch.where(
        row_visible, scores_masked, torch.zeros_like(scores_masked)
    )

    attn = F.softmax(scores_masked, dim=-1)
    attn = attn * row_visible.to(attn.dtype)

    out = torch.matmul(attn, v_compressed)
    if return_scores:
        # Return *pre-softmax* scores for downstream top-K selection; softmaxed
        # scores would collapse the per-query top-K to ~uniform after softmax.
        return out, scores_masked, starts, block_size
    return out


def _derive_select_indices_from_compress(
    scores, starts, block_size, n_select_blocks, pool_over_heads_and_queries=True
):
    """Per-batch top-N block selection from compress branch scores.

    Args:
        scores: [B, H, S, n_blocks] pre-softmax compress attention scores.
        starts: [n_blocks] block start positions in token space.
        block_size: block width in tokens.
        n_select_blocks: target K_block count.
        pool_over_heads_and_queries: if True (per-batch mode), pool scores over
            head + query dims → per-batch [B, n_blocks], top-K across blocks,
            then expand to token positions. If False (per-query, expensive),
            top-K per (B, H, S) and expand - not yet implemented here.

    Returns:
        [B, K_tokens] long tensor of selected token indices, sorted.
    """
    if scores is None:
        return None
    B, H, S, n_blocks = scores.shape
    if pool_over_heads_and_queries:
        # Replace -inf with very-negative finite to avoid NaN in mean.
        scores_safe = scores.masked_fill(scores == float("-inf"), -1e9)
        pooled = scores_safe.mean(dim=(1, 2))  # [B, n_blocks]
        K_block = min(n_select_blocks, n_blocks)
        _, top_blocks = torch.topk(pooled, K_block, dim=-1)  # [B, K_block]
        # Expand each block to its block_size tokens.
        offsets = torch.arange(block_size, device=scores.device)
        # block starts at starts[top_blocks]: [B, K_block]
        block_starts = starts[top_blocks]  # [B, K_block]
        token_idx = (
            block_starts.unsqueeze(-1)  # [B, K_block, block_size]
            + offsets.unsqueeze(0).unsqueeze(0)
        ).clamp(max=S - 1)
        token_idx = token_idx.view(B, -1)  # [B, K_block * block_size]
        # Unique + sort per batch.
        out = []
        for b in range(B):
            uniq = torch.unique(token_idx[b])
            out.append(uniq)
        # Pad to common length.
        max_K = max(o.numel() for o in out)
        padded = torch.zeros(B, max_K, dtype=torch.long, device=scores.device)
        for b, o in enumerate(out):
            padded[b, : o.numel()] = o
            if o.numel() < max_K:
                padded[b, o.numel() :] = o[-1]  # repeat last
        return padded
    raise NotImplementedError("per-query select_indices not wired yet")


def _select_branch(q, k, v, select_indices: Optional[torch.Tensor], scale: float):
    """Select branch: attention restricted to per-batch selected positions.

    `select_indices` has shape [B, K] (sorted, unique, in [0, S)). Same
    indices are used for all queries within a batch - coarse but correct.
    A more faithful NSA implementation chooses per-query top blocks; we
    use per-batch (cheaper, matches v3.0 style).

    If `select_indices` is None, returns zero (branch turned off).
    """
    if select_indices is None:
        return torch.zeros_like(q)
    B, H, S, D = q.shape
    idx = select_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D)
    k_sel = torch.gather(k, 2, idx)  # [B, H, K, D]
    v_sel = torch.gather(v, 2, idx)
    # Causal mask: query position must be >= selected position.
    pos_q = torch.arange(S, device=q.device)
    can_see = pos_q[:, None].unsqueeze(0) >= select_indices.unsqueeze(1)  # [B, S, K]
    can_see = can_see.unsqueeze(1)  # [B, 1, S, K]
    scores = torch.matmul(q, k_sel.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~can_see, float("-inf"))
    row_visible = can_see.any(dim=-1, keepdim=True)
    scores = torch.where(row_visible, scores, torch.zeros_like(scores))
    attn = F.softmax(scores, dim=-1)
    attn = attn * row_visible.to(attn.dtype)
    return torch.matmul(attn, v_sel)


class NSAAttention(nn.Module):
    """NSA triple-branch attention layer.

    Input contract (matches v4 contextual stream):
        q, k, v: [B, H, S, D] with H=num_heads (caller already expanded GQA)
        select_indices: optional [B, K] - top-k positions from the router.
                        If None, the select branch is skipped.

    Output: [B, H, S, D] mixed across the three branches.

    Gates are produced by a small Linear(H*D → 3*H) per token, applied
    per-head - every branch has its own mix weight per head per timestep.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        compress_block_size: int = 32,
        compress_stride: int = 16,
        window_size: int = 512,
        mode: str = "nsa",
        layer_idx: int = 0,
        select_from_compress: bool = False,
        n_select_blocks: int = 8,
    ):
        """`mode`: 'nsa' (default, all 3 branches every layer) or 'asa'
        (drop compress; alternate select/sliding per-layer per arXiv:2511.00819).
        ASA evidence: VideoNSA + ASA papers show compress is FLOPs bottleneck
        with low marginal value; sliding dominates common-sense, select
        dominates long-context."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compress_block_size = compress_block_size
        self.compress_stride = compress_stride
        self.window_size = window_size
        if mode not in ("nsa", "asa"):
            raise ValueError(f"mode must be 'nsa' or 'asa', got {mode!r}")
        self.mode = mode
        self.layer_idx = layer_idx
        self.select_from_compress = bool(select_from_compress)
        self.n_select_blocks = int(n_select_blocks)
        # ASA layer-alternation: even → sliding-only, odd → select-only.
        self.asa_branch = (
            ("window" if (layer_idx % 2 == 0) else "select") if mode == "asa" else None
        )

        # Gates: per-token, per-head. NSA = 3 outputs; ASA = 1 (only the
        # active branch per layer, kept for output magnitude calibration).
        n_branches = 3 if mode == "nsa" else 1
        self.gate_proj = nn.Linear(
            num_heads * head_dim, n_branches * num_heads, bias=True
        )
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.02)
        with torch.no_grad():
            self.gate_proj.bias.zero_()
            if mode == "nsa":
                # last num_heads slots = window gate; bias toward window branch.
                self.gate_proj.bias[-num_heads:].fill_(1.0)
            else:
                # ASA: single branch, bias to ~0.7 (sigmoid(1.0) ≈ 0.73).
                self.gate_proj.bias.fill_(1.0)

    def forward(
        self, q, k, v, select_indices: Optional[torch.Tensor] = None
    ) -> NSAOutput:
        B, H, S, D = q.shape
        scale = 1.0 / (D**0.5)
        zeros = q.new_zeros((B, H, S))  # placeholder gate when branch off

        q_flat = q.transpose(1, 2).reshape(B, S, H * D)

        if self.mode == "asa":
            # ASA: single branch per layer, alternating across depth.
            gate = torch.sigmoid(self.gate_proj(q_flat))  # [B, S, H]
            g = gate.transpose(1, 2).unsqueeze(-1)  # [B, H, S, 1]
            if self.asa_branch == "window":
                out_w = _sliding_window_attention(q, k, v, self.window_size, scale)
                out = g * out_w
                return NSAOutput(
                    out=out,
                    gate_compress=zeros,
                    gate_select=zeros,
                    gate_window=g.squeeze(-1).mean(dim=1),
                )
            else:  # select
                out_s = _select_branch(q, k, v, select_indices, scale)
                out = g * out_s
                return NSAOutput(
                    out=out,
                    gate_compress=zeros,
                    gate_select=g.squeeze(-1).mean(dim=1),
                    gate_window=zeros,
                )

        # NSA full triple-branch. When select_from_compress and no external
        # select_indices given, derive per-batch top-K block indices from the
        # compress branch's attention scores (parameter-free selection head).
        if self.select_from_compress and select_indices is None:
            out_c, c_scores, c_starts, c_block = _compress_branch(
                q,
                k,
                v,
                self.compress_block_size,
                self.compress_stride,
                scale,
                return_scores=True,
            )
            derived_idx = _derive_select_indices_from_compress(
                c_scores, c_starts, c_block, self.n_select_blocks
            )
            out_s = _select_branch(q, k, v, derived_idx, scale)
        else:
            out_c = _compress_branch(
                q, k, v, self.compress_block_size, self.compress_stride, scale
            )
            out_s = _select_branch(q, k, v, select_indices, scale)
        out_w = _sliding_window_attention(q, k, v, self.window_size, scale)

        gates = torch.sigmoid(self.gate_proj(q_flat))  # [B, S, 3*H]
        g_c, g_s, g_w = gates.split(H, dim=-1)
        g_c = g_c.transpose(1, 2).unsqueeze(-1)
        g_s = g_s.transpose(1, 2).unsqueeze(-1)
        g_w = g_w.transpose(1, 2).unsqueeze(-1)
        out = g_c * out_c + g_s * out_s + g_w * out_w
        return NSAOutput(
            out=out,
            gate_compress=g_c.squeeze(-1).mean(dim=1),
            gate_select=g_s.squeeze(-1).mean(dim=1),
            gate_window=g_w.squeeze(-1).mean(dim=1),
        )
