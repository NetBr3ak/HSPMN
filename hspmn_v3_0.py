"""
Proprietary / All Rights Reserved - Non-Commercial Use Only
Source-available for portfolio viewing only. Commercial use, unauthorized modification, reproduction, or distribution is strictly prohibited. All rights reserved.
"""

import math
from typing import Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    from kernels_v3_0 import sparse_query_sparse_key_attention

    HAS_TRITON_KERNELS = True
except ImportError:
    HAS_TRITON_KERNELS = False

from utils_v3_0 import HSPMNConfig

__all__ = ["HSPMNBlock", "TopKRouter", "HSPMNConfig"]


class RouterOutput(NamedTuple):
    """Output from the TopKRouter."""

    mask: torch.Tensor
    indices: torch.Tensor
    kv_mask: torch.Tensor
    kv_indices: torch.Tensor
    probs: torch.Tensor
    aux_loss: torch.Tensor


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Inverse frequency Rotary Embedding optimized for torch.compile."""

    def __init__(self, dim: int, max_len: int = 131072, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cached_cos", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "cached_sin", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x: torch.Tensor, seq_len: int):
        # Dynamic cache resizing if needed
        if seq_len > self.cached_cos.shape[2]:
            self._build_cache(seq_len)
        cos = self.cached_cos[:, :, :seq_len, :]
        sin = self.cached_sin[:, :, :seq_len, :]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embeddings (RoPE)."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class TopKRouter(nn.Module):
    """ALF-LB Router: deterministic top-k with dual entropy regularization."""

    def __init__(
        self,
        dim: int,
        target_sparsity: float = 0.2,
        sparsity_coef: float = 0.1,
        entropy_coef: float = 0.01,
        local_window: int = 64,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, 1, bias=False)
        self.register_buffer("route_bias", torch.zeros(1))
        self.bias_update_rate = 0.001

        self.register_buffer("target_sparsity", torch.tensor(target_sparsity))
        self._sparsity_float = float(target_sparsity)  # avoids .item() GPU→CPU sync

        self.token_entropy_coef = entropy_coef
        self.batch_entropy_coef = entropy_coef * 5.0
        self.local_window = local_window

        nn.init.xavier_uniform_(self.gate.weight, gain=0.02)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        B, S, _ = x.shape

        logits = self.gate(x).squeeze(-1) + self.route_bias
        probs = torch.sigmoid(logits)

        # MXFP8 noise suppression
        probs = torch.where(probs < 0.05, torch.zeros_like(probs), probs)

        if self.training:
            # Dual entropy: minimize token-level, maximize batch-level
            token_entropy = -(
                probs * (probs + 1e-10).log()
                + (1.0 - probs) * (1.0 - probs + 1e-10).log()
            ).mean()
            batch_prob = probs.mean()
            batch_entropy = -(
                batch_prob * (batch_prob + 1e-10).log()
                + (1.0 - batch_prob) * (1.0 - batch_prob + 1e-10).log()
            )
            aux_loss = (self.token_entropy_coef * token_entropy) - (
                self.batch_entropy_coef * batch_entropy
            )
        else:
            aux_loss = probs.new_zeros(())

        k = max(1, int(S * self._sparsity_float))

        _, indices = torch.topk(logits, k, dim=1, sorted=False)

        indices, _ = torch.sort(indices, dim=-1)
        indices = indices.contiguous()

        K_kv = min(S, k + self.local_window)

        if self.training:
            mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
            mask.scatter_(1, indices, True)
            kv_logits = logits.clone()
            if self.local_window > 0:
                kv_logits[:, -self.local_window :] += 10000.0
        else:
            mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
            mask.scatter_(1, indices, True)
            kv_logits = logits
            if self.local_window > 0:
                kv_logits[:, -self.local_window :] += 10000.0

        _, kv_indices = torch.topk(kv_logits, K_kv, dim=1, sorted=False)
        kv_indices, _ = torch.sort(kv_indices, dim=-1)
        kv_indices = kv_indices.contiguous()

        kv_mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        kv_mask.scatter_(1, kv_indices, True)

        if self.training:
            # Out-of-graph bias update for load balance
            e_i = self.target_sparsity - probs.detach().mean()
            self.route_bias.add_(self.bias_update_rate * torch.sign(e_i))

        return RouterOutput(mask, indices, kv_mask, kv_indices, probs, aux_loss)


class LinearStateSpaceStream(nn.Module):
    """Reflexive stream: chunked causal linear attention + SwiGLU MLP.

    Receives pre-projected Q, K, V from HSPMNBlock (shared projections).
    Applies ELU+1 feature map and per-chunk GQA expansion.
    """

    _MAX_ATTN_BYTES: int = 128 * (1 << 20)  # 128 MiB budget for A=[B,H,C,C]
    _MAX_BATCH: int = 64

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        hidden = dim * mlp_ratio

        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

        # Chunk size (compile-time constant)
        _bytes_per_elem = 2
        _budget = self._MAX_BATCH * num_heads * _bytes_per_elem
        self._chunk_size: int = max(
            64, int(math.isqrt(max(1, self._MAX_ATTN_BYTES // max(1, _budget))))
        )
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(self._chunk_size, self._chunk_size)),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        for w in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(w.weight, gain=0.02)

    def forward(
        self, q_raw: torch.Tensor, k_raw: torch.Tensor, v_raw: torch.Tensor
    ) -> torch.Tensor:
        """q_raw [B,S,dim], k_raw [B,S,kv_dim], v_raw [B,S,kv_dim] -> [B,S,dim]."""
        B, S, _ = q_raw.shape

        # ELU+1 feature map for linear attention kernel
        q = (F.elu(q_raw) + 1.0).view(B, S, self.num_heads, self.head_dim)
        k = (F.elu(k_raw) + 1.0).view(B, S, self.num_kv_heads, self.head_dim)
        v = v_raw.view(B, S, self.num_kv_heads, self.head_dim)

        # Chunked causal linear attention
        C = self._chunk_size
        _S = int(S)  # concretise for loop bound; Dynamo adds shape guard
        attn_out = torch.empty(
            B, S, self.num_heads, self.head_dim, device=q_raw.device, dtype=q_raw.dtype
        )

        # Running state (fp32 for long-range stability)
        state = torch.zeros(
            B,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=q_raw.device,
            dtype=torch.float32,
        )
        z_state = torch.zeros(
            B, self.num_heads, self.head_dim, device=q_raw.device, dtype=torch.float32
        )

        for start in range(0, _S, C):
            end = min(start + C, _S)
            q_c = q[:, start:end]
            # Per-chunk GQA expansion (memory-efficient: one chunk at a time)
            k_c = k[:, start:end].repeat_interleave(self.kv_groups, dim=2)
            v_c = v[:, start:end].repeat_interleave(self.kv_groups, dim=2)
            C_eff = end - start

            A = torch.einsum("bthk, bshk -> bhts", q_c, k_c)
            A = A * self._causal_mask[:C_eff, :C_eff].to(dtype=A.dtype)
            num_intra = torch.einsum("bhts, bshv -> bthv", A, v_c)

            num_cross = torch.einsum("bthk, bhkv -> bthv", q_c.float(), state)
            num = num_intra.float() + num_cross

            den_intra = A.sum(dim=-1).permute(0, 2, 1)
            den_cross = torch.einsum("bthk, bhk -> bth", q_c.float(), z_state)
            den = den_intra.float() + den_cross

            attn_out[:, start:end] = (num / (den.unsqueeze(-1) + 1e-6)).to(q_raw.dtype)

            state = state + torch.einsum("bchk, bchv -> bhkv", k_c.float(), v_c.float())
            z_state = z_state + k_c.float().sum(dim=1)

        attn_out = attn_out.view(B, S, self.dim)

        gate_out = self.gate_proj(attn_out)
        gate_out = torch.nan_to_num(gate_out, nan=0.0).clamp(
            min=-1e4
        )  # F.silu NaN guard
        return self.down_proj(F.silu(gate_out) * self.up_proj(attn_out))


class HSPMNBlock(nn.Module):
    """Dual-stream block: reflexive (linear) + contextual (sparse attention)."""

    def __init__(self, config: HSPMNConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.router = TopKRouter(
            config.dim,
            config.sparsity_k,
            config.router_sparsity_coef,
            config.router_entropy_coef,
        )
        self.norm = nn.RMSNorm(config.dim)
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)
        self.reflexive = LinearStateSpaceStream(
            config.dim,
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            config.mlp_ratio,
        )

        self.num_sink_tokens = config.num_sink_tokens
        self.sink_tokens = nn.Parameter(
            torch.zeros(1, self.num_sink_tokens, config.dim)
        )

        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.dim)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=scale)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=scale)
        nn.init.normal_(self.sink_tokens, std=0.02)

    def _attention_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_indices: torch.Tensor,
        kv_indices_full: torch.Tensor,
        B: int,
        S: int,
        D: int,
        seq_len_offset: int,
    ) -> torch.Tensor:
        """SQSK attention via Triton kernel."""
        # q: [B, H, S, D] -> [B, S, H, D]
        # k, v: [B, H_kv, num_sinks + S_kv_full, D] -> [B, num_sinks + S_kv_full, H_kv, D]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Gather selected queries
        indices_expanded = (
            q_indices.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.num_heads, self.head_dim)
        )
        q_selected = torch.gather(q_t, 1, indices_expanded).contiguous()

        k_selected = k_t.contiguous()
        v_selected = v_t.contiguous()

        # Prepare causal positions
        real_q_indices = (q_indices + seq_len_offset).to(torch.int32)
        real_sink_pos = torch.zeros(
            B, self.num_sink_tokens, dtype=torch.int32, device=q.device
        )
        real_kv_pos = kv_indices_full.to(torch.int32)
        real_full_kv_indices = torch.cat([real_sink_pos, real_kv_pos], dim=1)

        # Run SQSK Attention
        attn_out = sparse_query_sparse_key_attention(
            q_selected, k_selected, v_selected, real_q_indices, real_full_kv_indices
        )

        # Scatter back
        out = torch.zeros_like(q_t)
        out.scatter_(1, indices_expanded, attn_out)

        return self.o_proj(out.view(B, S, D))

    def _attention_flex(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: torch.Tensor,
        kv_indices_full: torch.Tensor,
        B: int,
        S: int,
        D: int,
        S_kv_full: int,
        seq_len_offset: int,
    ) -> torch.Tensor:
        """FlexAttention path with causal + sparsity block mask."""

        def mask_mod(b, h, q_idx, kv_idx):
            # Causal logic shifted by Sink Tokens count and sequence offset
            real_q_pos = q_idx + seq_len_offset
            real_kv_pos = kv_indices_full[
                b, torch.clamp(kv_idx - self.num_sink_tokens, min=0)
            ]

            causal = real_q_pos >= real_kv_pos
            q_ok = q_mask[b, q_idx]

            return causal & q_ok

        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=1,
            Q_LEN=S,
            KV_LEN=S_kv_full + self.num_sink_tokens,
            device=q.device,
        )

        out = flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))

    def _attention(
        self,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        router_out: RouterOutput,
        past_key_values: tuple = None,
    ) -> Tuple[torch.Tensor, tuple]:
        B, S, _ = q_raw.shape
        D = self.dim

        q = q_raw.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_raw.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_raw.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        seq_len_offset = 0
        if past_key_values is not None:
            past_k, past_v, past_kv_indices = past_key_values
            seq_len_offset = past_k.shape[2]

        cos, sin = self.rope(q, S + seq_len_offset)
        cos_curr = cos[:, :, seq_len_offset : seq_len_offset + S, :]
        sin_curr = sin[:, :, seq_len_offset : seq_len_offset + S, :]

        q, k = apply_rotary_pos_emb(q, k, cos_curr, sin_curr)

        kv_idx_exp = (
            router_out.kv_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(-1, self.num_kv_heads, -1, self.head_dim)
        )
        k_sparse = torch.gather(k, 2, kv_idx_exp)
        v_sparse = torch.gather(v, 2, kv_idx_exp)

        if past_key_values is not None:
            past_k, past_v, past_kv_indices = past_key_values
            k_sparse_full = torch.cat([past_k, k_sparse], dim=2)
            v_sparse_full = torch.cat([past_v, v_sparse], dim=2)
            kv_indices_full = torch.cat(
                [past_kv_indices, router_out.kv_indices + seq_len_offset], dim=1
            )
        else:
            k_sparse_full = k_sparse
            v_sparse_full = v_sparse
            kv_indices_full = router_out.kv_indices

        new_past_key_values = (k_sparse_full, v_sparse_full, kv_indices_full)
        S_kv_full = k_sparse_full.shape[2]

        sink = self.sink_tokens.expand(B, -1, -1)
        k_sink = (
            self.k_proj(self.norm(sink))
            .view(B, self.num_sink_tokens, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_sink = (
            self.v_proj(self.norm(sink))
            .view(B, self.num_sink_tokens, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Sinks live at position 0; RoPE at position 0 is identity (cos=1, sin=0),
        # so we skip the no-op apply_rotary_pos_emb call here.
        k_full = torch.cat([k_sink, k_sparse_full], dim=2)
        v_full = torch.cat([v_sink, v_sparse_full], dim=2)

        # Triton SQSK kernel (inference only)
        if (
            HAS_TRITON_KERNELS
            and router_out.indices is not None
            and not self.training
            and q.is_cuda
        ):
            attn_out = self._attention_triton(
                q,
                k_full,
                v_full,
                router_out.indices,
                kv_indices_full,
                B,
                S,
                D,
                seq_len_offset,
            )
            return attn_out, new_past_key_values

        # Fallback to FlexAttention
        attn_out = self._attention_flex(
            q,
            k_full,
            v_full,
            router_out.mask,
            kv_indices_full,
            B,
            S,
            D,
            S_kv_full,
            seq_len_offset,
        )
        return attn_out, new_past_key_values

    def forward(
        self, x: torch.Tensor, past_key_values: tuple = None
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        router_out = self.router(x)
        x_norm = self.norm(x)

        # Shared QKV projection (used by both streams)
        q_raw = self.q_proj(x_norm)
        k_raw = self.k_proj(x_norm)
        v_raw = self.v_proj(x_norm)

        reflexive_out = self.reflexive(q_raw, k_raw, v_raw)

        attn_out, new_past_key_values = self._attention(
            q_raw, k_raw, v_raw, router_out, past_key_values
        )

        # Gate by continuous router probabilities for gradient flow
        attn_out = attn_out * router_out.probs.unsqueeze(-1)

        return x + reflexive_out + attn_out, router_out.aux_loss, new_past_key_values
