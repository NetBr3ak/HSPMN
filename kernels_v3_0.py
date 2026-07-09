"""
Proprietary / All Rights Reserved - Non-Commercial Use Only
Source-available for portfolio viewing only. Commercial use, unauthorized modification, reproduction, or distribution is strictly prohibited. All rights reserved.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["N_ACTIVE_Q", "N_ACTIVE_K"],
)
@triton.jit
def _sqsk_fwd_kernel(
    Q,
    K,
    V,
    Out,
    Q_indices,
    K_indices,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_iqb,
    stride_iqm,
    stride_ikb,
    stride_ikn,
    sm_scale,
    Z,
    H,
    H_KV,
    N_ACTIVE_Q,
    N_ACTIVE_K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PRE_LOAD_V: tl.constexpr = False,
):
    """
    Forward kernel for Sparse-Query Sparse-Key (SQSK) Attention.
    Uses TMA block pointers with order=(1,0) for burst coalescing.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // (H // H_KV)

    # Base pointers
    Q_ptr = Q + off_z * stride_qb + off_h * stride_qh
    K_ptr = K + off_z * stride_kb + off_h_kv * stride_kh
    V_ptr = V + off_z * stride_vb + off_h_kv * stride_vh
    Out_ptr = Out + off_z * stride_ob + off_h * stride_oh
    Q_Idx_ptr = Q_indices + off_z * stride_iqb
    K_Idx_ptr = K_indices + off_z * stride_ikb

    # TMA block pointers
    q_block_ptr = tl.make_block_ptr(
        base=Q_ptr,
        shape=(N_ACTIVE_Q, HEAD_DIM),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    out_block_ptr = tl.make_block_ptr(
        base=Out_ptr,
        shape=(N_ACTIVE_Q, HEAD_DIM),
        strides=(stride_om, stride_od),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q indices for this block
    q_real_pos = tl.load(
        Q_Idx_ptr + offs_m * stride_iqm, mask=offs_m < N_ACTIVE_Q, other=-1
    )
    max_q_pos = tl.max(q_real_pos, 0)

    # Load Q via TMA
    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # Accumulators
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Pipeline loop
    for start_n in range(0, N_ACTIVE_K, BLOCK_N):
        # Skip if all keys are beyond max_q_pos (causal)
        k_real_pos = tl.load(
            K_Idx_ptr + (start_n + offs_n) * stride_ikn,
            mask=(start_n + offs_n) < N_ACTIVE_K,
            other=float("inf"),
        )

        if tl.min(k_real_pos, 0) <= max_q_pos:
            k_block_ptr = tl.make_block_ptr(
                base=K_ptr,
                shape=(N_ACTIVE_K, HEAD_DIM),
                strides=(stride_kn, stride_kd),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, HEAD_DIM),
                order=(1, 0),
            )
            # FP8 TMA L2 bypass adaptation
            v_order: tl.constexpr = (
                (0, 1)
                if V.dtype.element_ty == getattr(tl, "float8e5", None)
                else (1, 0)
            )

            v_block_ptr = tl.make_block_ptr(
                base=V_ptr,
                shape=(N_ACTIVE_K, HEAD_DIM),
                strides=(stride_vn, stride_vd),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, HEAD_DIM),
                order=v_order,
            )

            # Load K, V via TMA
            k = tl.load(k_block_ptr, boundary_check=(0, 1))

            # QK^T
            qk = tl.dot(q, tl.trans(k))
            qk *= sm_scale

            # Causal mask
            mask = k_real_pos[None, :] <= q_real_pos[:, None]
            qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.max(qk, 1)
            m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            p = tl.exp(qk - m_ij_safe[:, None])
            l_ij = tl.sum(p, 1)

            m_i_new = tl.maximum(m_i, m_ij)
            m_i_new_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)

            alpha = tl.exp(m_i - m_i_new_safe)
            beta = tl.exp(m_ij - m_i_new_safe)

            l_i_new = alpha * l_i + beta * l_ij

            v = tl.load(v_block_ptr, boundary_check=(0, 1))

            # Use q.dtype to be agnostic to BF16/FP16
            p = p.to(q.dtype)
            v = v.to(q.dtype)
            acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p, v)

            l_i = l_i_new
            m_i = m_i_new

    # Finalize and store. Guard against empty rows (no key passed the causal
    # mask, e.g. tail block where offs_m >= N_ACTIVE_Q): l_i==0 would yield NaN.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / safe_l[:, None]
    tl.store(out_block_ptr, acc.to(q.dtype), boundary_check=(0, 1))


def sparse_query_sparse_key_attention(q, k, v, q_indices, k_indices):
    """SQSK attention wrapper. Indices MUST be sorted."""
    assert k.shape[1] == v.shape[1]
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    HEAD_DIM = q.shape[-1]
    sm_scale = 1.0 / (HEAD_DIM**0.5)

    # Output
    o = torch.empty_like(q)

    def grid(META):
        return (triton.cdiv(q.shape[1], META["BLOCK_M"]), q.shape[0] * q.shape[2])

    _sqsk_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        q_indices,
        k_indices,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q_indices.stride(0),
        q_indices.stride(1),
        k_indices.stride(0),
        k_indices.stride(1),
        sm_scale,
        q.shape[0],
        q.shape[2],
        k.shape[2],
        q.shape[1],
        k.shape[1],
        HEAD_DIM=HEAD_DIM,
    )
    return o
