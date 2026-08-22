"""Triton FlashAttention-2 forward kernel (flash_forward b/c)."""
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # fp32 on-chip accumulators
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q = tl.load(Q_block_ptr)  # (Q_TILE, D)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr)                       # (K_TILE, D)
        v = tl.load(V_block_ptr)
        s = tl.dot(q, tl.trans(k)) * scale             # (Q_TILE, K_TILE)

        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s = tl.where(q_idx[:, None] >= k_idx[None, :], s, -1e6)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])                 # unnormalized P tile
        alpha = tl.exp(m_i - m_new)                    # rescale for old sums
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    tl.store(L_block_ptr, lse)


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 forward via the fused Triton kernel above."""

    Q_TILE_SIZE = 32
    K_TILE_SIZE = 32

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        *lead, Nq, D = Q.shape
        Nk = K.shape[-2]
        B = 1
        for s in lead:
            B *= s

        Qf = Q.reshape(B, Nq, D).contiguous()
        Kf = K.reshape(B, Nk, D).contiguous()
        Vf = V.reshape(B, Nk, D).contiguous()
        O = torch.empty_like(Qf)
        L = torch.empty(B, Nq, dtype=torch.float32, device=Q.device)

        grid = (triton.cdiv(Nq, FlashAttentionTriton.Q_TILE_SIZE), B)
        flash_fwd_kernel[grid](
            Qf, Kf, Vf, O, L,
            Qf.stride(0), Qf.stride(1), Qf.stride(2),
            Kf.stride(0), Kf.stride(1), Kf.stride(2),
            Vf.stride(0), Vf.stride(1), Vf.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            1.0 / (D ** 0.5),
            D=D,
            Q_TILE_SIZE=FlashAttentionTriton.Q_TILE_SIZE,
            K_TILE_SIZE=FlashAttentionTriton.K_TILE_SIZE,
            is_causal=is_causal,
        )

        O = O.reshape(*lead, Nq, D)
        L = L.reshape(*lead, Nq)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        from cs336_systems.flash_attention import _flash_backward
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = _flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
