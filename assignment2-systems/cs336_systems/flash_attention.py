"""FlashAttention-2 implementations (flash_forward / flash_backward problems)."""
import torch


def _flash_backward_impl(Q, K, V, O, dO, L, is_causal: bool):
    """Recomputation backward for FlashAttention-2 (handout eqs. 13-19).
    P is recomputed from Q, K and the logsumexp L instead of being stored."""
    in_dtype = Q.dtype
    Q, K, V, O, dO = (t.float() for t in (Q, K, V, O, dO))
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    S = (Q @ K.transpose(-1, -2)) * scale                       # (B, Nq, Nk)
    if is_causal:
        q_idx = torch.arange(S.shape[-2], device=S.device)[:, None]
        k_idx = torch.arange(S.shape[-1], device=S.device)[None, :]
        S = torch.where(q_idx >= k_idx, S, torch.tensor(-1e6, dtype=S.dtype, device=S.device))
    P = torch.exp(S - L[..., None])                             # (B, Nq, Nk)
    dV = P.transpose(-1, -2) @ dO                               # (B, Nk, d)
    dP = dO @ V.transpose(-1, -2)                               # (B, Nq, Nk)
    D = (O * dO).sum(dim=-1, keepdim=True)                      # (B, Nq, 1)
    dS = P * (dP - D)
    dQ = (dS @ K) * scale                                       # (B, Nq, d)
    dK = (dS.transpose(-1, -2) @ Q) * scale                     # (B, Nk, d)
    return dQ.to(in_dtype), dK.to(in_dtype), dV.to(in_dtype)


_flash_backward_compiled = torch.compile(_flash_backward_impl)


def _flash_backward(Q, K, V, O, dO, L, is_causal):
    *lead, Nq, d = Q.shape
    Nk = K.shape[-2]
    B = 1
    for s in lead:
        B *= s
    dQ, dK, dV = _flash_backward_compiled(
        Q.reshape(B, Nq, d).contiguous(), K.reshape(B, Nk, d).contiguous(),
        V.reshape(B, Nk, d).contiguous(), O.reshape(B, Nq, d).contiguous(),
        dO.reshape(B, Nq, d).contiguous(), L.reshape(B, Nq).contiguous(),
        is_causal,
    )
    return dQ.reshape(Q.shape), dK.reshape(K.shape), dV.reshape(V.shape)



def _flash_forward_tiled(Q, K, V, is_causal, q_tile=16, k_tile=16):
    """Pure-PyTorch FlashAttention-2 forward (Algorithm 1), vectorized over batch
    and query-tile rows. Returns (O, L). Shapes: Q (B, Nq, d), K/V (B, Nk, d)."""
    B, Nq, d = Q.shape
    Nk = K.shape[1]
    scale = 1.0 / (d ** 0.5)
    dtype = Q.dtype
    dev = Q.device

    O = torch.empty(B, Nq, d, dtype=dtype, device=dev)
    L = torch.empty(B, Nq, dtype=torch.float32, device=dev)

    for q0 in range(0, Nq, q_tile):
        q1 = min(q0 + q_tile, Nq)
        Q_i = Q[:, q0:q1, :]                                   # (B, bq, d)
        bq = Q_i.shape[1]
        m = torch.full((B, bq), float("-inf"), dtype=torch.float32, device=dev)
        l = torch.zeros(B, bq, dtype=torch.float32, device=dev)
        O_i = torch.zeros(B, bq, d, dtype=torch.float32, device=dev)

        for k0 in range(0, Nk, k_tile):
            k1 = min(k0 + k_tile, Nk)
            K_j = K[:, k0:k1, :]
            V_j = V[:, k0:k1, :]
            S_ij = (Q_i @ K_j.transpose(-1, -2)) * scale       # (B, bq, bk)
            if is_causal:
                q_idx = torch.arange(q0, q1, device=dev)[:, None]
                k_idx = torch.arange(k0, k1, device=dev)[None, :]
                S_ij = torch.where(q_idx >= k_idx, S_ij, torch.tensor(-1e6, dtype=S_ij.dtype, device=dev))

            S_ij = S_ij.float()
            m_new = torch.maximum(m, S_ij.amax(dim=-1))        # (B, bq)
            P_tilde = torch.exp(S_ij - m_new[..., None])       # (B, bq, bk)
            alpha = torch.exp(m - m_new)                       # rescale factor (B, bq)
            l = l * alpha + P_tilde.sum(dim=-1)
            O_i = O_i * alpha[..., None] + P_tilde @ V_j.float()
            m = m_new

        O_i = O_i / l[..., None]
        L_i = m + torch.log(l)
        O[:, q0:q1, :] = O_i.to(dtype)
        L[:, q0:q1] = L_i

    return O, L


class FlashAttentionPyTorch(torch.autograd.Function):
    """FlashAttention-2 forward in pure PyTorch (debug reference for the Triton kernel)."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
        # flatten any leading batch dims
        *lead, Nq, d = Q.shape
        *_, Nk, _ = K.shape
        Qf = Q.reshape(-1, Nq, d)
        Kf = K.reshape(-1, Nk, d)
        Vf = V.reshape(-1, Nk, d)

        O, L = _flash_forward_tiled(Qf, Kf, Vf, is_causal)

        O = O.reshape(*lead, Nq, d)
        L = L.reshape(*lead, Nq)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = _flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
