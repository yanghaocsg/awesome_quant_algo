# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch port of the LeapQuant KDA window flush (arXiv:2609.38166, per-key-channel gate).

Faithful literal translation of the mathematical reference in kda/definition.json.
No cutlass / tilelang / B300 dependency; runs on any torch device (CUDA or CPU).
Interface mirrors the definition contract:
    run(codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, q0) -> (codes_new, s_k_new, s_v_new, u_new, q_new)
Differs from the GDN port only in the state rebuild: the checkpoint is decayed per key channel
by exp(pcum) and each buffered rank-1 update by the product of decay factors seen since append.
"""
import torch

QMAX = 127.0
EPS = 1e-12


def _mgs(m):
    """Modified Gram-Schmidt on the columns of m, projections applied twice (rank-deficient heads stay orthogonal)."""
    cols = []
    for j in range(m.shape[-1]):
        v = m[..., j]
        for _ in range(2):
            for c in cols:
                v = v - c * (c * v).sum(-1, keepdim=True)
        cols.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30))
    return torch.stack(cols, -1)


def decay_matrix(gbuf):
    """D_j = prod_{i > j} gbuf_i  (decay entry j has seen since it was appended)."""
    after = torch.flip(torch.cumprod(torch.flip(gbuf.float(), [-2]), -2), [-2])   # prod_{i >= j}
    return torch.cat([after[..., 1:, :], torch.ones_like(after[..., :1, :])], -2)  # prod_{i > j}


def rebuild_state(codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum):
    """1. Rebuild fp32 state S = (q^T u + dequant(codes)) diag(exp(pcum)) + sum_j w_j ubuf_j (kbuf_j * D_j)^T."""
    s_old = torch.einsum("bhrv,bhrk->bhvk", q.float(), u.float())
    s_old = s_old + codes.float() * (s_k[..., None, :] / QMAX) * s_v[..., :, None]
    D = decay_matrix(gbuf)
    s = s_old * torch.exp(pcum)[..., None, :] + torch.einsum("bhj,bhjv,bhjk->bhvk", w, ubuf.float(), kbuf.float() * D)
    return s


def subspace_iteration(s, q0):
    """2. ONE round of subspace iteration from q0 -> new R Compensator Tokens, stored as fp16."""
    x = s.transpose(-1, -2)                       # [K, V]
    qk = _mgs(x @ q0)                             # [K, R]
    qv = _mgs(x.transpose(-1, -2) @ qk)           # [V, R], orthonormal columns
    uk = x @ qv                                   # [K, R]
    u_new = uk.transpose(-1, -2).to(torch.float16)
    q_new = qv.transpose(-1, -2).to(torch.float16)
    return u_new, q_new


def quantize(s, u_new, q_new):
    """3. Residual E = S - q_new^T u_new, smoothing scales, int8 codes in [-127, 127]."""
    e = s - torch.einsum("bhrv,bhrk->bhvk", q_new.float(), u_new.float())
    sk = e.abs().mean(-2).sqrt().clamp_min(EPS)
    sv = (e.abs() / sk[..., None, :]).amax(-1).clamp_min(EPS)
    c = torch.round(e / (sk[..., None, :] * sv[..., :, None]) * QMAX).clamp(-QMAX, QMAX).to(torch.int8)
    return c, sk, sv


def run(codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, q0):
    """LeapQuant window flush for Kimi Delta Attention (per-key-channel gate).

    codes int8 [B, H, V, K]; s_k/s_v f32 [B, H, K]/[B, H, V]
    u/q f16 [B, H, R, K]/[B, H, R, V] (old Compensator Tokens)
    kbuf/ubuf bf16 [B, H, L, K]/[B, H, L, V]; gbuf f16 [B, H, L, K] (per-channel decay factor)
    w f32 [B, H, L]; pcum f32 [B, H, K] (log-gate summed over the window; checkpoint decay = exp(pcum))
    q0 f32 [V, R] (fixed start of the subspace iteration)
    """
    s = rebuild_state(codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum)
    u_new, q_new = subspace_iteration(s, q0)
    c, sk, sv = quantize(s, u_new, q_new)
    return c, sk, sv, u_new, q_new
