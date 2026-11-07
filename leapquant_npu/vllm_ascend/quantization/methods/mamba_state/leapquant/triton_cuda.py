# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend (2): LeapQuant kernels in standard Triton (CUDA backend).

Stage A covers the regular reduction / elementwise path: residual smoothing,
per-value symmetric INT8 quantization, and dequantization. It imports stock
``triton`` (not ``vllm.triton_utils``) so it runs on a normal CUDA environment
such as the AutoDL RTX 6000D. Off CUDA, each wrapper falls back to the torch
reference in ``state.py``; the kernel path is validated on the GPU.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .state import INT8_MAX

BLOCK_DV = 64
BLOCK_DK = 64


@triton.jit
def _smoothing_scale_kernel(
    r_ptr,
    c_ptr,
    stride_b,
    stride_dv,
    stride_dk,
    DV: tl.constexpr,
    DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    batch = tl.program_id(0)
    dk_off = tl.program_id(1) * BLOCK_DK + tl.arange(0, BLOCK_DK)
    dk_mask = dk_off < DK
    acc = tl.zeros((BLOCK_DK,), dtype=tl.float32)
    for dv_start in range(0, DV, BLOCK_DV):
        dv_off = dv_start + tl.arange(0, BLOCK_DV)
        dv_mask = dv_off < DV
        ptrs = r_ptr + batch * stride_b + dv_off[:, None] * stride_dv + dk_off[None, :] * stride_dk
        vals = tl.load(ptrs, mask=dv_mask[:, None] & dk_mask[None, :], other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=0)
    tl.store(c_ptr + batch * DK + dk_off, tl.sqrt(acc / DV), mask=dk_mask)


@triton.jit
def _int8_quant_kernel(
    r_ptr,
    c_ptr,
    q_ptr,
    sb_ptr,
    stride_b,
    stride_dv,
    stride_dk,
    DV: tl.constexpr,
    DK: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    SMOOTH: tl.constexpr,
):
    batch = tl.program_id(0)
    dv_pid = tl.program_id(1)
    dk_off = tl.arange(0, BLOCK_DK)
    dk_mask = dk_off < DK
    row = tl.load(
        r_ptr + batch * stride_b + dv_pid * stride_dv + dk_off * stride_dk,
        mask=dk_mask,
        other=0.0,
    ).to(tl.float32)
    c = tl.load(c_ptr + batch * DK + dk_off, mask=dk_mask, other=1.0)
    scaled = tl.where(SMOOTH, row / c, row)
    sb = tl.maximum(tl.max(tl.abs(scaled), axis=0) / INT8_MAX, 1e-12)
    x = scaled / sb
    # Round half away from zero without relying on a libdevice rint intrinsic.
    qf = tl.where(
        x >= 0,
        (x + 0.5).to(tl.int32).to(tl.float32),
        (x - 0.5).to(tl.int32).to(tl.float32),
    )
    q = tl.clamp(qf, -INT8_MAX, INT8_MAX)
    tl.store(q_ptr + batch * DV * DK + dv_pid * DK + dk_off, q.to(tl.int8), mask=dk_mask)
    tl.store(sb_ptr + batch * DV + dv_pid, sb)


@triton.jit
def _int8_dequant_kernel(
    q_ptr,
    sb_ptr,
    c_ptr,
    out_ptr,
    DV: tl.constexpr,
    DK: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    batch = tl.program_id(0)
    dv_pid = tl.program_id(1)
    dk_off = tl.arange(0, BLOCK_DK)
    dk_mask = dk_off < DK
    q = tl.load(q_ptr + batch * DV * DK + dv_pid * DK + dk_off, mask=dk_mask, other=0).to(tl.float32)
    sb = tl.load(sb_ptr + batch * DV + dv_pid)
    c = tl.load(c_ptr + batch * DK + dk_off, mask=dk_mask, other=1.0)
    tl.store(out_ptr + batch * DV * DK + dv_pid * DK + dk_off, q * sb * c, mask=dk_mask)


def _flatten(state: torch.Tensor) -> tuple[torch.Tensor, tuple, int, int, int]:
    lead = state.shape[:-2]
    dv, dk = state.shape[-2], state.shape[-1]
    batch = math.prod(lead) if lead else 1
    return state.reshape(batch, dv, dk).contiguous(), lead, batch, dv, dk


def cuda_smoothing_scale(residual: torch.Tensor) -> torch.Tensor:
    """Per-key smoothing factor ``c`` with shape ``[..., 1, dk]``."""
    flat, lead, batch, dv, dk = _flatten(residual)
    if flat.device.type != "cuda":
        return flat.abs().mean(dim=-2).sqrt().reshape(*lead, 1, dk)
    c = torch.empty((batch, dk), dtype=torch.float32, device=flat.device)
    _smoothing_scale_kernel[(batch, triton.cdiv(dk, BLOCK_DK))](
        flat, c, flat.stride(0), flat.stride(1), flat.stride(2), dv, dk, BLOCK_DV, BLOCK_DK
    )
    return c.reshape(*lead, 1, dk)


def cuda_int8_quant(
    state: torch.Tensor,
    smooth: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``state`` -> ``(int8 residual, scale_b [...,dv,1], scale_c [...,1,dk])``."""
    flat, lead, batch, dv, dk = _flatten(state)
    if flat.device.type != "cuda":
        if smooth:
            c = flat.abs().mean(dim=-2, keepdim=True).sqrt().clamp_min(1e-8)
            scaled = flat / c
        else:
            c = torch.ones((batch, 1, dk))
            scaled = flat
        sb = (scaled.abs().amax(dim=-1, keepdim=True) / INT8_MAX).clamp_min(1e-12)
        q = (scaled / sb).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        return (
            q.reshape(*lead, dv, dk),
            sb.reshape(*lead, dv, 1),
            c.reshape(*lead, 1, dk),
        )

    c = torch.empty((batch, dk), dtype=torch.float32, device=flat.device)
    _smoothing_scale_kernel[(batch, triton.cdiv(dk, BLOCK_DK))](
        flat, c, flat.stride(0), flat.stride(1), flat.stride(2), dv, dk, BLOCK_DV, BLOCK_DK
    )
    q = torch.empty((batch, dv, dk), dtype=torch.int8, device=flat.device)
    sb = torch.empty((batch, dv), dtype=torch.float32, device=flat.device)
    _int8_quant_kernel[(batch, dv)](
        flat,
        c,
        q,
        sb,
        flat.stride(0),
        flat.stride(1),
        flat.stride(2),
        dv,
        dk,
        triton.next_power_of_2(dk),
        smooth,
    )
    return q.reshape(*lead, dv, dk), sb.reshape(*lead, dv, 1), c.reshape(*lead, 1, dk)


def cuda_int8_dequant(
    q: torch.Tensor,
    scale_b: torch.Tensor,
    scale_c: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct the fp32 state from the Stage A payload."""
    lead = q.shape[:-2]
    dv, dk = q.shape[-2], q.shape[-1]
    batch = math.prod(lead) if lead else 1
    qf, sbf, cf = q.reshape(batch, dv, dk), scale_b.reshape(batch, dv), scale_c.reshape(batch, dk)
    if qf.device.type != "cuda":
        out = qf.float() * sbf.unsqueeze(-1) * cf.unsqueeze(1)
        return out.reshape(*lead, dv, dk).to(dtype)

    out = torch.empty((batch, dv, dk), dtype=torch.float32, device=qf.device)
    _int8_dequant_kernel[(batch, dv)](qf, sbf, cf, out, dv, dk, triton.next_power_of_2(dk))
    return out.reshape(*lead, dv, dk).to(dtype)
