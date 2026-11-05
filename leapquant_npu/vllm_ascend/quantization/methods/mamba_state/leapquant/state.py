# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LeapQuant recurrent-state quantization (M1 reference implementation).

LeapQuant (arXiv:2609.38166) quantizes the *recurrent state* of linear-attention
models (GDN / KDA) once per window instead of once per decode step, and keeps the
state's dominant low-rank structure in a few high-precision Compensator Tokens.

Layout note
-----------
vllm-ascend keeps the GDN recurrent state as ``ssm_state`` shaped
``[num_slots, num_v_heads, head_v_dim, head_k_dim]``, i.e. the transpose of the
paper's ``S in R^{d_k x d_v}``. The paper's components map onto that layout as:

* Compensator Tokens ``R ~ K U^T``, ``K in R^{d_v x r}``, ``U in R^{d_k x r}``
  (paper: ``tilde u in R^{d_v}``, ``tilde k in R^{d_k}``).
* Residual smoothing ``C`` acts on the **key** axis, which is the last axis here.
* The symmetric INT8 scale ``B`` is per-value-channel, i.e. per row here.

This module is torch-only on purpose: it runs on CPU in unit tests and holds no
NPU / vLLM dependency. The decode kernel integration lives in ``gdn.py`` and is
described in ``patches/gdn_integration.md``.
"""

from dataclasses import dataclass

import torch

INT8_MAX = 127

__all__ = [
    "EncodedState",
    "LeapQuantSpec",
    "closes_window",
    "decode",
    "encode",
    "fit_compensator_tokens",
    "is_window_start",
    "roundtrip",
    "state_mse",
]


@dataclass(frozen=True)
class LeapQuantSpec:
    """Static description of one LeapQuant encoding.

    ``window`` is the number of decode steps between two state quantizations.
    ``compensator_tokens`` (``r``) is the number of rank-one high-precision
    terms kept out of the quantized tensor; ``0`` disables the component and
    is useful for ablations.
    """

    window: int = 16
    bits: int = 8
    compensator_tokens: int = 4
    smooth: bool = True
    power_iters: int = 4
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")
        if self.bits != 8:
            # M1 only ships the 8-bit path; 4/6-bit need a different payload
            # format (see the M2/M3 plan) and must not silently alias to INT8.
            raise ValueError(f"M1 supports bits=8 only, got {self.bits}")
        if self.compensator_tokens < 0:
            raise ValueError(f"compensator_tokens must be >= 0, got {self.compensator_tokens}")
        if self.power_iters <= 0:
            raise ValueError(f"power_iters must be positive, got {self.power_iters}")


@dataclass
class EncodedState:
    """Low-bit representation of one window-boundary state.

    All tensors keep the leading dimensions of the input state, so a batch of
    slots / heads is encoded in one call.

    ``residual`` is int8, ``scale_b`` / ``scale_c`` are fp32, and the
    Compensator Tokens are fp16 because that is what the paper keeps in
    high precision; the fp16 round-trip is part of the error we measure.
    """

    residual: torch.Tensor
    scale_b: torch.Tensor
    scale_c: torch.Tensor
    comp_k: torch.Tensor
    comp_u: torch.Tensor

    @property
    def comp_tokens(self) -> int:
        return self.comp_k.shape[-1]


def _unit(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.norm().clamp_min(eps)


def fit_compensator_tokens(
    residual: torch.Tensor,
    num_tokens: int,
    iters: int = 4,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``num_tokens`` rank-one terms of ``residual`` via power iteration.

    Implements equation (6) of the paper with a deterministic power-iteration
    warm start (the residual's own energy direction) instead of a random one,
    so repeated runs and unit tests are bit-reproducible.

    Args:
        residual: ``[..., d_v, d_k]`` float32.
        num_tokens: rank ``r``. ``0`` returns empty factors.

    Returns:
        ``(K, U)`` with shapes ``[..., d_v, r]`` and ``[..., d_k, r]``.
    """
    if num_tokens == 0:
        shape_k = (*residual.shape[:-2], residual.shape[-2], 0)
        shape_u = (*residual.shape[:-2], residual.shape[-1], 0)
        return residual.new_zeros(shape_k), residual.new_zeros(shape_u)

    lead = residual.shape[:-2]
    d_v, d_k = residual.shape[-2], residual.shape[-1]
    flat = residual.reshape(-1, d_v, d_k)

    k_out = residual.new_zeros((flat.shape[0], d_v, num_tokens))
    u_out = residual.new_zeros((flat.shape[0], d_k, num_tokens))

    for bi in range(flat.shape[0]):
        work = flat[bi].clone()
        for h in range(num_tokens):
            # Deterministic start: the direction carrying most absolute mass.
            v = _unit(work.abs().sum(dim=0), eps)
            u = torch.zeros(d_v, dtype=work.dtype, device=work.device)
            for _ in range(iters):
                u = _unit(work @ v, eps)
                v = _unit(work.transpose(-1, -2) @ u, eps)
            sigma = torch.dot(u, work @ v)
            k_out[bi, :, h] = u
            u_out[bi, :, h] = v
            work = work - sigma * torch.outer(u, v)

    return k_out.reshape(*lead, d_v, num_tokens), u_out.reshape(*lead, d_k, num_tokens)


def encode(state: torch.Tensor, spec: LeapQuantSpec) -> EncodedState:
    """Quantize one window-boundary state (paper equations (5) and (7)).

    Args:
        state: ``[..., d_v, d_k]``, i.e. the vllm-ascend ``ssm_state`` layout.
    """
    work = state.to(torch.float32)
    comp_k, comp_u = fit_compensator_tokens(work, spec.compensator_tokens, spec.power_iters, spec.eps)

    if spec.compensator_tokens > 0:
        residual = work - comp_k @ comp_u.transpose(-1, -2)
    else:
        residual = work

    if spec.smooth:
        # c_i = sqrt(mean_j |R[i, j]|) over the value axis, floored to keep the
        # rescale invertible for all-zero rows.
        scale_c = residual.abs().mean(dim=-2, keepdim=True).sqrt().clamp_min(spec.eps)
    else:
        scale_c = torch.ones_like(residual[..., :1, :])

    # Per-value-channel symmetric INT8: one scale per row, over the key axis.
    scaled = residual / scale_c
    scale_b = (scaled.abs().amax(dim=-1, keepdim=True) / INT8_MAX).clamp_min(1e-12)
    quantized = (scaled / scale_b).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)

    return EncodedState(
        residual=quantized,
        scale_b=scale_b,
        scale_c=scale_c,
        comp_k=comp_k.to(torch.float16),
        comp_u=comp_u.to(torch.float16),
    )


def decode(encoded: EncodedState, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Reconstruct the fp32 state (paper equation (7), plus eq. (10) readout form)."""
    residual = (encoded.residual.to(torch.float32) * encoded.scale_b) * encoded.scale_c
    if encoded.comp_tokens > 0:
        comp = encoded.comp_k.to(torch.float32) @ encoded.comp_u.to(torch.float32).transpose(-1, -2)
        residual = residual + comp
    return residual if dtype is None else residual.to(dtype)


def roundtrip(state: torch.Tensor, spec: LeapQuantSpec) -> torch.Tensor:
    """Quantize then dequantize, preserving the input dtype."""
    return decode(encode(state, spec), dtype=state.dtype)


def state_mse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """Mean squared error used as the accuracy yardstick for M1."""
    diff = reference.to(torch.float32) - actual.to(torch.float32)
    return diff.pow(2).mean().item()


def is_window_start(num_computed_tokens: torch.Tensor, window: int) -> torch.Tensor:
    """Rows whose next decode step opens a new Window.

    ``num_computed_tokens`` is the per-request count *before* this step, so a
    row with ``n % window == 0`` is about to process its first token of a new
    window and must start from the quantized boundary state. Fresh requests
    (``n == 0``) are excluded: their state is empty and quantization is a no-op.
    """
    return (num_computed_tokens % window == 0) & (num_computed_tokens > 0)


def closes_window(num_computed_tokens: torch.Tensor, num_new_tokens: torch.Tensor, window: int) -> torch.Tensor:
    """Rows whose last processed token lands exactly on a window boundary.

    Used after a prefill chunk or a decode step to decide whether the freshly
    written state has to be quantized before the next step reads it.
    """
    return (num_computed_tokens + num_new_tokens) % window == 0
