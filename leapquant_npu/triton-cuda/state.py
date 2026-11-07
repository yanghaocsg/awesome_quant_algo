# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LeapQuant torch reference used as ground truth and fallback by triton-cuda.

Device-agnostic, stock PyTorch. Same logic as the standalone torch-cpu backend.
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
    "roundtrip",
    "state_mse",
]


@dataclass(frozen=True)
class LeapQuantSpec:
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
            raise ValueError(f"this path supports bits=8 only, got {self.bits}")
        if self.compensator_tokens < 0:
            raise ValueError(f"compensator_tokens must be >= 0, got {self.compensator_tokens}")
        if self.power_iters <= 0:
            raise ValueError(f"power_iters must be positive, got {self.power_iters}")


@dataclass
class EncodedState:
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
    work = state.to(torch.float32)
    comp_k, comp_u = fit_compensator_tokens(work, spec.compensator_tokens, spec.power_iters, spec.eps)

    if spec.compensator_tokens > 0:
        residual = work - comp_k @ comp_u.transpose(-1, -2)
    else:
        residual = work

    if spec.smooth:
        scale_c = residual.abs().mean(dim=-2, keepdim=True).sqrt().clamp_min(spec.eps)
    else:
        scale_c = torch.ones_like(residual[..., :1, :])

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
    residual = (encoded.residual.to(torch.float32) * encoded.scale_b) * encoded.scale_c
    if encoded.comp_tokens > 0:
        comp = encoded.comp_k.to(torch.float32) @ encoded.comp_u.to(torch.float32).transpose(-1, -2)
        residual = residual + comp
    return residual if dtype is None else residual.to(dtype)


def roundtrip(state: torch.Tensor, spec: LeapQuantSpec) -> torch.Tensor:
    return decode(encode(state, spec), dtype=state.dtype)


def state_mse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    diff = reference.to(torch.float32) - actual.to(torch.float32)
    return diff.pow(2).mean().item()


def is_window_start(num_computed_tokens: torch.Tensor, window: int) -> torch.Tensor:
    return (num_computed_tokens % window == 0) & (num_computed_tokens > 0)


def closes_window(num_computed_tokens: torch.Tensor, num_new_tokens: torch.Tensor, window: int) -> torch.Tensor:
    return (num_computed_tokens + num_new_tokens) % window == 0
