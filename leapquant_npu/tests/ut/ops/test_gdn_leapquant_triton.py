# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Stage A Triton wrappers.

vLLM is not installed on this machine, so ``vllm.triton_utils`` is stubbed
before import. The wrappers take the torch fallback for a non-NPU tensor, which
is the only path exercised here; the real kernel launch is validated on NPU.
"""

import sys
import types

import torch

# --- stub vllm.triton_utils so kernels.py imports without vLLM ---
if "vllm" not in sys.modules:
    vllm_mod = types.ModuleType("vllm")
    tu_mod = types.ModuleType("vllm.triton_utils")

    def _next_power_of_2(n):
        return 1 << (n - 1).bit_length()

    def _cdiv(a, b):
        return (a + b - 1) // b

    tu_mod.triton = types.SimpleNamespace(
        jit=lambda f: f, cdiv=_cdiv, next_power_of_2=_next_power_of_2
    )
    tu_mod.tl = types.SimpleNamespace()
    vllm_mod.triton_utils = tu_mod
    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.triton_utils"] = tu_mod

from vllm_ascend.quantization.methods.mamba_state.leapquant import kernels  # noqa: E402
from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (  # noqa: E402
    INT8_MAX,
    state_mse,
)


def _state(seed=0, dv=32, dk=32):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(dv, dk, generator=g)
    s[0, :] *= 30.0
    s[:, 0] *= 12.0
    return s


def test_smoothing_scale_matches_reference():
    state = _state(seed=1)
    got = kernels.triton_smoothing_scale(state)
    ref = state.abs().mean(dim=-2).sqrt().reshape(1, -1)
    torch.testing.assert_close(got, ref)
    assert got.shape == (1, 32)


def test_quant_payload_matches_reference_within_tie():
    state = _state(seed=2)
    q, sb, sc = kernels.triton_int8_quant(state, smooth=True)

    ref_c = state.abs().mean(dim=-2).sqrt().clamp_min(1e-8)
    ref_scaled = state / ref_c
    ref_sb = (ref_scaled.abs().amax(-1, keepdim=True) / INT8_MAX).clamp_min(1e-12)
    ref_q = (ref_scaled / ref_sb).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)

    # tie rule may differ by one quantization level
    assert (q.to(torch.int16) - ref_q.to(torch.int16)).abs().max().item() <= 1
    torch.testing.assert_close(sb.squeeze(-1), ref_sb.squeeze(-1), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(sc.squeeze(-2), ref_c, rtol=1e-4, atol=1e-6)


def test_roundtrip_through_wrappers_is_close_to_torch():
    state = _state(seed=3)
    q, sb, sc = kernels.triton_int8_quant(state, smooth=True)
    out = kernels.triton_int8_dequant(q, sb, sc)
    assert out.shape == state.shape
    assert torch.isfinite(out).all()
    # reconstructed fp32 must track the torch round-trip closely
    torch.testing.assert_close(out, state, rtol=0.05, atol=0.05)


def test_quant_without_smoothing():
    state = _state(seed=4)
    q, sb, sc = kernels.triton_int8_quant(state, smooth=False)
    assert torch.all(sc == 1.0)
    out = kernels.triton_int8_dequant(q, sb, sc)
    assert state_mse(state, out) < 1.0


def test_leading_dims_preserved():
    state = torch.randn(2, 4, 32, 32)
    q, sb, sc = kernels.triton_int8_quant(state, smooth=True)
    assert q.shape == (2, 4, 32, 32)
    assert sb.shape == (2, 4, 32, 1)
    assert sc.shape == (2, 4, 1, 32)
    out = kernels.triton_int8_dequant(q, sb, sc)
    assert out.shape == state.shape
