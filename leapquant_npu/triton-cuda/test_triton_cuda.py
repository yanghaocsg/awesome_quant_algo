# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA tests for the self-contained standard-Triton Stage A kernels.

Skips at module level when Triton or a CUDA device is missing, so this file is
inert on the CPU machine and runs on the AutoDL RTX 6000D.
"""

import pytest

pytest.importorskip("triton")
import torch  # noqa: E402

import triton_cuda  # noqa: E402
from state import INT8_MAX, state_mse  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for triton-cuda kernels", allow_module_level=True)


def _state(seed=0, dv=32, dk=32, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(dv, dk, generator=g)
    s[0, :] *= 30.0
    s[:, 0] *= 12.0
    return s.to(device)


def test_smoothing_scale_matches_reference():
    state = _state(seed=1)
    got = triton_cuda.cuda_smoothing_scale(state)
    ref = state.abs().mean(dim=-2).sqrt().reshape(1, -1)
    torch.testing.assert_close(got, ref)


def test_quant_payload_matches_reference_within_tie():
    state = _state(seed=2)
    q, sb, sc = triton_cuda.cuda_int8_quant(state, smooth=True)

    ref_c = state.abs().mean(dim=-2).sqrt().clamp_min(1e-8)
    ref_scaled = state / ref_c
    ref_sb = (ref_scaled.abs().amax(-1, keepdim=True) / INT8_MAX).clamp_min(1e-12)
    ref_q = (ref_scaled / ref_sb).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)

    assert (q.to(torch.int16) - ref_q.to(torch.int16)).abs().max().item() <= 1
    torch.testing.assert_close(sb.squeeze(-1), ref_sb.squeeze(-1), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(sc.squeeze(-2), ref_c, rtol=1e-4, atol=1e-6)


def test_roundtrip_through_kernels_is_close():
    state = _state(seed=3)
    q, sb, sc = triton_cuda.cuda_int8_quant(state, smooth=True)
    out = triton_cuda.cuda_int8_dequant(q, sb, sc)
    assert out.shape == state.shape
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, state, rtol=0.05, atol=0.05)


def test_quant_without_smoothing():
    state = _state(seed=4)
    q, sb, sc = triton_cuda.cuda_int8_quant(state, smooth=False)
    assert torch.all(sc == 1.0)
    out = triton_cuda.cuda_int8_dequant(q, sb, sc)
    assert state_mse(state, out) < 1.0


def test_leading_dims_preserved():
    state = torch.randn(2, 4, 32, 32, device="cuda")
    q, sb, sc = triton_cuda.cuda_int8_quant(state, smooth=True)
    assert q.shape == (2, 4, 32, 32)
    assert sb.shape == (2, 4, 32, 1)
    assert sc.shape == (2, 4, 1, 32)
    assert triton_cuda.cuda_int8_dequant(q, sb, sc).shape == state.shape
