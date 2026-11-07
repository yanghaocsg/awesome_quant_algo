# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the self-contained torch-cpu LeapQuant implementation.

Torch-only and run on CPU: they validate the encoding round-trip, the
window-boundary logic, and the per-window vs per-step error on a simulated GDN
recurrence. No NPU, Triton, or running vLLM engine is required.
"""

import torch

from state import (
    LeapQuantSpec,
    closes_window,
    decode,
    encode,
    is_window_start,
    roundtrip,
    state_mse,
)


def _state(dv=32, dk=32, outlier=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(dv, dk, generator=g)
    if outlier:
        s[0, :] *= 30.0
        s[:, 0] *= 12.0
    return s


def test_zero_state_roundtrip_is_exact():
    spec = LeapQuantSpec()
    zero = torch.zeros(16, 16)
    assert torch.equal(roundtrip(zero, spec), zero)


def test_roundtrip_is_finite_and_preserves_shape():
    spec = LeapQuantSpec()
    state = _state()
    rec = roundtrip(state, spec)
    assert rec.shape == state.shape
    assert torch.isfinite(rec).all()


def test_compensator_tokens_reduce_error():
    state = _state(seed=1)
    with_comp = LeapQuantSpec(compensator_tokens=4)
    without_comp = LeapQuantSpec(compensator_tokens=0)
    assert state_mse(state, roundtrip(state, with_comp)) < state_mse(state, roundtrip(state, without_comp))


def test_compensator_reconstruction_rank_is_at_most_r():
    spec = LeapQuantSpec(compensator_tokens=4)
    enc = encode(_state(seed=2), spec)
    comp = enc.comp_k.float() @ enc.comp_u.float().transpose(-1, -2)
    assert torch.linalg.matrix_rank(comp).item() <= spec.compensator_tokens


def test_smoothing_balances_column_magnitudes():
    spec = LeapQuantSpec(compensator_tokens=4, smooth=True)
    state = _state(seed=3)
    enc = encode(state, spec)
    raw = state - (enc.comp_k.float() @ enc.comp_u.float().transpose(-1, -2))
    smoothed = raw / enc.scale_c

    def _col_spread(x):
        # scale_c is one factor per key-column (last axis); measure the
        # spread of column magnitudes (max over the value axis, dim=-2).
        col_max = x.abs().amax(dim=-2)
        return (col_max.max() / col_max.median()).item()

    assert _col_spread(smoothed) < _col_spread(raw)


def test_window_boundary_helpers():
    window = 16
    nct = torch.tensor([0, 15, 16, 31, 32])
    assert is_window_start(nct, window).tolist() == [False, False, True, False, True]

    new = torch.ones_like(nct)
    # n + 1 lands on a boundary for n = 15, 31
    assert closes_window(nct, new, window).tolist() == [False, True, False, True, False]


def test_leading_dims_are_preserved():
    spec = LeapQuantSpec()
    batch = torch.randn(2, 4, 32, 32)
    enc = encode(batch, spec)
    assert enc.residual.shape == (2, 4, 32, 32)
    assert decode(enc).shape == batch.shape


def test_encoding_is_deterministic():
    spec = LeapQuantSpec()
    state = _state(seed=4)
    a = encode(state, spec)
    b = encode(state, spec)
    assert torch.equal(a.residual, b.residual)
    assert torch.equal(a.scale_b, b.scale_b)


def test_invalid_spec_raises():
    for kwargs in (
        {"window": 0},
        {"bits": 4},
        {"compensator_tokens": -1},
        {"power_iters": 0},
    ):
        try:
            LeapQuantSpec(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def _gdn_final_mse(mode, steps=64, window=16):
    dv, dk = 32, 32
    g = torch.Generator().manual_seed(7)
    alphas = 0.85 + 0.14 * torch.rand(steps, dk, generator=g)
    ks = torch.randn(steps, dk, generator=g)
    ks = ks / ks.norm(dim=-1, keepdim=True)
    vs = torch.randn(steps, dv, generator=g) * 0.5
    betas = torch.rand(steps, dk, generator=g) * 0.1
    spec = LeapQuantSpec(window=window)

    def step(s, t):
        u = vs[t] - s @ betas[t]
        return s * alphas[t].unsqueeze(0) + torch.outer(u, ks[t])

    ref = torch.zeros(dv, dk)
    cur = torch.zeros(dv, dk)
    for t in range(steps):
        ref = step(ref, t)
        cur = step(cur, t)
        if mode == "per_step":
            cur = roundtrip(cur, spec)
        elif mode == "per_window" and (t + 1) % window == 0:
            cur = roundtrip(cur, spec)
    return state_mse(ref, cur)


def test_per_window_is_more_accurate_than_per_step():
    assert _gdn_final_mse("per_window") < _gdn_final_mse("per_step")
