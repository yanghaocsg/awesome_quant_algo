"""CPU smoke check for the LeapQuant M1 reference implementation.

Not a unit test: it prints the numbers I need to pick defensible thresholds
for tests/ut/ops/test_gdn_leapquant.py.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (  # noqa: E402
    LeapQuantSpec,
    encode,
    decode,
    roundtrip,
    state_mse,
)

torch.manual_seed(0)

DV, DK = 64, 64
SPEC = LeapQuantSpec(window=16, bits=8, compensator_tokens=4, smooth=True)


def make_state(scale=1.0, outlier=True):
    s = torch.randn(DV, DK) * scale
    if outlier:
        s[0, :] *= 30.0
        s[:, 0] *= 12.0
    return s


print("=" * 66)
print("[1] single-shot roundtrip on one state (32x32, outliers)")
s = make_state()[:32, :32]
for label, spec in [
    ("int8 only            ", LeapQuantSpec(compensator_tokens=0, smooth=False)),
    ("int8 + smoothing     ", LeapQuantSpec(compensator_tokens=0, smooth=True)),
    ("int8 + comp(4)       ", LeapQuantSpec(compensator_tokens=4, smooth=False)),
    ("int8 + comp(4) + smoo", SPEC),
]:
    rec = roundtrip(s, spec)
    print(f"  {label}  MSE={state_mse(s, rec):.3e}  max|err|={(s - rec).abs().max():.3e}")

print("=" * 66)
print("[2] rank of the compensator contribution (must be <= r)")
enc = encode(make_state(), SPEC)
comp = enc.comp_k.float() @ enc.comp_u.float().transpose(-1, -2)
print(f"  r={enc.comp_tokens}  matrix_rank={torch.linalg.matrix_rank(comp).item()}")
print(f"  |comp| / |state| = {(comp.norm() / make_state().norm()):.3f}")

print("=" * 66)
print("[3] smoothing balances key-row magnitudes")
raw = make_state() - comp
c = enc.scale_c.squeeze()
print(f"  raw residual row-max/row-median = {(raw.abs().max(-1).values / raw.abs().median()).max():.2f}")
print(f"  smoothed     row-max/row-median = {((raw / c).abs().max(-1).values / (raw / c).abs().median()).max():.2f}")


print("=" * 66)
print("[4] GDN recurrence over 64 steps: per-step vs per-window (paper Fig.3)")


def run(mode, steps=64, window=16, spec=SPEC):
    dk, dv = 32, 32
    device = torch.device("cpu")
    torch.manual_seed(7)
    alphas = (0.85 + 0.14 * torch.rand(steps, dk)).to(device)
    ks = torch.randn(steps, dk)
    ks = ks / ks.norm(dim=-1, keepdim=True)
    vs = torch.randn(steps, dv) * 0.5
    betas = torch.rand(steps, dk) * 0.1

    def step(s, t):
        u = vs[t] - s @ betas[t]
        return s * alphas[t].unsqueeze(0) + torch.outer(u, ks[t])

    ref = torch.zeros(dv, dk)
    s = torch.zeros(dv, dk)
    for t in range(steps):
        ref = step(ref, t)
        s = step(s, t)
        if mode == "per_step":
            s = roundtrip(s, spec)
        elif mode == "per_window":
            if (t + 1) % window == 0:
                s = roundtrip(s, spec)
    return state_mse(ref, s), ref, s


for mode in ("per_window", "per_step"):
    mse, ref, s = run(mode)
    print(f"  {mode:11s}  final MSE vs fp32 = {mse:.4e}")

mse_ps, _, _ = run("per_step")
mse_pw, _, _ = run("per_window")
print(f"  per-window is {mse_ps / max(mse_pw, 1e-30):.1f}x more accurate")

print("=" * 66)
print("[5] spec validation")
for bad in (LeapQuantSpec(window=0), LeapQuantSpec(bits=4), LeapQuantSpec(compensator_tokens=-1)):
    try:
        bad.__post_init__()
        print("  FAIL: no error raised")
    except ValueError as e:
        print(f"  ok: {e}")

print("=" * 66)
print("[6] batching: leading dims preserved")
batch = torch.randn(2, 4, DV, DK)
enc_b = encode(batch, SPEC)
print(f"  in  {tuple(batch.shape)} -> residual {tuple(enc_b.residual.shape)}  scale_b {tuple(enc_b.scale_b.shape)}")
print(f"  out {tuple(decode(enc_b).shape)}  mse={state_mse(batch, decode(enc_b)):.3e}")
