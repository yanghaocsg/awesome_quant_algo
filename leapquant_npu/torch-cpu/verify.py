"""CPU smoke check for the self-contained torch-cpu LeapQuant implementation.

Not a unit test: it prints the numbers used to pick defensible thresholds in
test_state.py.
"""

import torch

from state import LeapQuantSpec, decode, encode, roundtrip, state_mse

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
print("[3] smoothing balances key-column magnitudes")
raw = make_state() - comp
c = enc.scale_c.squeeze()
print(f"  raw residual col-max/col-median = {(raw.abs().amax(-2) / raw.abs().median()).max():.2f}")
print(f"  smoothed     col-max/col-median = {(((raw / c).abs().amax(-2)) / (raw / c).abs().median()).max():.2f}")

print("=" * 66)
print("[4] GDN recurrence over 64 steps: per-step vs per-window (paper Fig.3)")


def run(mode, steps=64, window=16, spec=SPEC):
    dk, dv = 32, 32
    torch.manual_seed(7)
    alphas = 0.85 + 0.14 * torch.rand(steps, dk)
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
        elif mode == "per_window" and (t + 1) % window == 0:
            s = roundtrip(s, spec)
    return state_mse(ref, s), ref, s


for mode in ("per_window", "per_step"):
    mse, _, _ = run(mode)
    print(f"  {mode:11s}  final MSE vs fp32 = {mse:.4e}")

mse_ps, _, _ = run("per_step")
mse_pw, _, _ = run("per_window")
print(f"  per-window is {mse_ps / max(mse_pw, 1e-30):.1f}x more accurate")

print("=" * 66)
print("[5] spec validation")
for kwargs in (
    {"window": 0},
    {"bits": 4},
    {"compensator_tokens": -1},
):
    try:
        LeapQuantSpec(**kwargs)
        print("  FAIL: no error raised")
    except ValueError as e:
        print(f"  ok: {e}")

print("=" * 66)
print("[6] batching: leading dims preserved")
batch = torch.randn(2, 4, DV, DK)
enc_b = encode(batch, SPEC)
print(f"  in  {tuple(batch.shape)} -> residual {tuple(enc_b.residual.shape)}  scale_b {tuple(enc_b.scale_b.shape)}")
print(f"  out {tuple(decode(enc_b).shape)}  mse={state_mse(batch, decode(enc_b)):.3e}")
