# torch-cpu Backend

Self-contained, device-agnostic PyTorch implementation of LeapQuant
(arXiv:2609.38166) and the numerical ground truth for the other backends. It
runs on CPU with stock PyTorch: no NPU, no Triton, no running vLLM engine.

## What this backend is

LeapQuant quantizes the **recurrent state** of linear-attention models (GDN /
KDA) once per window instead of once per decode step, and preserves the
state's dominant structure in a few high-precision terms. The three paper
components land in `state.py` as follows, using the vllm-ascend `ssm_state`
layout `[..., d_v, d_k]` (the transpose of the paper's `S`):

1. **Per-window quantization** — symmetric INT8 with one scale per value
   channel (per row), `scale_b`; applied only at a window boundary.
2. **Compensator Tokens** — the top-`r` rank-one terms `K U^T` in fp16, fit by
   deterministic power iteration (`fit_compensator_tokens`).
3. **Residual smoothing** — one factor per key column, `scale_c =
   sqrt(mean over d_v |R|)`, which balances column magnitudes and tames
   outliers before INT8.

## Why a window-boundary round-trip is sufficient

At a boundary the state is quantized and dequantized once, then iteration
continues in fp32 inside the window starting from that reconstructed state.
This is numerically equivalent to the paper's equation (4): within a window all
updates stay high precision and only the starting point carries quantization
error. Because no kernel is changed, the M1 path needs only two hooks in the
engine; the quantizer itself is pure PyTorch.

## Paper equations

The math below follows the paper's notation. The paper keeps the state as
`S in R^{d_k x d_v}` while the vllm-ascend `ssm_state` is its transpose
`[..., d_v, d_k]`; the equations are identical, only the axes are swapped.

**Eq. (4) — high-precision updates inside a window.** With window length `p`
and a quantized boundary state `\hat S_0 = RoundTrip(S_0)`,

$$\tilde S_t = F_t(\tilde S_{t-1}), \quad \tilde S_0 = \hat S_0, \quad t = 1,\dots,p$$

`F_t` is the fp32 recurrent update. Quantization enters only at the window
start; every in-window step stays high precision. This is why one round-trip
at the boundary is sufficient.

**Eq. (5) — quantized decomposition of the state.**

$$S \approx K U^\top + R, \qquad R = B\,Q\,C$$

where `Q` is the int8 payload, `B` (`scale_b`) is the per-value (per-row)
symmetric scale, `C` (`scale_c`) is the per-key (per-column) smoothing, and
`K U^T` the rank-`r` Compensator Tokens. The encoder subtracts the low-rank
term first, then quantizes the residual:

$$R = S - K U^\top, \qquad
Q = \mathrm{clip}\Big(\mathrm{round}\big((R/C)/B\big),\,-127,\,127\big)$$

**Eq. (6) — Compensator Tokens by power iteration.** Each rank-one component
is the residual's dominant singular direction, found by repeated GEMV with
normalization, then deflated:

$$u \leftarrow R v,\quad v \leftarrow R^\top u, \qquad
R \leftarrow R - \sigma\, u v^\top, \quad \sigma = u^\top R v$$

Repeated `r` times to build `K` and `U` (`fit_compensator_tokens`). This
implementation uses a deterministic warm start instead of a random vector.

**Eq. (7) — reconstruction (decode).**

$$\hat S = (Q \cdot B)\cdot C + K U^\top$$

The int8 payload is scaled back elementwise and the low-rank term is added
(`decode`).

## Files

| File | Purpose |
| --- | --- |
| `state.py` | Spec, encoded payload, power iteration, encode/decode/roundtrip, window helpers |
| `test_state.py` | 10 unit tests (round-trip, rank, smoothing, window logic, GDN simulation) |
| `verify.py` | 6 smoke checks that print the accuracy numbers |
| `gdn.md` | GDN recurrent state in plain terms + decode-time inference path |
| `kda.md` | KDA per-channel variant, same treatment |
| `deltanet_reading_note.md` | Translation + analysis of the Gated DeltaNet chapter (GatedDeltaNet code, KV-cache vs state arithmetic, layout/semantics cross-check against this backend) |
| `kv_vs_state_arithmetic.py` | Recomputes KV-cache vs recurrent-state bytes for any geometry: crossover `T*`, per-length ratios, GQA / dtype / LeapQuant state variants |
| `README.md` | This summary |

## Run

```bash
python -m pytest test_state.py -v
python verify.py
```

## Measured results (CPU)

- Unit tests: **10 passed**.
- Smoke checks: **6 passed**.
- Single 32x32 round-trip with outliers:
  - INT8 only: MSE 2.48e-02, max|err| 1.59
  - INT8 + smoothing: MSE 1.62e-03
  - INT8 + comp(4) + smoothing: MSE 1.39e-03, max|err| 0.44
- Compensator: reconstruction rank 4, `|comp| / |state|` about 0.008.
- 64-step GDN simulation: per-window MSE 7.76e-07 vs per-step MSE 4.93e-06 —
  per-window is about 6.3x more accurate at this short horizon (the paper
  reports a larger gap on long sequences).
- Batched `[2,4,64,64]` round-trip: MSE 3.37e-05, shape fully preserved.

## Design choices

- Deterministic power-iteration warm start (the residual's own energy
  direction) so results are bit-reproducible across runs and tests.
- Smoothing and scales are floored with a small epsilon so the rescale stays
  invertible on zero rows.
- The encoded payload keeps the input leading dims, so a batch of slots/heads
  is encoded in one call.

## Scope and limits

- INT8 payload only; 4/6-bit need a different payload format.
- This is a reference and validation path, not the fused production kernel.
- The unsupported matrix from the RFC still applies (prefix caching,
  spec-decode cross-window state, PD separation / KV transfer).
