# LeapQuant Three-Backend Plan

LeapQuant is landed on three interchangeable backends that share one torch
reference. Backend 1 (torch-cpu) is already in place and verified; backend 3
(triton-ascend) is **paused**. Backend 2 (triton-cuda) is built now.

## Backend overview

| | (1) torch-cpu | (2) triton-cuda | (3) triton-ascend |
| --- | --- | --- | --- |
| Stack | pure torch on CPU | standard Triton, CUDA backend | `vllm.triton_utils` + triton-ascend |
| Hardware | CPU | NVIDIA GPU (AutoDL RTX 6000D) | Ascend NPU |
| Deps | torch | triton, torch (CUDA) | triton-ascend, CANN |
| Where validated | this PC (CPU) | AutoDL CUDA instance | NPU machine |
| Custom kernel | none | yes (Stage A first) | yes (ported from 2) |
| Status | **done** | build now | **paused** |

### Why this order

- **(1) torch-cpu** is the device-agnostic reference in `state.py`. It holds no
  hardware dependency, runs anywhere, and is the numerical ground truth that
  every later backend is checked against. It is already verified (10 unit
  tests + 6 smoke checks).
- **(2) triton-cuda** writes the kernels against mature, standard Triton and
  validates them on the NVIDIA card already available (AutoDL), the practical
  place to debug kernel logic without NPU-specific tooling.
- **(3) triton-ascend** later ports the already-correct `@triton.jit` logic to
  the Ascend backend. Most kernel code is shared; only backend intrinsics,
  launch parameters and memory constraints change. Paused until 2 is stable.

## Shared vs backend-specific code

```
leapquant/
  state.py            # (1) torch-cpu reference; shared ground truth + fallback
  config.py           # shared config/validation
  triton_cuda.py      # (2) standard-triton kernels (import triton), to be added
  kernels.py          # (3) ascend-style draft, kept but paused
```

- `state.py` is the single source of truth. Each backend is checked against it
  and the wrappers use it as the off-device fallback.
- (1) needs no extra file: `state.py` itself is the torch-cpu backend.
- (2) is a new file using `import triton` / `import triton.language as tl`
  rather than `vllm.triton_utils`, so it runs on a stock CUDA environment.

## Tests

```
tests/ut/ops/
  test_gdn_leapquant.py             # (1) torch-cpu reference (green, 10)
  test_gdn_leapquant_triton_cuda.py # (2) skip unless CUDA + triton present
```

The backend test feeds identical inputs to the kernel path and `state.py`,
then compares on the same device. It uses `pytest.importorskip` and a device-
availability guard so it silently skips on this CPU machine.

## Milestones

- **M-A (backend 1)**: done — `state.py` on CPU, 10 unit tests and 6 verify
  smoke checks all passing.
- **M-B (backend 2)**: `triton_cuda.py` Stage A kernels (smoothing, int8
  quant, dequant) + CUDA test; run on AutoDL RTX 6000D.
- **M-C (backend 3, paused)**: port to triton-ascend, then fuse into `gdn.py`.

## Acceptance criteria

- **(1) torch-cpu**: verified — 10 unit tests pass and the 6 verify checks
  pass; with smoothing the round-trip MSE is on the order of 1e-3 (see the
  measured M1 numbers in the README), shapes and dtypes preserved.
- **(2) triton-cuda**: int8 payload differs from the torch encoder by at most
  1 (half-away vs half-to-even tie rule); reconstructed fp32 tensor passes
  `torch.testing.assert_close` and stays within the M1 error envelope.

## Environment constraints

- This PC: CPU-only, no triton / torch-npu — runs backend 1 and fallback logic.
- AutoDL RTX 6000D: CUDA card — validates backend 2 only; it cannot run
  triton-ascend.
- NPU machine (A2/A3): for backend 3 when it is resumed.
