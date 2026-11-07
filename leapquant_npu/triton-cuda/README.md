# triton-cuda Backend

Self-contained LeapQuant kernels written in **standard Triton with the CUDA
backend**. It is the place to develop and debug the kernel logic on a mature
toolchain and the NVIDIA card already available (AutoDL RTX 6000D), before the
same logic is ported to triton-ascend.

## Why this backend exists

The torch-cpu backend proves the algorithm but is not a fused kernel, and
triton-ascend depends on NPU-specific tooling. Standard Triton on CUDA is
stable and runnable on the AutoDL instance, so the `@triton.jit` logic is
written and validated here first. The kernels are device-agnostic Triton; the
later Ascend port mostly changes backend intrinsics and launch parameters, not
the algorithm.

## Stage A kernels

| Kernel | Grid | Role |
| --- | --- | --- |
| `_smoothing_scale_kernel` | `(B, cdiv(dk, BK))` | per-key `c = sqrt(mean over d_v |R|)` |
| `_int8_quant_kernel` | `(B, d_v)` | per-row `scale_b` and symmetric INT8 round/clamp |
| `_int8_dequant_kernel` | `(B, d_v)` | reconstruct `q * scale_b * scale_c` |

`B` is the flattened leading dims. Rounding is half away from zero implemented
explicitly so no libdevice `rint` intrinsic is required. Each wrapper falls
back to the torch reference when the tensor is not on CUDA, so the module
still imports and runs on CPU (the GPU test simply skips).

The power-iteration Compensator Tokens are a later stage (a data-dependent
GEMV loop); Stage A covers the `compensator_tokens=0` path end to end.

## Files

| File | Purpose |
| --- | --- |
| `state.py` | torch reference: ground truth and off-device fallback |
| `triton_cuda.py` | Stage A kernels + wrappers |
| `test_triton_cuda.py` | GPU tests, skip without CUDA/Triton |
| `README.md` | This summary |

## Run

On a CUDA machine with Triton installed (e.g. the AutoDL instance):

```bash
python -m pytest test_triton_cuda.py -v
```

On a machine without CUDA or Triton the whole module skips at module level.

## Acceptance criteria

- INT8 payload differs from the torch encoder by at most 1 (half-away vs
  half-to-even tie rule).
- Scales match the torch reference within fp32 tolerance.
- Reconstructed fp32 tensor passes `torch.testing.assert_close`
  (`rtol=atol=0.05` on the 32x32 outlier case) and stays within the M1 error
  envelope measured on the torch-cpu backend.

## Limits

- Not runnable on the CPU-only PC; validation happens on the AutoDL RTX 6000D.
- Stage A only: no power iteration and no fused decode into `gdn.py` yet.
- INT8 payload only; 4/6-bit and low-bit fused decode are out of scope.
