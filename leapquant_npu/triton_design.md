# LeapQuant Triton Kernel Design (NPU)

Maps the torch reference in `state.py` onto Ascend Triton kernels. The local
machine is CPU-only (no `triton`, `triton-ascend`, or `torch-npu`), so these
kernels are written against the vllm-ascend Triton style and **must be compiled
and validated on an NPU machine**; every wrapper keeps a torch fallback so the
logic and the CPU unit tests still run here.

## Conventions

- Import through `vllm.triton_utils` (`tl`, `triton`), never `import triton`
  directly, matching `ops/triton/mamba/state_index.py`.
- A state tensor is `[B, dv, dk]` where `B` is the flattened leading dims
  (slot x head). Each kernel takes explicit strides rather than assuming
  contiguity.
- Every wrapper validates inputs, then branches on `tensor.device.type`:
  `npu` -> launch kernel; otherwise -> torch reference (same numbers).
- Compile-time sizes use `tl.constexpr`; reductions use `tl.sum(..., axis=...)`.

## Kernel split

| torch reference | Triton kernel | grid | reduction |
| --- | --- | --- | --- |
| `scale_c = sqrt(mean over dv |R|)` | `_smoothing_scale_kernel` | `(B, cdiv(dk, BK))` | sum over dv |
| per-row `scale_b` + int8 round/clamp | `_int8_quant_kernel` | `(B, dv)` | amax over dk |
| `q*scale_b*scale_c` | `_int8_dequant_kernel` | `(B, dv)` | none |
| `fit_compensator_tokens` (power iter) | `_power_iter_kernel` | `(B,)` | staged GEMV loop |

## Stages

- **Stage A (this change)**: `_smoothing_scale_kernel`, `_int8_quant_kernel`,
  `_int8_dequant_kernel`. These are regular reduction / elementwise kernels
  with no loop-carried dependency, so they are the safest first step and cover
  the `compensator_tokens=0` path end to end.
- **Stage B**: `_power_iter_kernel` for the Compensator Tokens. Power iteration
  needs repeated GEMV (`u = R v`, `v = R^T u`) with normalization and rank
  deflation inside one kernel; it is isolated because it is the only kernel
  with a data-dependent inner loop.
- **Stage C**: fuse quant/dequant into the `gdn.py` decode/prefill hooks and
  replace the torch round-trip used by M1.

## Validation on NPU

```bash
# same input to kernel wrapper and torch reference, must match within fp32 noise
python -m pytest tests/ut/ops/test_gdn_leapquant_triton.py -v
```

Each test feeds identical random states to both paths and uses
`torch.testing.assert_close`. The kernel rounds half away from zero while the
torch encoder rounds half to even, so the int8 payload may differ by at most 1
on exact ties; the reconstructed fp32 tensor is compared with `assert_close`.
First run on a small `[B, dv, dk]` (e.g. 32x32), then real head
dimensions (128) before wiring into `gdn.py`. The CPU test stubs
`vllm.triton_utils` because vLLM is not installed locally; the fallback path
never touches Triton.

## Out of scope here

- 4/6-bit payloads and the fused low-bit decode kernel (M3).
- TP / prefix-caching / spec-decode layouts; the unsupported matrix from the
  RFC still applies.
