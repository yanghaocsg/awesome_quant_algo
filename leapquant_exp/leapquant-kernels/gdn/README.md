# LeapQuant Gated DeltaNet checkpoint flush

CuTe DSL window-boundary re-quantization of the recurrent state for Gated DeltaNet.
This is the flush operation, with the checkpoint format and one-round subspace
iteration specified in [NVlabs/kda#10](https://github.com/NVlabs/kda/pull/10).
It does not implement the per-token decode operation or a complete model layer.

The final accepted implementation achieved **2.551x overall**
and **2.813x on batches 128, 256 and 512** versus the
request's TileLang baseline on NVIDIA B300 SXM6 AC. All six workloads passed the
original accuracy criterion. These are the final paired KDA measurements from
2026-10-24T10:57:39Z; they are kernel timings for the deployment interface, not timings
of allocation, input packing or the functional `run` wrapper.

## Requirements and supported inputs

- NVIDIA B300, CUDA compute capability **10.3**. This release advertises only the
  tested architecture; other GPU architectures are not validated.
- CUDA 13-compatible driver; tested with PyTorch 2.12.1+cu130,
  `nvidia-cutlass-dsl[cu13]==4.6.0` and `cuda-bindings==13.3.1`.
- Python 3.12 and `kernels==0.17.2` were used for package validation.
- Fixed geometry: 32 heads, value/key dimension 128, window 16, rank 4.
  Dense benchmark batches: 16, 32, 64, 128, 256 and 512.
- `codes`: int8 `[B,32,128,128]`; `s_k`/`s_v`: fp32 `[B,32,128]`;
  `u`/`q`: fp16 `[B,32,4,128]`; `kbuf`/`ubuf`: bf16 `[B,32,16,128]`;
  `w`: fp32 `[B,32,16]`; `q0`: contiguous fp32 `[128,4]` on the same CUDA device.
- `p`: fp32 `[B,32]`, the window gate product in `(0,1]`.

Checkpoint scales are positive and the request's deployment domain has state
magnitudes below 1e4. See the complete [upstream input contract](https://github.com/NVlabs/kda/tree/48f02e6ba0e98a1841ab6f0a2d99a90b2de1a326/requests/conless-leapquant-gdn-flush)
and [reference definition](https://huggingface.co/kernels/humanfia-lab/leapquant/blob/v1/gdn/definition.json).

## Load and use

```python
import torch
from kernels import get_kernel

op = get_kernel("humanfia-lab/leapquant", version=1, trust_remote_code=True).gdn
B, H, V, K, L, R = 16, 32, 128, 128, 16, 4
codes = torch.zeros(B, H, V, K, dtype=torch.int8, device="cuda")
s_k = torch.full((B, H, K), 0.05, device="cuda")
s_v = torch.full((B, H, V), 0.3, device="cuda")
u = torch.zeros(B, H, R, K, dtype=torch.float16, device="cuda")
q = torch.zeros(B, H, R, V, dtype=torch.float16, device="cuda")
kbuf = (torch.randn(B, H, L, K, device="cuda") * 0.3).bfloat16()
ubuf = (torch.randn(B, H, L, V, device="cuda") * 0.3).bfloat16()
w = torch.rand(B, H, L, device="cuda")
p = torch.ones(B, H, device="cuda")
q0 = torch.linalg.qr(torch.randn(V, R, device="cuda"))[0].contiguous()

new_codes, new_s_k, new_s_v, new_u, new_q = op.run(
    codes, s_k, s_v, u, q, kbuf, ubuf, w, p, q0
)
```

`run` returns five fresh checkpoint tensors and leaves the supplied inputs
unchanged. It includes allocation/packing overhead. Use serial calls to `run`;
for independent concurrent streams, use an independent caller-owned `Pool` per
stream and keep calls on each pool ordered.

For serving, use the original caller-owned pool interface:

```python
pool = op.Pool(B + 8, device="cuda")  # allocates buffers and JIT-compiles kernels
sl = slice(1, B + 1)                 # slot 0 is reserved
pool.codes[sl], pool.s_k[sl], pool.s_v[sl] = codes, s_k, s_v
pool.uq[sl, :, :, :K], pool.uq[sl, :, :, K:] = u, q
pool.kbuf[sl], pool.ubuf[sl], pool.w[sl] = kbuf, ubuf, w
pool.p[sl, :, 0] = p
pool.hcnt[sl] = L
idx = torch.arange(1, B + 1, dtype=torch.int32, device="cuda")
op.flush_inplace(pool, idx, q0)       # checkpoint stays in pool; returns None
```

`idx` is a contiguous CUDA int32 vector of valid, unique slot indices. Index 0
is skipped. Only selected heads whose fill counter equals 16 are flushed;
non-due heads and unselected slots remain unchanged. A due head resets `w = 0`, `p = 1`.
The fill counter remains at `L`: the decode step interprets this as "flushed,
empty" and advances it. Flush does not clear it, because other CTAs may still
be scanning the due flags.
`q0` is the supplied fixed start basis, not the previous window's factors.

Construct the pool and perform warmup before CUDA graph capture. Refill the
window buffers and counters before each new window. Calls use the current CUDA
stream on the pool's device. Keep the pool, index vector and `q0` buffers alive
while a graph using their addresses is alive.

## Accuracy and timing

The original benchmark compares decoded checkpoints with the exactly rebuilt
fp64 state for structured edge-case heads, a second window and unstructured
in-domain random inputs at each batch size. For every head, reconstruction
error may be at most 1.01x the reference's error; nearly exact reference heads
use the original absolute cap `1e-3 * mean(abs(S))`. Outputs must be finite and
int8 codes must remain in [-127,127]. The upstream benchmark and thresholds are
included unchanged.

Deployment timing uses `flush_inplace`, all programs due, eight disjoint slot
sets rotated to avoid L2 reuse, four warmups and 16 measured iterations, and
CUDA kernel time from the PyTorch profiler. Large means batches >=128.

| Batch | Final paired speedup | Accuracy |
|---:|---:|---|
| 16 | 2.150x | pass |
| 32 | 2.249x | pass |
| 64 | 2.557x | pass |
| 128 | 2.816x | pass |
| 256 | 2.887x | pass |
| 512 | 2.739x | pass |
| Geomean, all six | 2.551x | 6/6 pass |
| Geomean, batch >=128 | 2.813x | 3/3 pass |

These measurements are workload- and hardware-specific. The functional `run`
wrapper is not the timed deployment entrypoint.

## Reproduce

Download the combined repository, then enter this component directory:

```bash
hf download humanfia-lab/leapquant --repo-type kernel --revision v1 --local-dir leapquant-kernels
cd leapquant-kernels/gdn
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r ../requirements.txt
python benchmark.py --impl ../build/torch-cuda/_gdn.py  # six original workloads
python benchmark.py                                    # original TileLang baseline
```

TileLang and `nvcc` on `PATH` are needed only to benchmark the original baseline.
The CuTe implementation JIT-compiles on first use; do not include compilation in
steady-state timings or perform first-use compilation inside graph capture.


The unchanged kernel passed all six original workloads (three input families
each) and deployment checks on B300 before consolidation. The speedup table
retains the paired optimization-run measurement, not a new baseline comparison.

## Matched precision comparison

The output dtypes are unchanged. The original baseline already uses mixed
precision. This implementation retains the rebuilt state and residual in FP32
TMEM, uses scaled FP16 projection operands, and includes local approximations
such as reciprocal operations. See [precision_comparison.json](precision_comparison.json)
for the 2026-10-26 B300 comparison on identical inputs for both implementations.

Both pass all 18 original cases (six batches and three input families).
The original judge averages the batch dimension for each head index. The report
also applies the same bound independently to each (batch item, head), an extra,
stricter check. See the repository root for those results and the KDA outlier.
Reproduce from the repository root with `python compare_precision.py gdn`.

## Files and provenance

The combined runtime is in `../build/torch-cuda/`; this component's unchanged
kernel is `_gdn.py`. The files in this directory provide its original
benchmark, baseline, definition, workloads, benchmark and precision results.
Shared dependencies and licenses are at the repository root.

The runtime kernel is byte-for-byte the final accepted submission
`8f705942197446f4` (SHA-256 `a734b59e75a8c8e870d68280c59c4fe05e9e3d2334c4d7e62d7101c67c9cd9b6`), from a 48-hour KDA experiment
with GPT 6.1 and Claude Opus 5.5. Request sources are pinned to public commit
`48f02e6ba0e98a1841ab6f0a2d99a90b2de1a326`. The release adds a public import wrapper, metadata
and precision comparison tooling; it does not change the kernel algorithm. Apache-2.0;
see `../LICENSE.txt` and `../NOTICE`.

## Available functions

- `run`
- `Pool`
- `flush_inplace`
