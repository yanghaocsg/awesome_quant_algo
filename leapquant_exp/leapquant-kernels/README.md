---
library_name: kernels
license: apache-2.0
tags:
  - cuda
  - cute-dsl
  - cutlass
  - blackwell
  - b300
  - leapquant
  - linear-attention
---

# LeapQuant

One repository containing the Gated DeltaNet (GDN) and Kimi Delta Attention
(KDA) window-boundary state re-quantization kernels. Both use CuTe DSL and
Blackwell tensor cores. The kernel algorithms and bytes are unchanged from
their final accepted optimization submissions.

| Component | Documentation and benchmark | Final paired speedup, all six batches | Batches >=128 |
|---|---|---:|---:|
| Gated DeltaNet | [gdn/](gdn/README.md) | 2.551x | 2.813x |
| Kimi Delta Attention | [kda/](kda/README.md) | 2.999x | 3.128x |

These are B300 deployment-kernel measurements against the request's TileLang
baseline, from the 2026-10-24 paired evaluations. They do not measure a complete
model or the allocation/packing overhead of the functional interface. Both
implementations passed the original six workloads and deployment validation;
the component directories retain the reports and exact source hashes.

## Install and load

Tested on NVIDIA B300 (compute capability 10.3), Python 3.12,
PyTorch 2.12.1+cu130 and nvidia-cutlass-dsl[cu13] 4.6.0. Other architectures
are not advertised. See each component README for shapes and input contracts.

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install "nvidia-cutlass-dsl[cu13]==4.6.0" cuda-bindings==13.3.1 kernels==0.17.2 huggingface-hub==2.2.0
```

```python
from kernels import get_kernel

op = get_kernel("humanfia-lab/leapquant", version=1, trust_remote_code=True)
gdn = op.gdn  # run(...), Pool(...), flush_inplace(...)
kda = op.kda  # run(...), Pool(...), flush_inplace(...)
```

The `main` and `v1` branches both contain the complete release. `v1` is the
version loaded above. The runtime sources are `build/torch-cuda/_gdn.py` and
`build/torch-cuda/_kda.py`; metadata, the public import wrapper and the runtime
license are alongside them. Each component directory contains its README,
original workload, benchmark, baseline and reference definition, plus benchmark
and precision results. Dependency pins and Apache-2.0 licensing are shared at the root.

## Precision

See the standalone [precision analysis](PRECISION_ANALYSIS.md) for the original
and current arithmetic paths, deliberate approximations, matched measurements,
and validation limits. Input/output formats and original scoring thresholds
are unchanged; both implementations pass all 18 original comparison cases.

The additional individual-state check found one current KDA outlier in the
batch-128 random case (error ratio 1.01256057 against the individual limit 1.01).
The original benchmark averages the batch dimension; the extra check treats
individual states separately. Full records are linked in the analysis document.
Reproduce with `python compare_precision.py gdn` and
`python compare_precision.py kda`.

## Reproduce

```bash
hf download humanfia-lab/leapquant --repo-type kernel --revision v1 --local-dir leapquant-kernels
cd leapquant-kernels
python -m pip install -r requirements.txt
cd gdn
python benchmark.py --impl ../build/torch-cuda/_gdn.py
cd ../kda
python benchmark.py --impl ../build/torch-cuda/_kda.py
```

Run `python benchmark.py` in either component directory for the original
TileLang baseline (requires the pinned TileLang and nvcc on PATH). Construct and
warm the caller-owned pool before CUDA graph capture. The flush operates only on
heads whose counter equals 16; the counter remains at 16 for the decode step.
