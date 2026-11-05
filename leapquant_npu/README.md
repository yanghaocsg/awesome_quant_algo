# LeapQuant NPU (M1)

Development workspace for bringing LeapQuant (arXiv:2609.38166) per-window
recurrent-state quantization to Ascend NPU via
[vllm-ascend](https://github.com/vllm-project/vllm-ascend).

- RFC: https://github.com/vllm-project/vllm-ascend/issues/21097
- Scope: NPU-only, enabled through `additional_config`, no upstream change.
- M1 goal: algorithm + numerical baseline only. Default-off, keeps the fp32
  state physically, and delivers **no speedup yet**; a quantize->dequantize
  round-trip at each window boundary is what reproduces per-window semantics.

## Layout

```
vllm_ascend/quantization/methods/mamba_state/
    __init__.py
    leapquant/
        __init__.py
        config.py     # LeapQuantConfig.from_vllm_config + startup checks
        state.py      # encode / decode / fit_compensator_tokens / window helpers
tests/ut/ops/test_gdn_leapquant.py   # 10 CPU test cases
tools/verify_m1.py                   # numeric smoke check
```

Paths mirror the target vllm-ascend tree so the contents can be copied over
directly.

## Run

```bash
# unit tests (CPU, torch-only)
python -m pytest tests/ut/ops/test_gdn_leapquant.py -v

# numeric smoke check (prints round-trip MSE and per-window vs per-step)
python tools/verify_m1.py
```

## Status

Reference implementation, not yet validated on device. The decode kernel
integration in `ops/gdn.py` and the NPU accuracy A/B are the next steps.
