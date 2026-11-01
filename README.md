# awesome_quant_algo

TurboQuant vs LeapQuant: **recurrent-state (Gated-DeltaNet) 8-bit quantization** experiments on Qwen3.5 (0.8B / 9B / 35B-A3B), Colab-based.

## Tracks

- **`turboquant_exp/`** — TurboQuant-based recurrent-state quantization, TurboQuant-vs-LeapQuant comparisons, and Qwen3.5 real-model probes (0.8B / 9B / 35B-A3B, **W8A8-FP8** for 35B). Every notebook ships **CPU + A100 GPU** variants. See `qwen_quant_series_plan.md` for the project plan.
- **`leapquant_exp/`** — LeapQuant explainers (WeChat / Instagram), HF kernel mirror (`leapquant-kernels/`), pure-PyTorch port (`gdn_torch.py`, `kda_torch.py`), port feasibility study and validation.

## How to run

Open any notebook in Google Colab → Runtime → Change runtime type → **A100 GPU**. Dependencies (torch, transformers) and model weights (Qwen3.5 series) are downloaded inside Colab.

## Method at a glance

- **TurboQuant**: per-step quantization of recurrent state — random orthogonal rotation + Lloyd-Max scalar quantizer; suffers from error accumulation + extreme-coordinate outliers on recursive state.
- **LeapQuant**: window-wise quantization — high-precision buffer inside each window, single quantization at window boundary, compensator tokens + residual smoothing; mitigates both issues.
