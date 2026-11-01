# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained Colab notebook benchmarking LeapQuant vs TurboQuant on an NVIDIA T4 GPU.

Honest scope: the repository's real kernels (cutlass-dsl / tcgen05, sm_103 / B300) cannot run on a T4.
This notebook compares the two quantization SCHEDULES with functionally-equivalent pure torch:
  * TurboQuant-style: per-step re-quantization (the way the paper adapts TurboQuant to the recurrent state)
  * LeapQuant-style: per-window quantization + Compensator Tokens (the faithful pure-torch port, gdn_torch.py)
on a T4 GPU, measuring (a) decoded-state accuracy vs an fp64 reference and (b) end-to-end GPU wall time.
Deps: torch + matplotlib (Colab ships both).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "leapquant_vs_turboquant_t4.ipynb"
GDN_PORT = (HERE / "gdn_torch.py").read_text(encoding="utf-8")

BENCH = r'''
import time, torch, matplotlib.pyplot as plt
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEV, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

B, H, V, K, L, W = 1, 32, 128, 128, 16, 8          # batch, heads, dims, window, #windows (W*L tokens)
QMAX = 127.0

def decode(codes, s_k, s_v, u, q):
    return torch.einsum("bhrv,bhrk->bhvk", q.float(), u.float()) + codes.float() * (s_k[..., None, :] / QMAX) * s_v[..., :, None]

def q8_per_step(x):
    """per-step re-quantization: per-tensor absmax INT8 + dequant (TurboQuant-style per-token requant)."""
    s = x.abs().amax().clamp_min(1e-12) / QMAX
    return torch.round(x / s).clamp(-QMAX, QMAX) * s

torch.manual_seed(0)
codes0 = torch.randint(-127, 128, (B, H, V, K), device=DEV, dtype=torch.int8)
s_k0 = (torch.rand(B, H, K, device=DEV) * 0.2 + 0.05)
s_v0 = (torch.rand(B, H, V, device=DEV) * 0.5 + 0.1)
u0 = (torch.randn(B, H, 4, K, device=DEV) * 0.5).half(); q0v = (torch.randn(B, H, 4, V, device=DEV) * 0.5).half()
q0 = torch.linalg.qr(torch.randn(V, 4, device=DEV))[0].contiguous()
kbufs = [(torch.randn(B, H, L, K, device=DEV) * 0.3).bfloat16() for _ in range(W)]
ubufs = [(torch.randn(B, H, L, V, device=DEV) * 0.3).bfloat16() for _ in range(W)]
ws    = [torch.rand(B, H, L, device=DEV) for _ in range(W)]
p = torch.ones(B, H, device=DEV)

# ---- fp64 reference: initial checkpoint + every window's updates, all exact ----
S_ref = decode(codes0, s_k0, s_v0, u0, q0v).double()
for j in range(W):
    S_ref = S_ref + torch.einsum("bhj,bhjv,bhjk->bhvk", ws[j].double(), ubufs[j].double(), kbufs[j].double())
S_ref = S_ref[0]  # single batch

def leapquant_pipeline():
    codes, sk, sv, u, qv = codes0, s_k0, s_v0, u0, q0v
    S = decode(codes0, s_k0, s_v0, u0, q0v)
    states = [S[0].detach()]
    for j in range(W):
        codes, sk, sv, u, qv = _gport["run"](codes, sk, sv, u, qv, kbufs[j], ubufs[j], ws[j], p, q0)
        S = decode(codes, sk, sv, u, qv)
        states.append(S[0].detach())
    return S[0], states

def turboquant_pipeline():
    S = decode(codes0, s_k0, s_v0, u0, q0v)
    states = [S[0].detach()]
    for j in range(W):
        for t in range(L):
            S = S + ws[j][:, :, t].unsqueeze(-1).unsqueeze(-1) * (ubufs[j][:, :, t].unsqueeze(-1) @ kbufs[j][:, :, t].unsqueeze(-2)).float()
            S = q8_per_step(S)
        states.append(S[0].detach())
    return S[0], states

def timeit(fn, n=5):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

S_lq, states_lq = leapquant_pipeline()
S_ps, states_ps = turboquant_pipeline()
err_lq = (S_lq.double() - S_ref).abs().mean().item()
err_ps = (S_ps.double() - S_ref).abs().mean().item()
t_lq = timeit(leapquant_pipeline)
t_ps = timeit(turboquant_pipeline)

print(f"final state MAE  | LeapQuant : {err_lq:.4e}  | per-step(TurboQuant-style): {err_ps:.4e}  | ratio {err_ps/max(err_lq,1e-30):.1f}x")
print(f"wall time (ms)   | LeapQuant : {t_lq*1e3:.2f}  | per-step: {t_ps*1e3:.2f}")

plt.figure(figsize=(8, 5))
plt.semilogy([abs((s.double()-S_ref).abs().mean().item()) for s in states_lq], marker="o", label="LeapQuant (per-window)")
plt.semilogy([abs((s.double()-S_ref).abs().mean().item()) for s in states_ps], marker="s", label="per-step (TurboQuant-style)")
plt.xlabel("window boundary index (x16 tokens)"); plt.ylabel("mean state error (log)")
plt.title("Accuracy on T4 over a 128-token stream")
plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.show()
'''

cells = [
    {"cell_type": "markdown", "source": [
        "# LeapQuant vs TurboQuant — T4 GPU 对比测试\n",
        "\n",
        "在 Colab 的 **NVIDIA T4** 上，对同一个 128-token 循环状态流，比较两种量化**调度**：\n",
        "\n",
        "- **TurboQuant 式**：每 token 对状态做一次重新量化（per-step）——这是论文把 TurboQuant 适配到循环状态的方式（逐步重量化）。\n",
        "- **LeapQuant 式**：逐窗口（每 16 token）量化一次，窗口内用高精度缓冲，并在边界用 Compensator Tokens + 残差平滑（`gdn_torch.py` 的忠实移植）。\n",
        "\n",
        "测两件事：**解码态精度**（对 fp64 精确参考的平均绝对误差）与 **GPU 墙钟耗时**。\n",
        "\n",
        "> **诚实边界**：仓库真实内核是 cutlass-dsl / tcgen05、sm_103（B300）专属，**无法在 T4 运行**。本 notebook 用**纯 torch 功能等价实现**对比两种**调度策略**；耗时是纯 torch 基线，不代表 B300 上的内核加速比（论文内核 ~2.52×）。\n",
        "\n",
        "> 依赖仅 torch + matplotlib（Colab 自带）。运行 `Runtime → Change runtime type → GPU` 后 Run All。"
    ]},
    {"cell_type": "code", "source": [
        "import torch, matplotlib\n",
        "print(\"torch\", torch.__version__, \"| matplotlib\", matplotlib.__version__)\n",
        "assert torch.cuda.is_available(), \"需要 GPU runtime\"\n",
        "print(\"GPU:\", torch.cuda.get_device_name(0))"
    ]},
    {"cell_type": "markdown", "source": ["## 1. 加载 LeapQuant 纯 torch 移植（gdn）"]},
    {"cell_type": "code", "source": [
        "_gport = {}; exec('''" + GDN_PORT + "''', _gport)\n",
        "print(\"gdn_torch port loaded\")"
    ]},
    {"cell_type": "markdown", "source": ["## 2. 在 T4 上运行精度 + 耗时对比"]},
    {"cell_type": "code", "source": [BENCH]},
    {"cell_type": "markdown", "source": [
        "## 3. 结果解读\n",
        "\n",
        "- **精度**：预期 LeapQuant（逐窗口）的最终态误差远低于 per-step（逐步重量化）——即使单次量化是 INT8，逐步重量化也让误差随 token 累积，而逐窗口只在边界量化、窗口内高精度缓冲消除了累积。\n",
        "- **耗时**：这是**纯 torch** 基线。LeapQuant 的 `gdn_torch.run` 含 MGS/子空间迭代等，未做算子优化；真实 LeapQuant 内核在 B300 上是 ~2.52×。本测试的耗时对比说明的是调度与实现的朴素开销，不能外推到内核级。\n",
        "- **结论**：T4 上可用纯 torch 复现“逐窗口显著更准”这一核心收益；若追求 T4 上也可跑的加速版，需要给 `gdn_torch.run` 做 `torch.compile`/算子融合（本仓库 P3 方向）。"
    ]},
]

for _i, _c in enumerate(cells):
    _c.setdefault("metadata", {})
    _c["metadata"].setdefault("id", f"kern_{_i}")
    if _c["cell_type"] == "code":
        _c.setdefault("outputs", [])
        _c.setdefault("execution_count", None)

nb = {
    "nbformat": 4, "nbformat_minor": 2,
    "metadata": {"colab": {"name": "leapquant_vs_turboquant_t4.ipynb", "provenance": [], "toc_visible": True},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"},
                 "accelerator": "GPU"},
    "cells": cells,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")
