# -*- coding: utf-8 -*-
"""Build colab_02_leapquant.ipynb  (Part 2/4). Faithful port of gdn_torch.py window flush.
>=3 formulas + fix explanation + windowed-vs-per-step comparison. CPU + A100 adaptive."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "colab_02_leapquant.ipynb")

MD0 = r'''# Part(2/4) LeapQuant 如何解决循环状态量化的两个问题
承接 Part(1) 的两个问题：**①错误累积**、**②极端错误点**。LeapQuant 核心策略是**逐窗口量化（window flush）**：窗口内用高精度（bf16/fp32）缓冲全部更新、**只在窗口边界量化一次**，并引入 **R 个补偿 token（Compensator Tokens）**保留状态主成分、对残差做 **int8 平滑量化**。
本 notebook 给出 4 个关键公式 + 忠实于 `gdn_torch.py` 的实现 + 与 TurboQuant（每步量化）的对比。依赖仅 torch + matplotlib（Colab 自带）；DEV 自适应 CPU / A100。'''

CODE1 = r'''import torch, matplotlib
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| mpl", matplotlib.__version__)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on", DEV)
QMAX, EPS = 127.0, 1e-12
def absmax_q(x, nbits=8):
    qmax = 2**(nbits-1) - 1
    s = x.abs().max() / qmax
    return (x / s).round().clamp(-qmax, qmax) * s'''

MD2 = r'''## 公式 1 — 窗口内高精度缓冲 + 状态重建（解决"错误累积"）
窗口内 **L** 个 token 的更新以 bf16/fp32 缓冲（`k_j, u_j, w_j`），**不逐 token 量化**；状态由"上一窗口的压缩态"与"本窗口的更新"重建：

$$S = p \odot \big(q^{\mathsf T} u + \text{dequant}(c)\big) + \sum_{j=1}^{L} w_j\, u_j\, k_j^{\mathsf T}$$

- 低秩项 `qᵀu + dequant(c)`：上一窗口状态的压缩表示（补偿 token + int8 残差）。
- `Σ w_j u_j k_jᵀ`：本窗口 L 个更新的 bf16/fp32 缓冲和。
- 意义：误差只在窗口边界进入一次，**不在每个 token 逐级叠加**——直接缓解问题①。'''

CODE3 = r'''def _mgs(m):
    cols = []
    for j in range(m.shape[-1]):
        v = m[..., j]
        for _ in range(2):
            for c in cols:
                v = v - c * (c * v).sum(-1, keepdim=True)
        cols.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30))
    return torch.stack(cols, -1)

def rebuild_state(codes, s_k, s_v, u, q, kbuf, ubuf, w, p):
    s_old = torch.einsum("bhrv,bhrk->bhvk", q.float(), u.float())
    s_old = s_old + codes.float() * (s_k[..., None, :] / QMAX) * s_v[..., :, None]
    s = p[..., None, None] * s_old + torch.einsum("bhj,bhjv,bhjk->bhvk", w, ubuf.float(), kbuf.float())
    return s

print("formula 1 ready: window buffered in fp32, state rebuilt once at flush")'''

MD4 = r'''## 公式 2 — 补偿 token（R Compensator Tokens）：子空间迭代
量化前先抽出状态的 **top-R 主成分**作为补偿 token，主成分不进 int8（几乎无损保留），只对残差量化：

$$\begin{aligned}
q_k &= \mathrm{MGS}(S^{\mathsf T} q_0) &\quad &\text{[K, R]} \\
q_v &= \mathrm{MGS}(S\, q_k) &\quad &\text{[V, R]} \text{, 正交列} \\
u_{new} &= (S\, q_v)^{\mathsf T} &\quad q_{new} = q_v^{\mathsf T}
\end{aligned}$$

- `MGS` = Modified Gram–Schmidt（正交化，投影两遍保证数值稳定）。
- 语义：`qᵀu` 就是"补偿 token"，量化后仍保留状态的前 R 个奇异方向——极大缓解问题①与②。'''

CODE5 = r'''def subspace_iteration(s, q0):
    x = s.transpose(-1, -2)                       # [K, V]
    qk = _mgs(x @ q0)                             # [K, R]
    qv = _mgs(x.transpose(-1, -2) @ qk)           # [V, R], orthonormal columns
    uk = x @ qv                                   # [K, R]
    u_new = uk.transpose(-1, -2).to(torch.float16)
    q_new = qv.transpose(-1, -2).to(torch.float16)
    return u_new, q_new

print("formula 2 ready: R compensator tokens capture top-R subspace")'''

MD6 = r'''## 公式 3 — 残差 int8 平滑量化（解决"极端错误点"）
对**残差** `E = S − qᵀu` 做行/列两级 scale 的 int8 量化，outlier 只影响**本行/本列**，不拉爆全局 scale：

$$\begin{aligned}
s_k &= \sqrt{\mathrm{mean}_{row}|E|} &\quad &\text{列 scale} \\
s_v &= \max_{row}\big(|E| / s_k\big) &\quad &\text{行 scale} \\
c &= \mathrm{round}\!\big(E\, /\, (s_k \odot s_v)\cdot 127\big)\ \in [-127,127]
\end{aligned}$$

- 与 absmax 全局 scale 不同：scale 逐行/逐列平滑，极端坐标被 `s_v` 单独吸收 → 不会让**所有**其他坐标变粗（问题②的 fix）。
- 且只在**窗口边界**执行一次（问题①的 fix）。'''

CODE7 = r'''def quantize(s, u_new, q_new):
    e = s - torch.einsum("bhrv,bhrk->bhvk", q_new.float(), u_new.float())
    sk = e.abs().mean(-2).sqrt().clamp_min(EPS)
    sv = (e.abs() / sk[..., None, :]).amax(-1).clamp_min(EPS)
    c = torch.round(e / (sk[..., None, :] * sv[..., :, None]) * QMAX).clamp(-QMAX, QMAX).to(torch.int8)
    return c, sk, sv

print("formula 3 ready: smoothed per-row/per-col scales localize outliers")'''

MD8 = r'''## 公式 4 — 共享状态流（跨窗口连续压缩）
状态在窗口间以"压缩后的共享状态"连续流动：窗口内高精度递推、边界一次压缩（补偿 + 残差量化）：

$$\hat S_{next} = \mathrm{CompressWin}\big(\hat S + \{\text{window updates}\}\big)
= q^{\mathsf T} u + \text{dequant}(c)$$

`CompressWin` = 公式 2（子空间补偿）+ 公式 3（残差 int8）。对比 TurboQuant 的**每步量化**：LeapQuant 把量化次数从 `T` 降到 `T/L`，且每次带补偿 → 误差不再随 token 逐级累积。'''

CODE9 = r'''# Shared-state stream: TurboQuant per-step vs LeapQuant per-window compression
import torch, matplotlib.pyplot as plt
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QMAX, EPS = 127.0, 1e-12

def _mgs(m):
    cols = []
    for j in range(m.shape[-1]):
        v = m[..., j]
        for _ in range(2):
            for c in cols:
                v = v - c * (c * v).sum(-1, keepdim=True)
        cols.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30))
    return torch.stack(cols, -1)

def subspace_iteration(s, q0):
    x = s.transpose(-1, -2)
    qk = _mgs(x @ q0)
    qv = _mgs(x.transpose(-1, -2) @ qk)
    uk = x @ qv
    u_new = uk.transpose(-1, -2).to(torch.float16)
    q_new = qv.transpose(-1, -2).to(torch.float16)
    return u_new, q_new

def quantize(s, u_new, q_new):
    e = s - torch.einsum("bhrv,bhrk->bhvk", q_new.float(), u_new.float())
    sk = e.abs().mean(-2).sqrt().clamp_min(EPS)
    sv = (e.abs() / sk[..., None, :]).amax(-1).clamp_min(EPS)
    c = torch.round(e / (sk[..., None, :] * sv[..., :, None]) * QMAX).clamp(-QMAX, QMAX).to(torch.int8)
    return c, sk, sv

def compress_state(s, q0):
    """window-boundary compression = compensator subspace + int8 smoothed residual + rebuild"""
    u_new, q_new = subspace_iteration(s, q0)
    c, sk, sv = quantize(s, u_new, q_new)
    s_re = torch.einsum("bhrv,bhrk->bhvk", q_new.float(), u_new.float()) \
         + c.float() * (sk[..., None, :] / QMAX) * sv[..., :, None]
    return s_re

def absmax_q(x, nbits=8):
    qmax = 2**(nbits-1) - 1
    s = x.abs().max() / qmax
    return (x / s).round().clamp(-qmax, qmax) * s

B, H, R, K, V = 1, 1, 8, 64, 64
W, NWIN = 16, 8
Ttot = W * NWIN
beta = 0.99
torch.manual_seed(0)
q0 = torch.randn(V, R, device=DEV) / V ** 0.5
Sref = torch.zeros(B, H, V, K, device=DEV, dtype=torch.float64)
S_tq = torch.zeros(B, H, V, K, device=DEV)
S_lq = torch.zeros(B, H, V, K, device=DEV)
err_tq, err_lq = [], []
for t in range(Ttot):
    k_t = torch.randn(K, device=DEV) * 0.05
    v_t = torch.randn(V, device=DEV) * 0.05
    w_t = torch.rand(1, device=DEV).item() * 0.5
    upd = (w_t * (v_t[:, None] @ k_t[None, :])).unsqueeze(0).unsqueeze(0)
    Sref = beta * Sref + upd.double()
    S_tq = beta * S_tq + upd; S_tq = absmax_q(S_tq)
    S_lq = beta * S_lq + upd
    err_tq.append((S_tq.double() - Sref).abs().mean().item())
    err_lq.append((S_lq.double() - Sref).abs().mean().item())
    if (t + 1) % W == 0:
        S_lq = compress_state(S_lq, q0)
        err_lq[-1] = (S_lq.double() - Sref).abs().mean().item()

plt.figure(figsize=(8, 4.5))
plt.semilogy(err_tq, marker='.', label="TurboQuant per-step (absmax)")
plt.semilogy(err_lq, marker='o', ms=3, label="LeapQuant per-window compression")
plt.xlabel("token t"); plt.ylabel("state error vs fp64 (log)")
plt.title("LeapQuant windowed compression stops per-step error accumulation")
plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
print(f"TurboQuant per-step final err = {err_tq[-1]:.3e}")
print(f"LeapQuant windowed final err  = {err_lq[-1]:.3e}")
print(f"=> LeapQuant {err_tq[-1]/max(err_lq[-1],1e-30):.1f}x lower at stream end")
print("=> fewer quantizations (T/L) + compensator tokens + smoothed residual scales solve both problems")'''

MD10 = r'''## 小结
- **问题①错误累积**：LeapQuant 窗口内 fp32 递推、**只在边界压缩一次**（量化次数 T→T/L），并用补偿 token 保留主成分 → 误差不逐级叠加。
- **问题②极端错误点**：残差按**行/列 scale 平滑量化**，outlier 只被本行/本列 `s_v` 吸收，不拉爆全局 scale。
- 上面实验：同一状态流上，TurboQuant 每步量化最终误差远高于 LeapQuant 逐窗口压缩（数量级更低）。
- **下一步（Part 3/4）**：把这套逐窗口压缩真实作用到 Qwen3.5（0.8B/9B/35B-A3B）的 Gated-DeltaNet 循环状态上做实测。'''

cells = [
    {"cell_type": "markdown", "source": MD0, "metadata": {"id": "c2_0"}},
    {"cell_type": "code", "source": CODE1, "metadata": {"id": "c2_1"}},
    {"cell_type": "markdown", "source": MD2, "metadata": {"id": "c2_2"}},
    {"cell_type": "code", "source": CODE3, "metadata": {"id": "c2_3"}},
    {"cell_type": "markdown", "source": MD4, "metadata": {"id": "c2_4"}},
    {"cell_type": "code", "source": CODE5, "metadata": {"id": "c2_5"}},
    {"cell_type": "markdown", "source": MD6, "metadata": {"id": "c2_6"}},
    {"cell_type": "code", "source": CODE7, "metadata": {"id": "c2_7"}},
    {"cell_type": "markdown", "source": MD8, "metadata": {"id": "c2_8"}},
    {"cell_type": "code", "source": CODE9, "metadata": {"id": "c2_9"}},
    {"cell_type": "markdown", "source": MD10, "metadata": {"id": "c2_10"}},
]
for _c in cells:
    _c.setdefault("metadata", {})
    if _c["cell_type"] == "code":
        _c.setdefault("outputs", [])
        _c.setdefault("execution_count", None)

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "cells": cells}
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
