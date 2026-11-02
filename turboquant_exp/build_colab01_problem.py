# SPDX-License-Identifier: Apache-2.0
"""Build colab_01_problem.ipynb — Part(1): why recurrent-state quantization is hard
(1) error accumulation  (2) extreme-coordinate outliers, demonstrated with the OFFICIAL
pure-torch TurboQuant (data-oblivious analytic Lloyd-Max codebook + Haar rotation + QJL).

Self-contained: the official implementation (turboquant_torch.py) is inlined into a cell,
so the notebook needs only torch + scipy + matplotlib (all preinstalled in Colab).
CPU + A100 dual device via DEV auto-detect.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "colab_01_problem.ipynb"
CORE = (HERE / "turboquant_torch.py").read_text(encoding="utf-8")

SETUP_CODE = r'''
import torch, scipy, matplotlib
print("torch", torch.__version__, "| scipy", scipy.__version__,
      "| cuda", torch.cuda.is_available(), "| mpl", matplotlib.__version__)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on", DEV)
'''

SMOKE_CODE = r'''
# Build the official quantizer and verify codebook mathematical properties.
DIM, BITS = 128, 3
centroids, boundaries = solve_lloyd_max(DIM, BITS)
print("codebook: centroids", len(centroids), "boundaries", len(boundaries))
print("centroids strictly sorted:", bool(torch.all(centroids[1:] > centroids[:-1])))
print("boundaries strictly sorted:", bool(torch.all(boundaries[1:] > boundaries[:-1])))
print("symmetric around 0 (|mean| < 1e-4):", abs(centroids.mean().item()) < 1e-4)

mse_q = TurboQuantMSE(DIM, BITS)
prod_q = TurboQuantProd(DIM, BITS)
print("TurboQuantMSE / TurboQuantProd ready | DEV =", DEV)
'''

KVLOSS_CODE = r'''
# KV-cache compression is lossless: identical dataset & thresholds to the official tests.
# Same constants as conftest: DIM=128, SEED=42, 36 layers, 8 heads, 1024 prefill + 32 decode.
import torch.nn.functional as F
NUM_LAYERS, NUM_HEADS, PREFILL, DECODE = 36, 8, 1024, 32
SEQ = PREFILL + DECODE

def run_kv(bits):
    torch.manual_seed(42)
    # Both key and value caches are compressed (the official KV setup), so savings count both.
    kc = CompressedKVCache(DIM, bits, seed=42)
    vc = CompressedKVCache(DIM, bits, seed=42)
    data = {"k": [], "v": []}
    base_rt = {"k": [], "v": []}
    for layer in range(NUM_LAYERS):
        for name, cache in (("k", kc), ("v", vc)):
            x = torch.randn(1, NUM_HEADS, SEQ, DIM)       # same Gaussian prefill+decode data
            data[name].append(x)
            idx0, n0 = cache.quantizer.quantize(x)        # accuracy baseline (no packing)
            base_rt[name].append(cache.quantizer.dequantize(idx0, n0))
            cache.add(x, layer)
    storage_cos, quant_cos = [], []
    for name, cache in (("k", kc), ("v", vc)):
        for layer in range(NUM_LAYERS):
            packed = cache.get(layer)
            assert torch.isfinite(packed).all(), f"{name} layer {layer}: NaN/Inf"
            # (A) storage format lossless: packed path vs direct round-trip baseline
            storage_cos.append(F.cosine_similarity(
                base_rt[name][layer].flatten().float(), packed.flatten().float(), dim=0).item())
            # (B) quantization accuracy vs the original tensor
            quant_cos.append(F.cosine_similarity(
                data[name][layer].flatten().float(), packed.flatten().float(), dim=0).item())
    storage = kc.storage_bytes() + vc.storage_bytes()
    baseline = kc.baseline_bytes() + vc.baseline_bytes()
    stats = {"compression_ratio": baseline / storage,
             "savings_mib": (baseline - storage) / (1024 ** 2)}
    return storage_cos, quant_cos, stats, kc.nibble_packed

QUANT_THRESHOLD = {3: 0.90, 4: 0.95}
for bits in (3, 4):
    storage_cos, quant_cos, stats, nibble = run_kv(bits)
    print(f"--- {bits}-bit KV (nibble={nibble}) ---")
    print("(A) storage format cosine min:", round(min(storage_cos), 6),
          "| all > 0.999:", all(c > 0.999 for c in storage_cos))
    print("(B) quantization cosine min:", round(min(quant_cos), 6),
          "| all >", QUANT_THRESHOLD[bits], ":",
          all(c > QUANT_THRESHOLD[bits] for c in quant_cos))
    print("compression_ratio:", round(stats["compression_ratio"], 3),
          "(>1.9:", stats["compression_ratio"] > 1.9, ")")
    print("savings_mib:", round(stats["savings_mib"], 2),
          "(>50:", stats["savings_mib"] > 50, ")")
    assert all(c > 0.999 for c in storage_cos), "storage format regression"
    assert all(c > QUANT_THRESHOLD[bits] for c in quant_cos), "quantization quality regression"
    assert stats["compression_ratio"] > 1.9 and stats["savings_mib"] > 50
print("KV cache compression is effectively lossless at the official dataset/thresholds.")
'''

PROB1_CODE = r'''
# Problem 1 — error accumulation in a recurrent state, using the OFFICIAL TurboQuantMSE.
import math
import matplotlib.pyplot as plt

# Recurrent state: S_t = beta*S_{t-1} + w_t v_t k_t^T (beta ~ 1). A quantizer error injected
# at one token is carried into every later step. Re-quantizing the full state every token
# (TurboQuant-style) therefore accumulates; quantizing less often keeps the error bounded.
H, V, K, T = 8, 64, 128, 128
quant = TurboQuantMSE(K, 8)                       # official 8-bit quantizer over row vectors

torch.manual_seed(1)
betas = torch.full((T,), 0.99)
w = torch.rand(T) * 0.5
v = torch.randn(T, V) * 0.2
k = torch.randn(T, K) * 0.2

# fp64 reference state
S_ref = torch.zeros(H, V, K, dtype=torch.float64)
for t in range(T):
    S_ref = betas[t].double() * S_ref + w[t].double() * (
        v[t].double().unsqueeze(-1) @ k[t].double().unsqueeze(-2)
    )

def run(period):
    S = torch.zeros(H, V, K)
    errs = []
    for t in range(T):
        S = betas[t] * S + w[t] * (v[t].unsqueeze(-1) @ k[t].unsqueeze(-2))
        if (t + 1) % period == 0:
            idx, norms = quant.quantize(S)       # official normalize -> rotate -> bucketize
            S = quant.dequantize(idx, norms)
        errs.append((S.double() - S_ref).abs().mean().item())
    return errs

e1 = run(1)                                       # re-quantize every token
e16 = run(16)                                     # re-quantize every 16 tokens
plt.figure(figsize=(8, 4.5))
plt.semilogy(e1, marker='.', label="per-step TurboQuant (re-quantize every token)")
plt.semilogy(e16, marker='o', ms=3, label="quantize every 16 tokens")
plt.xlabel("token t"); plt.ylabel("state error vs fp64 (log)")
plt.title("Error accumulation: per-step re-quantization stacks error, less frequent stays low")
plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
print(f"final error: per-step={e1[-1]:.3e} vs every16={e16[-1]:.3e}  "
      f"({e1[-1] / max(e16[-1], 1e-30):.1f}x worse)")
'''

PROB2_CODE = r'''
# Problem 2 — extreme-coordinate outliers: the naive failure, and how TurboQuant mitigates it.
import torch.nn.functional as F

# (a) Naive per-tensor absmax scalar quantization (NO rotation): a single outlier inflates
#     the global scale s = max|x|, coarsening the step for every other coordinate.
def absmax_q(x, nbits=8):
    qmax = 2 ** (nbits - 1) - 1
    s = x.abs().max() / qmax
    return (x / s).round().clamp(-qmax, qmax) * s

torch.manual_seed(3)
S = torch.randn(4096, 128)
base_s = S.abs().max().item()
mult = [1, 2, 4, 8, 16, 32]
absmax_err = []
for m in mult:
    So = S.clone(); So[100, 7] = base_s * m
    absmax_err.append((So - absmax_q(So)).abs().mean().item())
clean_err = absmax_err[0]

# (b) Official TurboQuantMSE on the same outlier vectors: norm stored separately, Haar rotation
#     spreads the outlier energy so rotated coordinates stay ~ Gaussian for ANY direction.
tq2 = TurboQuantMSE(128, 3)
tq_cos = []
for m in mult:
    So = S.clone(); So[100, 7] = base_s * m
    idx, norms = tq2.quantize(So)
    sr = tq2.dequantize(idx, norms)
    tq_cos.append(F.cosine_similarity(So, sr, dim=-1).mean().item())

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(mult, absmax_err, marker='o', color='crimson', label="naive absmax MAE")
ax1.set_xlabel("outlier magnitude (x typical max)"); ax1.set_ylabel("absmax MAE", color='crimson')
ax2 = ax1.twinx()
ax2.plot(mult, tq_cos, marker='s', color='seagreen', label="TurboQuant reconstruction cosine")
ax2.set_ylabel("TurboQuant cosine", color='seagreen'); ax2.set_ylim(0.9, 1.0)
ax1.grid(alpha=0.3); fig.suptitle("Outlier: naive absmax degrades, Haar rotation + stored norm stay stable")
fig.tight_layout(); plt.show()
print("naive absmax MAE :", [round(e, 5) for e in absmax_err])
print("TurboQuant cosine :", [round(c, 4) for c in tq_cos])
print(f"absmax at x{mult[-1]} is {absmax_err[-1]/clean_err:.1f}x its clean error; "
      f"TurboQuant cosine only moves {tq_cos[0]-tq_cos[-1]:.4f}")
'''

REGRESSION_CODE = r'''
# TurboQuant regression suite — thresholds aligned with the official turboquant-vllm-npu tests.
import math
results = []
def check(name, ok):
    results.append((name, bool(ok)))
    print(("PASS" if ok else "FAIL"), name)

# (1) codebook properties across bit widths
for bits in (2, 3, 4):
    c, bnd = solve_lloyd_max(DIM, bits)
    check(f"b{bits}: 2^bits centroids", len(c) == 2 ** bits)
    check(f"b{bits}: 2^bits-1 boundaries", len(bnd) == 2 ** bits - 1)
    check(f"b{bits}: strictly sorted", bool(torch.all(c[1:] > c[:-1])))
    check(f"b{bits}: symmetric", abs(c.mean().item()) < 1e-4)

# (2) MSE below the theoretical bound and improves with more bits; index range; zero vector
mse_by_bits = []
for bits in (2, 3, 4):
    qm = TurboQuantMSE(DIM, bits)
    x = torch.randn(500, DIM)
    x = x / x.norm(dim=-1, keepdim=True)
    idx, norms = qm.quantize(x)
    check(f"b{bits}: indices in [0,{2**bits-1}]",
          bool(idx.min() >= 0 and idx.max() < 2 ** bits))
    xr = qm.dequantize(idx, norms)
    mse = (x - xr).pow(2).mean().item()
    mse_by_bits.append(mse)
    bound = 5 * math.sqrt(3) * math.pi / 2 * (1 / 4 ** bits)
    check(f"b{bits}: MSE {mse:.2e} < bound {bound:.2e}", mse < bound)
check("MSE decreases as bits increase", mse_by_bits[0] > mse_by_bits[1] > mse_by_bits[2])

zero = torch.zeros(4, DIM)
zi, zn = mse_q.quantize(zero)
zr = mse_q.dequantize(zi, zn)
check("zero vector stays zero", bool(torch.all(zr.abs() < 1e-6)))

# (3) norm preservation
xn = torch.randn(200, DIM)
orig_norm = xn.norm(dim=-1)
_, stored_norm = mse_q.quantize(xn)
rel = ((orig_norm - stored_norm.squeeze()) / (orig_norm + 1e-10)).abs().mean().item()
check(f"norm relative error {rel:.2e} < 1%", rel < 0.01)

# (4) QJL unbiased inner product
qry = torch.randn(300, DIM) * 0.1
key = torch.randn(300, DIM) * 0.1
true_ip = (qry * key).sum(-1)
ii, nm, sg, rn = prod_q.quantize(key)
est_ip = prod_q.estimate_inner_product(qry, ii, nm, sg, rn).squeeze(-1)
rel_bias = (est_ip - true_ip).mean().abs().item() / (true_ip.abs().mean() + 1e-10)
corr = torch.corrcoef(torch.stack([true_ip, est_ip]))[0, 1].item()
check(f"QJL relative bias {rel_bias:.3f} < 15%", rel_bias < 0.15)
check(f"QJL correlation {corr:.3f} > 0.9", corr > 0.9)

# (5) no NaN / Inf anywhere
check("no NaN/Inf in reconstruction", bool(torch.isfinite(zr).all() and torch.isfinite(est_ip).all()))

n_fail = sum(1 for _, ok in results if not ok)
print("\n=== regression:", "ALL PASS (%d checks)" % len(results) if n_fail == 0
      else f"{n_fail} FAIL / {len(results)}", "===")
'''

cells = [
    {"cell_type": "markdown", "source": [
        "# Part(1/4) 循环网络状态量化的两个问题（官方 TurboQuant 纯 torch 视角）\n",
        "\n",
        "**项目**：TurboQuant vs LeapQuant —— Qwen3.5（0.8B/9B/35B-A3B）Gated-DeltaNet **循环状态**的 8-bit 量化。\n",
        "\n",
        "本 notebook 用**官方纯 torch 版 TurboQuant**（`0xsero/turboquant`，arXiv:2504.19874）说明循环状态低比特量化的两个难点：\n",
        "\n",
        "1. **错误累积**：状态沿序列递归 `S_t = β_t S_{t-1} + w_t v_t k_tᵀ`，每步量化误差被递归带到后续所有步 → 总误差随序列长度增长。\n",
        "2. **极端错误点**：单坐标 outlier 改变归一化后的方向，使 Haar 旋转后的坐标分布偏离码本假设的 `N(0,1/d)` → 重建质量下降。\n",
        "\n",
        "**官方实现要点（与数据驱动 k-means 版的本质区别）**：\n",
        "\n",
        "- **数据无关（data-oblivious）固定码本**：对旋转后单位向量坐标的已知分布（Beta，d≥64 用 Gaussian 近似），用 `scipy.integrate.quad` 解析求 Lloyd-Max 条件期望，不依赖任何校准数据。\n",
        "- **归一化 + 独立存 norm**：量化方向、范数单独存储，整体幅度不影响质量。\n",
        "- **Haar 随机正交旋转**（QR + 符号校正）+ `bucketize` 标量量化；两阶段 QJL 提供无偏内积。\n",
        "\n",
        "> 依赖仅 torch + scipy + matplotlib（Colab 自带）；DEV 自适应 CPU / A100。"
    ]},
    {"cell_type": "code", "source": [SETUP_CODE]},
    {"cell_type": "markdown", "source": [
        "## 0. 官方 TurboQuant 纯 torch 实现（自包含内联）\n",
        "\n",
        "下面是官方算法的完整纯 torch 移植：`solve_lloyd_max` / `LloydMaxCodebook` / `TurboQuantMSE` / `TurboQuantProd`。码本与随机矩阵在 CPU float32 生成，运行时迁移到 DEV。"
    ]},
    {"cell_type": "code", "source": [CORE]},
    {"cell_type": "code", "source": [SMOKE_CODE]},
    {"cell_type": "markdown", "source": [
        "## 1. 先证明：KV cache 压缩精度无损（与官方相同测试集与阈值）\n",
        "\n",
        "采用与官方 `turboquant-vllm-npu` 完全一致的常量与数据：`DIM=128, SEED=42`，**36 层 × 8 heads × 1024 prefill + 32 decode** 的 Gaussian KV。逐层独立量化（无跨层递归），断言：\n",
        "\n",
        "- **(A) 存储格式无损**：pack/unpack（3-bit uint8、4-bit nibble）路径与直接 round-trip 基准之间，每层 cosine **> 0.999**，且无 NaN/Inf；\n",
        "- **(B) 量化本身精度**：原始 vs 还原，每层 cosine 3-bit **>0.90**、4-bit **>0.95**；\n",
        "- 压缩比 **> 1.9**、显存节省 **> 50 MiB**。\n",
        "\n",
        "结论：在 attention KV 这种**层间独立、无递归**的场景，TurboQuant 压缩几乎无损。"
    ]},
    {"cell_type": "code", "source": [KVLOSS_CODE]},
    {"cell_type": "markdown", "source": [
        "## 2. 问题 1 — 错误累积\n",
        "\n",
        "循环状态 `S_t = β_t S_{t-1} + w_t v_t k_tᵀ`：量化误差一旦进入状态，就被递归乘到后续每一步（β≈1 时几乎不衰减）。用官方 `TurboQuantMSE` 对 128-token 合成状态流做**每 token 重量化**，对比 fp64 参考；再与每 16 步量化对比。"
    ]},
    {"cell_type": "code", "source": [PROB1_CODE]},
    {"cell_type": "markdown", "source": [
        "## 3. 问题 2 — 极端错误点\n",
        "\n",
        "先看**朴素 per-tensor absmax 标量量化**（不旋转）：一个 outlier 就把全局 scale `s=max|x|` 拉大，所有其他坐标的量化步长随之变粗，MAE 随 outlier 倍数上升。\n",
        "\n",
        "再看**官方 TurboQuantMSE** 如何缓解：① 范数独立存储（只量化单位球方向），整体幅度不影响；② Haar 随机旋转把 outlier 能量分散到所有坐标——无论输入方向是否稀疏，旋转后坐标都近似 `N(0,1/d)`，数据无关码本因此保持稳健（重建 cosine 几乎不动）。这正是 TurboQuant 旋转设计的核心价值。"
    ]},
    {"cell_type": "code", "source": [PROB2_CODE]},
    {"cell_type": "markdown", "source": [
        "## 4. TurboQuant 回归测试\n",
        "\n",
        "对齐 `turboquant-vllm-npu` 官方测试口径：码本严格递增/对称/数量、MSE 理论界与随 bits 下降、索引范围、零向量、norm 保持、QJL 无偏内积（偏差<15%、相关>0.9）、无 NaN/Inf。"
    ]},
    {"cell_type": "code", "source": [REGRESSION_CODE]},
    {"cell_type": "markdown", "source": [
        "## 5. 小结\n",
        "\n",
        "- **KV 场景无损**：层间独立、无递归时，官方同口径测试（36 层 1024+32）每层 cosine > 0.999、压缩比 >1.9、节省 >50 MiB——TurboQuant 对 attention KV 几乎无损。\n",
        "- **循环状态问题 1（错误累积）**：状态沿序列递归，逐 token 重量化误差被带到后续每一步，随上下文累积（每步量化显著高于每 16 步量化）。\n",
        "- **循环状态问题 2（异常值漂移）**：朴素 absmax 下 outlier 拉大全局 scale、恶化全局；TurboQuant 的 Haar 旋转 + 独立存 norm 能缓解方向型 outlier，但消除不了问题 1 的逐 token 累积。\n",
        "- 因此 TurboQuant 的短板不在压缩算子本身，而在**循环状态的递归结构**（LeapQuant 论文 Table 1：8-bit TurboQuant 平均 49.6 vs LeapQuant 75.5）。\n",
        "\n",
        "**下一步（Part 2）**：LeapQuant 用逐窗口量化 + 补偿 token + 残差平滑同时缓解这两个问题。"
    ]},
]

for _i, _c in enumerate(cells):
    _c.setdefault("metadata", {})
    _c["metadata"].setdefault("id", f"p1_{_i}")
    if _c["cell_type"] == "code":
        _c.setdefault("outputs", [])
        _c.setdefault("execution_count", None)

nb = {
    "nbformat": 4, "nbformat_minor": 2,
    "metadata": {"colab": {"name": "colab_01_problem.ipynb", "provenance": [], "toc_visible": True},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"},
                 "accelerator": "GPU"},
    "cells": cells,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")
