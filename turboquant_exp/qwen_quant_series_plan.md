# TurboQuant vs LeapQuant：循环状态量化研究系列（Qwen3.5）— 项目规划

> 目标：以「问题 → 方法 → 实测 → 对比」四部分，在 Qwen3.5 系列（0.8B / 9B / 35B-A3B）的
> **Gated-DeltaNet（GDN）循环状态**上，系统对比 TurboQuant（每步量化）vs LeapQuant（逐窗口量化）两种 8-bit 状态量化。
> 交付形态：**5 个 Colab**，每个含 **CPU + A100 GPU 双版本**。TurboQuant 以 `github.com/0xsero/turboquant` 为基础复用算子。

---

## 0. 项目骨架（总纲）

| 部分 | 交付 Colab | 主题 | 运行时版本 |
|---|---|---|---|
| (1) 问题 | `colab_01_problem.ipynb` | 循环网络状态量化为何难（错误累积 + 极端错误点），用 TurboQuant 说明 | CPU + A100 |
| (2) 方法 | `colab_02_leapquant.ipynb` | LeapQuant 如何解决（≥3 公式 + fix），对比 TurboQuant | CPU + A100 |
| (3) 实测 | `colab_03_test_08b.ipynb` + `colab_04_test_9b_35b.ipynb` | TurboQuant / LeapQuant 在 Qwen3.5-0.8B、9B、35B-A3B 实测 | CPU + A100 |
| (4) 对比 | `colab_05_compare.ipynb` | 三档模型汇总对比 + 图表 + 结论 | CPU + A100 |

**通用约束**
- 每个 notebook 用 `DEV = cuda if torch.cuda.is_available() else cpu` 自适应；用小规模开关控制题数/序列，CPU 冒烟、A100 完整。
- 代码自包含、仅依赖 torch + transformers（Colab 自带），评测集在 Colab 内下载。
- 文件名/目录纯英文；全程中文说明文字。

---

## 1. 关键事实与硬约束（先读）

**模型**
- Qwen3.5-0.8B：GDN 混合，bf16 ≈1.6GB，任何设备可跑。
- Qwen3.5-9B：24 GDN 层 + 8 full-attn，bf16 ≈18GB，**A100-40GB 可装，T4 不可**。
- Qwen3.5-35B-A3B：MoE，35B total / 3B active，40 层（Gated-Attention + Gated-DeltaNet），HF 公开 `Qwen/Qwen3.5-35B-A3B`（Apache-2.0）。**bf16 ≈70GB > A100-40GB，装不下**。
  - **用户指定方案：W8A8（权重 8bit + 激活 8bit）**。权重 8bit ≈35GB，A100-40GB 可装（激活亦 8bit 省显存）。
  - 实现路径（用户指定）：**官方 FP8 预量化 `Qwen/Qwen3.5-35B-A3B-FP8`**（fine-grained fp8 block-128，权重 8bit + 激活 8bit，即 W8A8-FP8），transformers 直接加载（`torch_dtype=torch.float8_e4m3fn`），HF 已发布，兼容 transformers/vLLM/SGLang。AWQ 官方为 4bit 权重（非 W8A8），不作为本方案。
  - 注意：本实验的 **TurboQuant/LeapQuant 仍量化 GDN 循环状态**（状态以 bf16/fp32 维护）；FP8 仅用于缩小权重+激活显存，使 35B 装入 A100-40GB。
  - 风险：FP8 前向时激活反量化到 bf16（避免 fp8 累积误差），显存≈权重 35GB + 激活 buffer；40GB 可能紧张 → 必要时降级 0.8B 机制示意。
  - 备选：`Qwen/Qwen3.5-35B-A3B-AWQ`（4bit，非 W8A8）仅在显存极紧张时作降级。

**实现基础**
- TurboQuant：`github.com/0xsero/turboquant`（ICLR 2026, arXiv:2504.19874）。提供 `rotation.py`（随机正交旋转 + QJL）、`codebook.py`（Lloyd-Max for Beta）、`quantizer.py`（TurboQuantMSE/Prod）。**注意**：该仓库量化对象是 attention 层 KV cache，**不直接处理 GDN 循环状态**；我们**复用其量化算子（旋转 + Lloyd-Max）**，把对象改为 GDN 循环状态（等价于我们已有的 `TURBO_CODE` 每步量化）。
- LeapQuant：复用 `torch_port/gdn_torch.py`（GDN 窗口边界量化忠实移植：重建 + 补偿 + 平滑）。论文状态语义 S = β·S + w·v·kᵀ。
- transformers 5.22.1 **原生支持 Qwen3.5**（model_type=qwen3_5，含 `Qwen3_5GatedDeltaNet` 层）；`past_key_values` 为 `DynamicCache`，GDN 状态经 cache 暴露（可替换性在 `qwen35_state_probe.ipynb` 确认中）。

**评测口径**
- 评测：MMLU 子集 + 数学（GSM8K 简版），**跳过 LCB**（需代码沙箱、极慢）。
- 数字不作论文直接对比，只复现方法趋势（LeapQuant 逐窗口 ≫ TurboQuant 每步）。

---

## 2. 各部分详细规划

### Part (1) 循环状态量化的问题 — `colab_01_problem.ipynb`

**内容**：为什么循环网络的 8-bit 状态量化比 attention KV 量化更难，两个原因：
1. **错误累积（Error Accumulation）**：状态 S 沿序列递归 S_t = β_t S_{t-1} + w_t v_t k_tᵀ，每步量化误差 δ_t 累进到后续所有步 → 总误差随序列长度增长。
2. **极端错误点（Extreme / Outlier Coordinates）**：状态某些坐标出现极值，absmax 量化 s=max|x| 被极值拉大 → 其余坐标被压成粗粒度，或极值溢出。

**用 TurboQuant 说明**：用其每步量化（随机正交旋转 → 每坐标 Lloyd-Max）作用于合成/真实 GDN 状态流，演示：
- 状态重建误差随 token 累积曲线（正确复现「TurboQuant 误差持续上升」）。
- 极值坐标对 absmax 量化的影响（放大 vs 溢出）。

**公式（至少）**：
- 状态递推：`S_t = β_t S_{t-1} + w_t v_t k_tᵀ`
- absmax 标量量化：`x̂ = round(x / s) · s,  s = max|x|`
- Lloyd-Max 量化：在数据分布上最小化 `E[(x−q(x))²]`

**Colab 结构**：环境 → 合成状态流生成 → TurboQuant 每步量化 → 误差累积图 → 极端点演示 → 小结。CPU 小规模 + A100 完整。

### Part (2) LeapQuant 的解法 — `colab_02_leapquant.ipynb`

**内容**：LeapQuant 逐窗口量化如何同时缓解两个问题。

**≥3 个公式（全部给出推导/说明）**：
1. **窗口内高精度缓冲**：窗口长度 p 内保持 fp32，状态只读不量化：`S_{t} = S_{t-1}·β_t + w_t v_t k_tᵀ`（t 在窗口内，无量化）。
2. **窗口边界一次性量化（Leap）**：到窗口边界 p 才量化一次：`S̃_p = Q_leap(S_p)`，量化频率从「每步」降到「每 p 步」，错误累积大幅减缓。
3. **Compensator Tokens（补偿 token）**：窗口边界插入 R 个补偿 token，重建更贴近真实状态：`S̃_p ← S̃_p + Σ_{r=1..R} w_r v_r k_rᵀ`（补偿重建误差）。
4. **子空间迭代重建**：在低维子空间（如主成分）迭代找最优近似，抑制极端点：`Ŝ = argmin_Ŝ∈𝒮 ||S − Ŝ||`（投影到子空间）。
5. **残差平滑（可选）**：对量化残差 `R = S_p − S̃_p` 做平滑/低通，进一步压极端点。

**对照**：同窗口下 TurboQuant（每步）vs LeapQuant（逐窗口）的误差曲线与极值抑制效果。

**Colab 结构**：环境 → LeapQuant 算法实现（含公式对照）→ 与 TurboQuant 同测 → 误差对比图。CPU/A100 双版本。

### Part (3) 实际项目测试 — `colab_03_test_08b.ipynb` + `colab_04_test_9b_35b.ipynb`

**colab_03（0.8B，快，兼 CPU 完整演示）**：
- 加载 Qwen3.5-0.8B → 定位 GDN 循环状态 → 注入 TQ(每步) / LQ(逐窗口) 8bit → MMLU 子集 + GSM8K 简版 few-shot。
- 输出：两种方法准确率 + 状态误差 + 显存/耗时。

**colab_04（9B + 35B-A3B，A100）**：
- 9B：bf16 加载（A100-40GB OK）。
- 35B-A3B：**官方 FP8 加载**（`Qwen/Qwen3.5-35B-A3B-FP8`，W8A8-FP8，`torch_dtype=torch.float8_e4m3fn`），权重≈35GB 装入 A100-40GB。
- 同样注入 + 评测。

**核心工程**（两 colab 共用逻辑，前置于 Part(3) 的 `qwen35_state_probe.ipynb` 已确认）：
1. 加载模型，`use_cache=True` 前向，从 `DynamicCache` 定位 GDN 状态张量。
2. 逐 token rnn 推理 loop，在指定点（每步 / 每窗口）量化并替换 cache 中的状态张量。
3. 每模式评测，汇总准确率。

**Colab 结构**：环境 → 加载 → 状态注入（TQ/LQ 封装）→ 评测 → 结果。CPU 冒烟（0.8B / 小样本）+ A100 完整。

### Part (4) 对比分析 — `colab_05_compare.ipynb`

**内容**：
- 汇总三档模型（0.8B / 9B / 35B-A3B）在 TQ vs LQ 的准确率、状态重建误差、显存/吞吐。
- 图表：误差累积曲线（TQ 上升 vs LQ 平坦）、三模型准确率柱状对比、显存对比。
- 结论：LQ 逐窗口为何优于 TQ 每步（结合 Part1 的两个原因）。
- 口径声明：子集/小样本，数字仅作方法趋势。

**Colab 结构**：读入 Part(3) 结果（或复跑小规模）→ 汇总表 + 图 → 结论。CPU/A100 双版本。

---

## 3. 执行顺序与依赖

```
qwen35_state_probe.ipynb   (已就绪，确认 DynamicCache 状态可替换性 —— 前置门)
   ↓
colab_01_problem → colab_02_leapquant   (方法演示，可本地 CPU 冒烟)
   ↓
colab_03_test_08b → colab_04_test_9b_35b   (A100 实跑)
   ↓
colab_05_compare   (汇总)
```

- 依赖：`0xsero/turboquant` 算子（rotation + Lloyd-Max，CPU 可跑）、`torch_port/gdn_torch.py`、transformers 5.22.1、MMLU/GSM8K 评测集。
- 每 colab 先本机 CPU 冒烟（小规模、验证 device 敏感点，如 `torch.linspace(device=...)`），再 A100 实跑。
- Colab 准备流程（已打通）：登录 → 新建/上传 notebook → 命令面板「更改运行时类型」设 A100 → 连接。

---

## 4. 风险与降级

| 风险 | 应对 |
|---|---|
| DynamicCache 中 GDN 状态不可直接替换 | `state_probe` 先行确认；否则改用模型内 hook 或自写 GDN forward |
| 35B-A3B bf16 70GB > A100-40GB | **官方 FP8**（`Qwen/Qwen3.5-35B-A3B-FP8`，W8A8-FP8）缩小至 ~35GB；显存紧张则降级 |
| 官方 FP8 对 Qwen3.5（GDN/MoE 自定义层）支持 | 官方已在 HF 发布、transformers 5.22.1 支持 fp8；加载后实测 GDN 层完整性 |
| transformers 对 Qwen3.5 GDN 用 reference 内核（慢） | A100 上接受慢速；必要时换 fla 内核 |
| turboquant 算子面向 attention KV | 复用其旋转 + Lloyd-Max 算子，对象改为 GDN 状态（与 TURBO_CODE 等价） |
| 评测慢（9B/35B 逐 token 注入） | 限题数、限序列；CPU 用小样本 |

---

## 5. 交付物清单

| 文件 | 说明 |
|---|---|
| `qwen35_state_probe.ipynb` | 状态接口确认（已上传 Colab，待跑） |
| `colab_01_problem.ipynb` | Part(1) 问题演示 |
| `colab_02_leapquant.ipynb` | Part(2) 方法 + 公式 |
| `colab_03_test_08b.ipynb` | Part(3) 0.8B 实测 |
| `colab_04_test_9b_35b.ipynb` | Part(3) 9B + 35B-A3B 实测 |
| `colab_05_compare.ipynb` | Part(4) 对比分析 |
| `qwen35_turbo_vs_leap_plan.md` / 本文件 | 规划文档 |
| `torch_port/gdn_torch.py` | LeapQuant 侧移植（复用） |

所有 notebook 均含 **CPU + A100 GPU 双版本**（device 自适应 + 规模开关）。
