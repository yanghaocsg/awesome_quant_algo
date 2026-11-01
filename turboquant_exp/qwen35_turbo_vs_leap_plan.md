# Qwen3.5-9B：TurboQuant vs LeapQuant 实验规划

> 目标：在 Qwen3.5-9B 的循环状态（Gated DeltaNet / GDN 层）上，对比 **8-bit 状态量化** 的两种调度——
> **LeapQuant**（逐窗口 + Compensator Tokens + 残差平滑）vs **TurboQuant**（每 decode 步重量化），
> 复现 LeapQuant 论文 Table 1 中 Qwen3.5-9B 列的实验（AIME / GPQA / LCB / MMLU）。

---

## 0. 硬约束与路线决策（先读）

**模型事实**
- Qwen3.5-9B 公开权重：`huggingface.co/Qwen/Qwen3.5-9B`（Apache-2.0，2026-08 发布）。
- 架构：**24 层 Gated-DeltaNet（GDN，线性注意力、含循环状态）+ 8 层全注意力（GQA 4:1）**，Dense 9B 全激活。
- 这与 LeapQuant 论文所用的 Qwen3.5-9B 一致；GDN 状态更新语义 = 我们已移植的 `gdn_torch.py`。

**显存硬约束**
- BF16/FP16 权重 ≈ **18 GB**，**T4（16 GB）装不下**。
- 这是必须正视的事实：用 Qwen3.5-9B 本身，T4 无法加载完整权重。

**三条可行路线（需用户确认后执行）**

| 路线 | 硬件 | 说明 | 贴合度 |
|---|---|---|---|
| **A. 完整 9B** | Colab 选 **A100-80G 或 L4-24G** | bf16 加载真模型，最忠实复现 | ★★★★★ |
| **B. T4 + INT8 权重** | T4（16GB） | bitsandbytes 8bit 权重（~9GB），能在 T4 跑 | ★★★（权重本身也量化，与"仅状态 8bit"口径有混叠；GDN 层算子可能需 CPU/慢路径） |
| **C. T4 + 小模型代理** | T4（16GB） | 用 Qwen3.5-0.8B/2B/4B（同样 GDN 架构），方法一致、分数不同 | ★★★（非 9B，分数作示意） |

**建议**：若追求最贴近论文 → 路线 A（A100/L4）；若坚持 T4 → 路线 C（小模型）最现实，路线 B 可行但口径混叠、易踩算子坑。

---

## 1. 实验目标与指标

在 Qwen3.5-9B 的 **GDN 循环状态** 上对比：

- **LeapQuant（8-bit）**：窗口长度 p=16，窗口内 fp32 高精度缓冲，窗口边界做一次重建 + Compensator Tokens（R=4）+ 残差平滑 + INT8 codes（即 `gdn_torch.run`）。
- **TurboQuant（8-bit）**：每 decode 步对状态做重量化（随机旋转 + 每坐标 Lloyd-Max，或 INT8 absmax）。
- **基线**：FP32 状态 / BF16 状态。

**指标**
1. 下游任务准确率（%）：MMLU（子集）、数学推理（GSM8K / AIME 简版）、LCB（可选）。
2. 循环状态重建误差（对 fp64 参考）随上下文长度累积曲线。
3. 显存占用 / 解码吞吐（T4 或 A100 上的实测）。

---

## 2. 前置依赖与可行性探查（执行前必须完成）

- [ ] **确认 transformers 对 Qwen3.5 的 GDN 状态接口**：能否增量（recurrent）推理、state 张量形状、能否手动替换 state。
  - 若 transformers 无原生支持 → 改用 **flash-linear-attention (fla)** 库的 Qwen3.5/GatedDeltaNet 实现，或自写 GDN 层 forward。
- [ ] 评测集准备：MMLU（56 主题，取子集 ~200–500 题）、GSM8K、AIME 2024 简版、LCB（需代码执行沙箱，T4 上最贵，可跳过）。
- [ ] 硬件确认：路线 A / B / C。
- [ ] 依赖安装：torch、transformers（+ accelerate、bitsandbytes 视路线）、fla（视需要）。

---

## 3. 实验步骤（分阶段，每阶段可独立验证）

### Phase 0：环境与模型
1. Colab 选 GPU runtime（A100/L4 或 T4）。
2. 安装依赖；下载 `Qwen/Qwen3.5-9B` 权重。
3. 冒烟加载 + 一次前向，确认模型可用、GDN 层结构可见。

### Phase 1：定位循环状态
1. 抽取模型的 GDN 层循环状态（recurrent_state，每个 GDN 层一组）。
2. 校验：状态更新语义与 `gdn_torch` 的 `S += w·v·kᵀ`（delta rule）一致（用合成输入对拍）。
3. 确认 attention 层（8 层）无循环状态、不受本实验影响（其 KV 缓存保持原样）。

### Phase 2：状态量化注入（核心）
1. 写统一接口 `quantize_state(state, mode, window_boundary)`：
   - `mode=fp32 / bf16`：基线，直接存储。
   - `mode=leapquant`：窗口边界调用 `gdn_torch.run`（重建 + 补偿 + 平滑），窗口内不做量化（fp32 缓冲）。
   - `mode=turboquant`：每 token 对状态做 8bit 重量化（旋转 + Lloyd-Max 或 absmax）。
2. 手动 rnn 推理 loop：逐 token 前向，维护 state，在指定点量化后**替换模型的 state 缓存**，继续下一 token。
3. **单元验证**：合成状态流上，LeapQuant 终误差 ≪ TurboQuant（应见 5–10× 差距，呼应论文 Figure 3）。

### Phase 3：下游评测（few-shot）
1. 对每种 mode 跑 MMLU 子集 + 数学集 few-shot（0–5 shot），统计 top-1 准确率。
2. 控制规模：题数 100–500、max_new_tokens 适中，保证在所选 GPU 上 1 小时内完成。

### Phase 4：结果汇总
1. 表：`FP32 / BF16 / LeapQuant-8bit / TurboQuant-8bit` 在 MMLU / GSM8K（/ AIME / LCB 若跑）的准确率。
2. 图：状态误差随上下文 token 数累积（LeapQuant 平坦、TurboQuant 上升）。
3. 对照论文 Table 1 Qwen3.5-9B 分数（LeapQuant 8bit≈FP32≈87.9@AIME 等），标注为**小样本/子集复现**，数字仅供方法趋势参考。

---

## 4. 风险与降级

| 风险 | 应对 |
|---|---|
| transformers 无 Qwen3.5 GDN 增量状态接口 | 用 fla 库或自写 GDN 层 forward |
| 9B 评测在 GPU 上慢 | 限题数、限序列、必要时路线 C 小模型 |
| T4 显存不足（路线 A 不可行时） | 路线 B（int8 权重）或 C（小模型） |
| GDN 算子（gated delta）无 CUDA 内核 | 用 fla 已优化内核；纯 torch 慢路径兜底 |
| LCB 需代码沙箱、极慢 | 跳过，仅用 MMLU + 数学 |

---

## 5. 交付物

- 可执行 Colab notebook（分 cell：环境 / 加载 / 状态注入 / 评测 / 汇总）。
- 输出：对比表 + 误差累积图 + 结论（附数据来源与口径标注）。
- 脚本与结果归档到 `torch_port/`（文件名英文）。

---

## 附：口径声明（重要）

论文 Table 1 是完整模型在**全量评测集**上的分数。本实验若用路线 C（小模型）或子集，**分数不可与论文直接比较**；目标是复现**方法趋势**（LeapQuant 逐窗口 ≫ TurboQuant 每步），而非复刻数字。路线 A（A100/L4 完整 9B）才可做最接近的对照（仍受评测子集规模影响）。
