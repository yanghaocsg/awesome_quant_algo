# LeapQuant Flush 算子：纯 PyTorch 移植可行性分析

> 对象：`leapquant-kernels` 仓库（GDN / KDA 窗口边界 flush 内核）
> 现状：CuTe DSL + Blackwell tcgen05，仅 NVIDIA B300（sm_103）
> 目标：不依赖 cutlass-dsl / tilelang / B300 专属指令，用**纯 PyTorch**（torch + CUDA 或 torch_npu）重写同一算子
> 日期：2026-10-31

---

## 一、结论（先给答案）

| 维度 | 判定 | 说明 |
|---|---|---|
| **功能可行性** | ✅ 高 | 算法是"矩阵重建 + 子空间迭代 + 残差量化"，全为 torch 原生可表达运算 |
| **数值可行性** | ✅ 高 | reference（definition.json）本身就是纯 PyTorch FP32；纯 torch 实现天然贴近参考，误差只会更小 |
| **工程可行性** | ✅ 高 | 不依赖 B300 / cutlass / tilelang，任何 torch 支持的设备（含昇腾 torch_npu）都能跑 |
| **生产性能可行性** | ⚠️ 低 | eager torch 逐 kernel launch + S 在全局显存反复往返，预计比 CuTe 慢 10~30×；高并发生产不适用 |
| **研究/验证场景** | ✅ 可接受 | flush 是低频算子（每 16 token 窗口一次），端到端影响有限，纯 torch 足够复现与校验 |

**一句话：纯 torch 移植"能用、算得准、哪都能跑"，但"跑不快"；它是对照参考与跨平台验证的最佳载体，不是生产内核的替代品。**

---

## 二、移植对象：算子算法拆解

仓库内 GDN/KDA 两个 flush 共享同一算法骨架，仅在"门控方式"上有别：

| 阶段 | 计算内容 | 规模（单 head） | 性质 |
|---|---|---|---|
| **① 重建状态 S** | 从量化 checkpoint 恢复 fp32 状态：残差反量化 + 旧 rank-4 补偿 + 窗口 rank-1 更新 | V×K matmul（128×128，K=R=4 / L=16） | 访存 + matmul |
| **② 子空间迭代** | 一轮子空间迭代出新的 4 个补偿 Token（qr / MGS 双重正交 + 幂迭代） | R×R 与 V×R（4×4、128×4） | 小矩阵，qr 密集但量小 |
| **③ 残差量化** | E = S − qᵀu → 列/行 scale → round → int8 codes（clamp ±127） | 逐元素 + 归约 | elementwise + reduce |

**GDN（标量门控）**：`S = p·(Q_old U_oldᵀ + dequant(codes)) + Σ_j w_j·ubuf_j·kbuf_jᵀ`
**KDA（per-key-channel 门控）**：多出 `gbuf`(fp16 gate 历史) 与 `pcum`(fp32 累计 log-gate)，状态按 key 通道分别衰减。

纯 torch 三阶段全部可直接映射，见下表。

---

## 三、算法 → 纯 torch 映射

### 3.1 重建状态 S（以 GDN 为例）

| 参考数学 | 纯 torch 表达 |
|---|---|
| 残差反量化 `dequant(codes)` | `codes.float() * (s_k/127)[:,None] * s_v[None,:] * p`（broadcast） |
| 旧 rank-4 `Q_old U_oldᵀ` | `q @ u.mT`（fp16 输入，`q.float() @ u.float().mT`） |
| 窗口更新 `Σ_j w_j ubuf_j kbuf_jᵀ` | `einsum('lj,lk,j->vk', ubuf.float(), kbuf.float(), w)` 或 `(ubuf*w[:,None]) @ kbuf` |
| 叠加 | `S = p*(A + dequant) + buffer` |

> KDA 只需在叠加前对每列乘入 `pcum`/`gbuf` 的通道衰减因子，其余同构。

### 3.2 子空间迭代（新补偿 Token）

参考用 MGS 双重正交 + 幂迭代 1 轮。torch 等价表达：

- 直接：`q_new = torch.linalg.qr(S_comp)[0]`（S 的 V×R 主列，取 Q）
- 严格对齐参考：按 reference 的 MGS 循环用 `torch` 逐列重写（幂迭代 → 投影 → 归一化）
- 数值上 torch fp32 已优于 CuTe 的 tf32 路径，精度无风险

> 注意：`linalg.qr`/`cholesky` 在 torch_npu 上的算子覆盖需单独验证（昇腾 NPU 部分 linalg 后端可能缺失，需 fallback 到 CPU 或手写 Gram–Schmidt）。

### 3.3 残差量化

```python
E = S - q_new.float() @ u_new.float().mT          # V×K
s_k = E.abs().mean(dim=0).sqrt()                  # 列 scale（对齐 reference）
s_v = E.abs().amax(dim=1)                         # 行 scale
codes = torch.clamp(torch.round(E / (s_v[:,None]*s_k[None,:]) * 127), -127, 127).to(torch.int8)
```

三阶段全部 torch 原生，**功能 100% 可达**。

---

## 四、可行性维度详细评估

### 4.1 数值可行性（最高确定性）

- `definition.json` 的 `reference.run` 本身就是**纯 PyTorch FP32**，是所有人的数学真值。
- 纯 torch 移植等于"用同一参考算法的 torch 直译"，最坏情况下与 reference 同精度（逐 op 对齐），不会引入 CuTe 的 tf32/bf16 hi-lo 近似。
- 精度判据（err ≤ 1.01×ref_err 或 ≤1e-3·|S|）纯 torch 必然通过，且无需 PRECISION_ANALYSIS 里那些"FP32 往返规避"工程——因为 torch 全程就是 FP32。

### 4.2 功能/契约可行性

输入输出契约完全不变（codes int8、s_k/s_v fp32、u/q fp16、kbuf/ubuf bf16、w/p/pcum/gbuf 等），`run(...)` 签名可原样保留；`Pool`/`flush_inplace` 的 slot 池语义可用 torch 索引改写。对上层 vLLM 等调用方**接口透明**。

### 4.3 性能可行性（瓶颈所在）

CuTe 内核 2.5~3.1× 于 TileLang 基线的来源，恰是纯 torch 缺失的能力：

| CuTe 提速来源 | 纯 torch 对应代价 |
|---|---|
| 状态 S 常驻 **TMEM**，零搬移 | S 在 fp32 全局显存与 kernel 间反复往返（bandwidth-bound） |
| 单 kernel 全融合（TMA 流水 + 多组 consumer 并行） | eager 每阶段一个独立 kernel，多次 launch 间隙 |
| int8→fp32 用字节置换走 FMA 管（零 ALU） | `codes.float()` + broadcast 逐元素，开销可观 |
| bf16 hi/lo + tf32 混合精度打满 tensor core | torch 自动调度，无手工数据摆放 |
| 持久 CTA + GPU 侧 due 扫描 | 每窗口整批重入 |

**数量级估计**（B300 对照，经验值）：

| 实现 | 单 batch-128 flush 相对耗时（估算） |
|---|---|
| CuTe `_gdn.py`/`_kda.py` | 1×（基准） |
| TileLang `baseline.py` | ~3× |
| 纯 torch eager | **~15~30×** |
| 纯 torch + `torch.compile`(CUDA graphs) | ~6~10× |

> 关键缓解：**flush 是低频算子**。论文端到端加速仅 1.47×，说明 flush 在 decode 总时长占比小。即便 flush 慢 20×，若它占 decode 10% 时间，端到端也只慢约 2×。因此研究/验证/低并发场景可接受。

### 4.4 跨平台可行性（核心价值）

- **CUDA 任意代**：纯 torch 不依赖 sm_103，可跑在 A100/H100/4090 等。
- **昇腾**：`torch_npu` 支持 matmul/einsum/elementwise/broadcast；需专项验证 `linalg.qr`/`cholesky` 覆盖（缺则手写 Gram–Schmidt fallback）。
- 这是纯 torch 移植**最大的收益**：把只能在 B300 跑的内核，变成任何 torch 环境可复现的研究资产。

---

## 五、分阶段移植路径（推荐）

| 阶段 | 产出 | 目的 | 工作量 |
|---|---|---|---|
| **P1 参考直译** | `gdn.py`/`kda.py` 纯 torch `run()` | 验证契约 + 对齐 reference 数值 | 低（1~2 天） |
| **P2 对齐校验** | 复用仓库 `workloads.jsonl` + judge 判据 | 逐 batch 比对 baseline/当前 | 低 |
| **P3 性能收敛** | `torch.compile` + CUDA graph + 合并 kernel | 把 eager 慢 20× 压到 ~6× | 中 |
| **P4 昇腾适配** | torch_npu 版 + qr fallback | 跨平台复现 | 中 |
| （可选）P5 自定义算子 | 用 Triton/CUDA 融合关键阶段 | 生产级替代 | 高，仅生产需要才做 |

**建议落地**：先做 P1+P2（纯 torch 完整可跑、数值可证），P3 视性能是否够用再投入；不推荐为"复现研究"直接上 P5。

---

## 六、结论与建议

1. **可移植性成立**：数学、数值、工程三维度纯 torch 均高度可行，接口可保持对上层透明。
2. **性能是唯一硬伤**：eager 慢一个数量级，但受"flush 低频 + 端到端占比小"缓冲，研究场景够用；高并发生产仍应回到 CuTe/Triton/自定义算子。
3. **推荐形态**：纯 torch 作为**跨平台参考实现 + 数值验证真值**沉淀在仓库（P1/P2），生产路径另行选型（P3/P5）。
4. 若目标含**昇腾**：先做 P4 的 `linalg.qr` 覆盖探测，缺后端时用 Gram–Schmidt fallback，不影响主算法正确性。

---

## 附录：待办与风险

- [ ] 确认目标设备的 torch 版本与 torch_npu 版本
- [ ] 验证 `linalg.qr/cholesky` 在目标后端的算子覆盖（P2/P4 前置）
- [ ] 用仓库 `compare_precision.py` 的判据跑通 P2 数值门
- [ ] 性能目标量化：明确"可接受端到端"的 flush 预算（占 decode %）
- 风险：纯 torch 的 int8→fp32 与多次往返在极端大 batch（≥256）时带宽压力大，需实测。
