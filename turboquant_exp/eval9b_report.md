# Qwen3.5-9B 小样本评测报告（bf16 基线）

> 本报告记录 AutoDL GPU 实例上 Qwen3.5-9B 真实模型的小样本评测结果、评测方法的关键修复与分析。
> 数据来源：独立脚本 `eval9b.py` 的实际运行，结果文件 `eval9b_results.json`，日志 `eval9b2.log`。

---

## 1. 运行环境

| 项目 | 配置 |
|---|---|
| 实例 | AutoDL F73，单卡 NVIDIA RTX 6000D（84GB） |
| 镜像 | PyTorch 2.8.0 / Python 3.12.3 / CUDA 12.8 |
| 模型 | Qwen3.5-9B（24 GDN 层 + 8 full-attn），bf16 本地离线加载 |
| 模型路径 | `/root/autodl-tmp/qwen9b`（数据盘，15 文件，4 分片） |
| 参数量 | 8.95B；本地加载耗时约 8.8 秒 |
| 关键版本 | transformers 5.22.1、accelerate 1.15.0、datasets 5.1.0 |

注：`causal_conv1d` / `flash_attention` 未安装，自动回退到参考实现，仅影响速度，不影响正确性。

---

## 2. 评测口径

- **MMLU**：3 个子集（abstract_algebra、college_computer_science、global_facts），每子集 5 题，共 15 题，0-shot，单 token 选择 A-D。
- **GSM8K**：8 题，贪心生成（`do_sample=False`），取生成文本末尾数字与 gold `#### N` 比较。
- **性质声明**：本评测为**小样本趋势复现，不是完整基准**，样本量小、方差大，数值不与论文完整基准直接比较。

---

## 3. 评测结果

| 评测项 | 准确率 | 正确/总数 |
|---|---|---|
| MMLU abstract_algebra | 0.60 | 3/5 |
| MMLU college_computer_science | 0.60 | 3/5 |
| MMLU global_facts | 0.40 | 2/5 |
| **MMLU 综合** | **0.533** | **8/15** |
| **GSM8K** | **0.625** | **5/8** |

原始结果 JSON（`eval9b_results.json`）：

```json
{
  "MMLU:abstract_algebra": 0.6,
  "MMLU:college_computer_science": 0.6,
  "MMLU:global_facts": 0.4,
  "MMLU_overall": 0.533,
  "GSM8K": 0.625
}
```

---

## 4. 评测方法的关键修复

首轮运行 MMLU 全部为 0 分，经 `debug_mmlu.py` 定位，根因是**评测方法错误，而非模型答错**。

### 4.1 MMLU：全词表 argmax 被换行符抢占

- 原逻辑在整个词表上对下一位置取 argmax。模型在 "Answer:" 后倾向于先输出换行符（token id 198，decode 为 `\n`），其 logit 高于所有选项字母，导致拿到的预测恒为换行、判定全错。
- 调试证据：以 abstract_algebra 第一题为例，选项字母 token 的 logit 分别为 A 0.418、**B 4.489**、C 3.594、D 3.504，B 最高而 gold 正是 B——模型其实答对。
- 修复：**只在选项字母 token（A/B/C/D，plain id 32/33/34/35）内取 argmax**，排除换行符等非选项 token。修复后 MMLU 综合恢复为 0.533。

### 4.2 GSM8K：答案提取与生成长度

- gold 提取改为正则 `####\s*(-?[\d,]+)`，避免把推理过程中的数字拼接进去；并去除千分位逗号。
- `max_new_tokens` 从 32 放宽到 256，保证多步推理链完整生成；预测取生成文本中最后一个数字。

上述修复已同步到 `colab_04_test_9b_35b.ipynb` 的评测函数，独立脚本与 notebook 逻辑一致。

---

## 5. 分析与局限

1. **小样本、高方差**：每子集仅 5 题、GSM8K 仅 8 题，单题即引起 0.125-0.2 的准确率波动，结果只能反映粗略趋势，不能作为稳定能力估计。
2. **不可与论文完整基准直接比较**：论文 Table 1 中 9B 的 8-bit MMLU 为 83.8（LeapQuant）/ 80.5（TurboQuant），基于完整 MMLU 全集；本报告仅 15 题抽样，口径与样本量均不同，数值差异属预期。
3. **当前为 bf16 基线**：本结果是未做循环状态量化的 bf16 参考基线，后续 TurboQuant / LeapQuant 状态注入的对比应在此基线之上进行。
4. **参考实现偏慢**：GSM8K 长文本生成受参考内核影响耗时较长，安装 flash 类内核可提速，不改变结果。

---

## 6. 35B-A3B 状态

- Qwen3.5-35B-A3B-FP8 共 14 个分片、总大小约 37.45GB（float8_e4m3fn，W8A8）。
- 数据盘（50GB，9B 已占约 19GB）与系统盘（30GB）单盘均放不下，且用户已明确指示**暂停 35B**，未下载。
- notebook 已加保护：本地无 35B 模型时直接跳过、不触发在线下载，保证整份 notebook 可安全 Run All。

---

## 7. 相关交付物

| 文件 | 说明 |
|---|---|
| `eval9b.py` | 独立小样本评测脚本（约束字母 argmax + 正则 GSM8K） |
| `debug_mmlu.py` | MMLU 换行符根因调试脚本 |
| `eval9b_results.json` | 评测结果（运行时生成于云端数据盘） |
| `colab_04_test_9b_35b.ipynb` | 同步修复后的 9B/35B 实测 notebook |
| `build_colab04_test_9b_35b.py` | 上述 notebook 的生成脚本 |
