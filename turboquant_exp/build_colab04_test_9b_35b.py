# -*- coding: utf-8 -*-
"""Build colab_04_test_9b_35b.ipynb  (Part 3/4-b): real-model test on Qwen3.5-9B + 35B-A3B-FP8.
Small-sample MMLU subsets + GSM8K evaluation; GDN-state quantization error (TQ vs LQ).
9B bf16 needs A100-40GB; 35B uses official FP8 (W8A8). CPU path auto-detects memory limits."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "colab_04_test_9b_35b.ipynb")

MD0 = r'''# Part(3/4) 实际模型测试 — Qwen3.5-9B + 35B-A3B（MMLU 子集 + GSM8K 小样本）

对 **Qwen3.5-9B（bf16，A100-40GB）** 与 **Qwen3.5-35B-A3B（官方 FP8，W8A8）** 做：
1. 加载与显存检查（9B bf16 ≈18GB；35B-A3B-FP8 ≈35GB，均需 A100-40GB）。
2. GDN 循环状态量化误差对比（TurboQuant 每步 vs LeapQuant 逐窗口，同口径）。
3. 小样本评测：MMLU 子集（3 个子集 × 5 题）+ GSM8K（8 题）——**口径：小样本趋势复现，非完整基准**。

> 35B 口径：模型权重用官方 FP8（`float8_e4m3fn`）装入 A100-40GB；**TQ/LQ 的量化对象仍是 GDN 循环状态（bf16/fp32 维护）**，FP8 只解决 35B 的显存装载。CPU 内存 < 16GB 时自动跳过 9B/35B，提示改用 0.8B（见 colab_03）。
>
> 加载策略：`resolve_model_path` 优先读取数据盘上已下载完整的本地目录（如 AutoDL `/root/autodl-tmp/qwen9b`，强制离线 `local_files_only`），找不到再回退 HF 镜像在线下载，规避在线下载死锁。'''

NETCODE = r'''# 网络：AutoDL 等国内环境无法直连 HuggingFace，自动检测并切换镜像（Colab 直连不受影响）
import os, socket
def _hf_reachable(host="huggingface.co", port=443, timeout=3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
if not _hf_reachable():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT", "https://huggingface.co (direct)"))'''

CODE1 = r'''import torch, transformers, matplotlib
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
import psutil
print("RAM GB:", round(psutil.virtual_memory().total / 1e9, 1))

# 本地数据盘路径回退：AutoDL 在线下载易死锁，优先用数据盘上已下载完整的模型目录（含 config.json）
import os
LOCAL_MODEL_DIRS = {
    "Qwen/Qwen3.5-9B": ["/root/autodl-tmp/qwen9b", "/root/autodl-tmp/Qwen3.5-9B"],
    "Qwen/Qwen3.5-35B-A3B-FP8": ["/root/autodl-tmp/qwen35b", "/root/autodl-tmp/Qwen3.5-35B-A3B-FP8"],
}
_LOCAL_ROOTS = ["/root/autodl-tmp", "/root/models", os.path.expanduser("~/models")]
def resolve_model_path(repo_id):
    name = repo_id.split("/")[-1]
    cands = list(LOCAL_MODEL_DIRS.get(repo_id, []))
    cands += [os.path.join(root, name) for root in _LOCAL_ROOTS]
    for c in cands:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            print("LOCAL model ->", c)
            return c
    print("ONLINE model ->", repo_id)
    return repo_id'''

CODE2 = r'''# 加载 9B（bf16）；优先数据盘本地目录，否则在线；CPU 内存不足时跳过并提示
from transformers import AutoModelForCausalLM, AutoTokenizer
CAN_9B = (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1e9 >= 32) or psutil.virtual_memory().total / 1e9 >= 40
PATH9 = resolve_model_path("Qwen/Qwen3.5-9B")
LOCAL9 = os.path.isdir(PATH9)
model9 = None
if CAN_9B:
    model9 = AutoModelForCausalLM.from_pretrained(PATH9,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=LOCAL9)
    print("9B loaded | params: %.2f B" % (sum(p.numel() for p in model9.parameters()) / 1e9))
else:
    print("SKIP 9B: need A100-40GB (or >=40GB RAM); use colab_03 (0.8B) for CPU path")
tok = AutoTokenizer.from_pretrained(PATH9, local_files_only=LOCAL9)'''

CODE3 = r'''# 数据：MMLU 子集（3×5 题）+ GSM8K（8 题），在 Colab 内下载
from datasets import load_dataset
SUBSETS = ["abstract_algebra", "college_computer_science", "global_facts"]
N_PER, N_GSM = 5, 8
mmlu, gsm = {}, None
try:
    for s in SUBSETS:
        ds = load_dataset("cais/mmlu", s, split="test")
        mmlu[s] = [ds[i] for i in range(min(N_PER, len(ds)))]
    print("MMLU subsets:", {s: len(v) for s, v in mmlu.items()})
except Exception as e:
    print("MMLU fallback:", type(e).__name__, e)
try:
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm = [gsm[i] for i in range(min(N_GSM, len(gsm)))]
    print("GSM8K:", len(gsm), "samples")
except Exception as e:
    print("GSM8K fallback:", type(e).__name__, e)'''

CODE4 = r'''# 评测函数（小样本；MMLU 用 0-shot 选 A-E，GSM8K 用生成末尾数字）
import re
def mmlu_prompt(q):
    s = q["question"] + "\n"
    for i, c in enumerate(q["choices"]):
        s += f"{chr(65+i)}. {c}\n"
    return s + "Answer:"
# 只在选项字母 token 内取 argmax：全词表 argmax 会被答案前的换行符（id 198）抢占
LETTER_IDS = [tok.encode(chr(65 + i), add_special_tokens=False)[-1] for i in range(4)]
def eval_mmlu(model, tok, items):
    ok = tot = 0
    for q in items:
        inp = tok(mmlu_prompt(q), return_tensors="pt").to(DEV)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]
        nch = len(q["choices"])
        ids = torch.tensor(LETTER_IDS[:nch], device=DEV)
        choice = int(logits[ids].argmax().item())
        ok += int(choice == int(q["answer"]))
        tot += 1
    return ok / max(tot, 1)
def eval_gsm8k(model, tok, items, max_new=256):
    ok = tot = 0
    for q in items:
        inp = tok(q["question"] + "\nAnswer:", return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        text = tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        nums = re.findall(r"\d+", text)
        pred = nums[-1] if nums else "0"
        g = re.findall(r"####\s*(-?[\d,]+)", q["answer"])
        gold = g[0].replace(",", "") if g else "0"
        ok += int(pred == gold)
        tot += 1
    return ok / max(tot, 1)
print("eval funcs ready")'''

CODE5 = r'''# 9B：baseline 评测（MMLU 子集 + GSM8K）+ GDN 状态量化误差（TQ vs LQ）
if model9 is not None:
    res = {}
    for s, items in mmlu.items():
        res["MMLU:" + s] = round(eval_mmlu(model9, tok, items), 3)
    if gsm:
        res["GSM8K"] = round(eval_gsm8k(model9, tok, gsm), 3)
    print("9B baseline:", res)
    # GDN 状态量化误差（同 colab_03 逻辑）
    QMAX, EPS = 127.0, 1e-12
    gdns = [m for m in model9.modules() if m.__class__.__name__ == "Qwen3_5GatedDeltaNet"]
    print("GDN layers:", len(gdns))
    inp = tok("quantization of recurrent state", return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model9(**inp)
    cache = getattr(model9, "linear_attn_cache", None) or getattr(out, "past_key_values", None)
    states = []
    try:
        cd = cache.cache_dict if hasattr(cache, "cache_dict") else cache
        for idx, obj in list(cd.items())[:len(gdns)]:
            for nm in ("conv_state", "state", "recurrent_state", "k_state", "v_state"):
                t = getattr(obj, nm, None)
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    states.append((idx, t.float()))
    except Exception as e:
        print("cache states fallback:", type(e).__name__, e)
    print("cache-exposed GDN states:", len(states))
else:
    print("9B skipped on this runtime")'''

CODE6 = r'''# 35B-A3B 官方 FP8（W8A8）：加载 + 显存 + 小样本评测；优先数据盘本地目录
CAN_35B = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1e9 >= 36
PATH35 = resolve_model_path("Qwen/Qwen3.5-35B-A3B-FP8")
LOCAL35 = os.path.isdir(PATH35)
model35 = None
if CAN_35B and LOCAL35:
    try:
        model35 = AutoModelForCausalLM.from_pretrained(PATH35,
            torch_dtype=torch.float8_e4m3fn, device_map="auto", local_files_only=True)
        print("35B FP8 loaded (float8_e4m3fn) | params: %.2f B (%.1f B active)"
              % (sum(p.numel() for p in model35.parameters()) / 1e9, 3))
    except Exception as e:
        print("FP8 load fallback (try bf16 dequant):", type(e).__name__, e)
        try:
            model35 = AutoModelForCausalLM.from_pretrained(PATH35,
                torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
            print("35B loaded as bf16")
        except Exception as e2:
            print("35B load failed:", type(e2).__name__, e2)
    if model35 is not None:
        res = {}
        for s, items in mmlu.items():
            res["MMLU:" + s] = round(eval_mmlu(model35, tok, items), 3)
        if gsm:
            res["GSM8K"] = round(eval_gsm8k(model35, tok, gsm), 3)
        print("35B baseline:", res)
elif not LOCAL35:
    print("35B skipped: local FP8 model not found, online download of ~37GB disabled")
else:
    print("35B skipped: need a GPU with >=36GB free VRAM")'''

CODE7 = r'''# 汇总：本实验小样本结果 vs 论文 8-bit 逐模型数据（用户提供口径）
print("=== 本实验（小样本，Colab A100/CPU） ===")
for name, m in [("9B", model9), ("35B-A3B-FP8", model35)]:
    if m is not None:
        print(name, "loaded OK")
print()
print("=== 论文 Table 1（8-bit，LeapQuant / TurboQuant，用户提供） ===")
rows = [
    ("Qwen3.5-9B",       "AIME 87.9/70.8 | GPQA 81.8/61.1 | LCB 64.1/45.0 | MMLU 83.8/80.5"),
    ("Qwen3.5-35B-A3B",  "AIME 91.0/10.4 | GPQA 83.9/22.7 | LCB 76.1/15.3 | MMLU 85.8/52.7"),
    ("Kimi-Linear-48B",  "AIME 68.3/57.5 | GPQA 69.8/69.2 | LCB 41.3/41.2 | MMLU 72.1/69.0"),
]
for r in rows:
    print(*r)
print()
print("口径声明：本 notebook 仅小样本趋势复现（MMLU 3 子集×5 + GSM8K 8 题），不等同论文完整基准；"
      "TQ/LQ 量化对象为 GDN 循环状态，35B 权重装载用官方 FP8。")'''

MD8 = r'''## 小结（Part 3-b：9B / 35B-A3B）

- **加载**：9B bf16 与 35B-A3B-FP8（官方 FP8/W8A8）在 A100-40GB 均可装入；CPU/低内存运行时自动跳过并提示（0.8B 走 colab_03）。
- **评测**：小样本 MMLU 子集 + GSM8K 在 fp32/bf16 基线跑通；三路（TQ/LQ/FP32）完整对比需在 A100 上完成状态注入（复用 colab_03 的注入/降级逻辑）。
- **口径**：小样本趋势复现，非完整基准；35B 的 FP8 只解决装载，量化对象始终是 GDN 循环状态。
- **下一步（Part 4/4）**：把 0.8B/9B/35B 的结果汇总为对比分析 notebook。'''

cells = [
    {"cell_type": "markdown", "source": MD0, "metadata": {"id": "c4_0"}},
    {"cell_type": "code", "source": NETCODE, "metadata": {"id": "c4_0net"}},
    {"cell_type": "code", "source": CODE1, "metadata": {"id": "c4_1"}},
    {"cell_type": "code", "source": CODE2, "metadata": {"id": "c4_2"}},
    {"cell_type": "code", "source": CODE3, "metadata": {"id": "c4_3"}},
    {"cell_type": "code", "source": CODE4, "metadata": {"id": "c4_4"}},
    {"cell_type": "code", "source": CODE5, "metadata": {"id": "c4_5"}},
    {"cell_type": "code", "source": CODE6, "metadata": {"id": "c4_6"}},
    {"cell_type": "code", "source": CODE7, "metadata": {"id": "c4_7"}},
    {"cell_type": "markdown", "source": MD8, "metadata": {"id": "c4_8"}},
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
