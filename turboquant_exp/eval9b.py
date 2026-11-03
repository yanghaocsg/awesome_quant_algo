# -*- coding: utf-8 -*-
"""Standalone small-sample evaluation of Qwen3.5-9B on the AutoDL GPU instance.

Loads the local data-disk model (bf16, offline), then runs:
  - MMLU 3 subsets x 5 questions (0-shot, single-token A-E choice)
  - GSM8K 8 questions (greedy generation, final number compared with gold #### N)
Scope: small-sample trend reproduction, NOT a full benchmark.
Run: python eval9b.py
"""
import os
import socket
import time
import json
import re

import torch

# Network: switch to HF mirror when huggingface.co is unreachable (datasets only here).
try:
    socket.create_connection(("huggingface.co", 443), timeout=3)
except OSError:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "/root/autodl-tmp/qwen9b"
RESULT_PATH = "/root/autodl-tmp/eval9b_results.json"

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
print("LOADED B %.2f sec %.1f"
      % (sum(p.numel() for p in model.parameters()) / 1e9, time.time() - t0),
      flush=True)

SUBSETS = ["abstract_algebra", "college_computer_science", "global_facts"]
N_PER, N_GSM = 5, 8
mmlu = {}
for s in SUBSETS:
    ds = load_dataset("cais/mmlu", s, split="test")
    mmlu[s] = [ds[i] for i in range(min(N_PER, len(ds)))]
gsm_full = load_dataset("openai/gsm8k", "main", split="test")
gsm = [gsm_full[i] for i in range(min(N_GSM, len(gsm_full)))]
print("data: MMLU", {s: len(v) for s, v in mmlu.items()}, "| GSM8K", len(gsm),
      flush=True)


def mmlu_prompt(q):
    s = q["question"] + "\n"
    for i, c in enumerate(q["choices"]):
        s += "%s. %s\n" % (chr(65 + i), c)
    return s + "Answer:"


def eval_mmlu(items):
    ok = tot = 0
    detail = []
    for q in items:
        inp = tok(mmlu_prompt(q), return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]
        pred = tok.decode([logits.argmax().item()]).strip().upper()
        gold = chr(65 + q["answer"])
        hit = pred == gold
        ok += int(hit)
        tot += 1
        detail.append((gold, pred, int(hit)))
    return ok / max(tot, 1), detail


def eval_gsm8k(items, max_new=256):
    ok = tot = 0
    detail = []
    for q in items:
        inp = tok(q["question"] + "\nAnswer:", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        text = tok.decode(out[0][inp["input_ids"].shape[-1]:],
                          skip_special_tokens=True)
        nums = re.findall(r"\d+", text)
        pred = nums[-1] if nums else "0"
        gold_match = re.findall(r"####\s*(-?[\d,]+)", q["answer"])
        gold = gold_match[0].replace(",", "") if gold_match else "0"
        hit = pred == gold
        ok += int(hit)
        tot += 1
        detail.append((gold, pred, int(hit)))
    return ok / max(tot, 1), detail


results = {}
for s, items in mmlu.items():
    acc, detail = eval_mmlu(items)
    results["MMLU:" + s] = round(acc, 3)
    print("MMLU %s acc %.2f" % (s, acc), detail, flush=True)

all_mmlu = [v for items in mmlu.values() for v in items]
macc, _ = eval_mmlu(all_mmlu)
print("MMLU_OVERALL %.2f (15 Q)" % macc, flush=True)
results["MMLU_overall"] = round(macc, 3)

gacc, gdetail = eval_gsm8k(gsm)
results["GSM8K"] = round(gacc, 3)
print("GSM8K acc %.2f (8 Q)" % gacc, gdetail, flush=True)

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print("saved", RESULT_PATH, flush=True)
print("EVAL9B_DONE", flush=True)
