# -*- coding: utf-8 -*-
"""Debug why MMLU letter comparison reports miss even when the visible letters match."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

P = "/root/autodl-tmp/qwen9b"
m = AutoModelForCausalLM.from_pretrained(
    P, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
)
tok = AutoTokenizer.from_pretrained(P, local_files_only=True)
ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
q = ds[0]

s = q["question"] + "\n"
for i, c in enumerate(q["choices"]):
    s += "%s. %s\n" % (chr(65 + i), c)
s += "Answer:"
inp = tok(s, return_tensors="pt").to("cuda")
with torch.no_grad():
    lg = m(**inp).logits[0, -1]

tid = lg.argmax().item()
raw = tok.decode([tid])
pred = raw.strip().upper()
gold = chr(65 + q["answer"])
print("answer field:", repr(q["answer"]), type(q["answer"]))
print("gold repr:", repr(gold), [ord(x) for x in gold])
print("argmax token id:", tid)
print("raw decode repr:", repr(raw), [ord(x) for x in raw])
print("pred repr:", repr(pred), [ord(x) for x in pred])
print("eq pred==gold:", pred == gold)
print("--- candidate letter tokens ---")
for L in ["A", "B", "C", "D"]:
    id_plain = tok.encode(L, add_special_tokens=False)
    id_space = tok.encode(" " + L, add_special_tokens=False)
    lp = id_plain[0]
    print("letter", repr(L), "plain ids", id_plain, "space ids", id_space,
          "logit at plain id", round(float(lg[lp]), 3))
