# SPDX-License-Identifier: Apache-2.0
"""Smoke-run the generated Colab notebook cell-by-cell on the local machine (CPU),
to prove the notebook's loading chain and namespaces actually execute before you open it in Colab.
Replaces the full run cell (index 10) with a one-batch smoke to keep it fast.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "leapquant_torch_port.ipynb").read_text(encoding="utf-8"))
g = {}
skipped_run_cell = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if i == 10:  # the full BATCHES run cell -- do a one-batch smoke instead below
        skipped_run_cell = src
        continue
    exec(compile(src, f"<cell{i}>", "exec"), g)
    print(f"[cell {i}] ok")

print("--- smoke: one batch (16) per port via run_check ---")
for w, port, ref in (("gdn", g["gdn_torch_run"], g["ref_gdn"]), ("kda", g["kda_torch_run"], g["ref_kda"])):
    for bs, name, ok, ratio in g["run_check"](w, port, ref, [16], g["DEV"]):
        print(f"{w} | {name:<13} | {'PASS' if ok else 'FAIL'} | worst ratio {ratio:.6f}")
print("NOTEBOOK OK")
