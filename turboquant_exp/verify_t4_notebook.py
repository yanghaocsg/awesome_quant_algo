# SPDX-License-Identifier: Apache-2.0
"""Smoke-run the T4 benchmark notebook cells locally (CPU) to prove the logic executes correctly.
Skips the GPU-requirement assert cell (the notebook targets Colab T4)."""
import json, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "leapquant_vs_turboquant_t4.ipynb").read_text(encoding="utf-8"))
g = {}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "assert torch.cuda.is_available" in src:
        print(f"[cell {i}] SKIP (GPU assert, for Colab)")
        continue
    exec(compile(src, f"<cell{i}>", "exec"), g)
    print(f"[cell {i}] ok")
print("NOTEBOOK OK")
