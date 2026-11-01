# SPDX-License-Identifier: Apache-2.0
"""Smoke-run the TurboQuant-vs-LeapQuant notebook cells locally (CPU, Agg backend) to prove it executes."""
import json, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "turboquant_vs_leapquant.ipynb").read_text(encoding="utf-8"))
g = {}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    exec(compile("".join(c["source"]), f"<cell{i}>", "exec"), g)
    print(f"[cell {i}] ok")
print("NOTEBOOK OK")
