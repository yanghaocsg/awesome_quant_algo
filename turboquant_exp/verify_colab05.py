# -*- coding: utf-8 -*-
"""Verify + full-smoke colab_05_compare.ipynb (all cells are torch/matplotlib only)."""
import json, io, sys, os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib; matplotlib.use("Agg")

R = r"C:\code_2026\turboquant量化\turboquant_exp"
p = os.path.join(R, "colab_05_compare.ipynb")
nb = json.load(open(p, encoding="utf-8"))
errs, out = [], []
ns = {}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    cid = c.get("id") or "?"
    try:
        compile(src, "<cell>", "exec")
    except SyntaxError as e:
        errs.append((cid, str(e)))
        continue
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        exec(compile(src, "<cell>", "exec"), ns)
        sys.stdout = old
        out.append(f"[OK {cid}] " + buf.getvalue().strip())
    except Exception as e:
        sys.stdout = old
        out.append(f"[ERR {cid}] {type(e).__name__}: {e}")
print("syntax errors:", errs if errs else "NONE")
open(os.path.join(R, "smoke_05.txt"), "w", encoding="utf-8").write("\n\n".join(out))
print("full smoke written to smoke_05.txt")
