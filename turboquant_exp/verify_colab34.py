# -*- coding: utf-8 -*-
"""Syntax-verify + partial-smoke colab_03 / colab_04.
Model-loading cells need transformers+GPU and are skipped locally; torch-only cells are executed."""
import json, io, sys, os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib; matplotlib.use("Agg")

R = r"C:\code_2026\turboquant量化\turboquant_exp"
NAMES = ["colab_03_test_08b", "colab_04_test_9b_35b"]
SMOKE_OK_CELLS = {"colab_03_test_08b": {"c3_1", "c3_4"}}  # env + ops (torch-only)
out = []
for name in NAMES:
    p = os.path.join(R, name + ".ipynb")
    nb = json.load(open(p, encoding="utf-8"))
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errs = []
    ns = {}
    for c in codes:
        src = "".join(c["source"])
        cid = c.get("id") or c["metadata"].get("id", "?")
        try:
            compile(src, "<cell>", "exec")
        except SyntaxError as e:
            errs.append((cid, str(e)))
            continue
        if name in SMOKE_OK_CELLS and cid in SMOKE_OK_CELLS[name]:
            buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
            try:
                exec(compile(src, "<cell>", "exec"), ns)
                sys.stdout = old
                out.append(f"[SMOKE-OK {name}/{cid}] " + buf.getvalue().strip())
            except Exception as e:
                sys.stdout = old
                out.append(f"[SMOKE-ERR {name}/{cid}] {type(e).__name__}: {e}")
    print(name, "| code cells:", len(codes), "| syntax errors:", errs if errs else "NONE")
open(os.path.join(R, "smoke_34.txt"), "w", encoding="utf-8").write("\n\n".join(out))
print("partial smoke written to smoke_34.txt")
