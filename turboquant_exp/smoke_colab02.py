import json, io, sys, os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib; matplotlib.use("Agg")
P = r"C:\code_2026\turboquant量化\turboquant_exp\colab_02_leapquant.ipynb"
nb = json.load(open(P, encoding="utf-8"))
out = []
ns = {}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        exec(compile(src, "<cell>", "exec"), ns)
        sys.stdout = old
        out.append("[OK] " + c["metadata"].get("id", "?") + "\n" + buf.getvalue())
    except Exception as e:
        sys.stdout = old
        out.append("[ERR] " + c["metadata"].get("id", "?") + " " + type(e).__name__ + ": " + str(e))
        out.append("  in:\n" + src[:1500])
open(r"C:\code_2026\turboquant量化\turboquant_exp\smoke_02.txt", "w", encoding="utf-8").write("\n\n".join(out))
print("smoke complete")
