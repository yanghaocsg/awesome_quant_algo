# -*- coding: utf-8 -*-
"""Strict nbformat validation of all generated ipynb files."""
import json, os, sys
R = r"C:\code_2026\turboquant量化\turboquant_exp"
NAMES = ["colab_01_problem", "colab_02_leapquant", "colab_03_test_08b",
         "colab_04_test_9b_35b", "colab_05_compare"]
try:
    import nbformat
    HAVE = True
except Exception:
    HAVE = False
if not HAVE:
    print("nbformat not installed")
    sys.exit(2)
bad = 0
for n in NAMES:
    p = os.path.join(R, n + ".ipynb")
    try:
        nb = nbformat.read(p, as_version=4)
        nbformat.validate(nb)
        print(n, "VALID")
    except Exception as e:
        bad += 1
        print(n, "INVALID:", type(e).__name__, str(e)[:200])
sys.exit(0 if bad == 0 else 1)
