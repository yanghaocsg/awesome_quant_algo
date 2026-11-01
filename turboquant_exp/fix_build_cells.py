# -*- coding: utf-8 -*-
"""Fix build scripts (02/03/04/05): move cell id into metadata and ensure every cell has metadata.
nbformat v4 requires cell_type/source/metadata on every cell."""
import re, os
R = r"C:\code_2026\turboquant量化\turboquant_exp"
FILES = ["build_colab02_leapquant.py", "build_colab03_test_08b.py",
         "build_colab04_test_9b_35b.py", "build_colab05_compare.py"]
MD_RE = re.compile(r'\{"cell_type": "markdown", "id": "(\w+)", "source": (MD\w+)},')
CODE_RE = re.compile(r'\{"cell_type": "code", "id": "(\w+)", "source": (\w+), "metadata": \{}\},')
for f in FILES:
    p = os.path.join(R, f)
    src = open(p, encoding="utf-8").read()
    n_md = len(MD_RE.findall(src))
    n_code = len(CODE_RE.findall(src))
    src2 = MD_RE.sub(r'{"cell_type": "markdown", "source": \2, "metadata": {"id": "\1"}},', src)
    src2 = CODE_RE.sub(r'{"cell_type": "code", "source": \2, "metadata": {"id": "\1"}},', src2)
    assert n_md > 0 and n_code > 0, f
    assert not MD_RE.findall(src2) and not CODE_RE.findall(src2), f
    open(p, "w", encoding="utf-8").write(src2)
    print(f, "md:", n_md, "code:", n_code)
