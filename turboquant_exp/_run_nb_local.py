import matplotlib
matplotlib.use("Agg")
import nbformat

path = r"D:\gitcode_2026\awesome_quant_algo\turboquant_exp\colab_01_problem.ipynb"
nb = nbformat.read(path, as_version=4)
ns = {}
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    print(f"\n----- code cell {i} -----")
    exec(compile(cell.source, f"<cell {i}>", "exec"), ns)
print("\nNOTEBOOK EXECUTION OK")
