"""Convert a "# %%"-cell Python script into a Jupyter notebook. Usage: python src/build_notebook.py src/lgd_model.py notebooks/lgd_modelling.ipynb"""
import re, sys
import nbformat as nbf

src = open(sys.argv[1], encoding="utf-8").read()
blocks = re.split(r"^# %%(.*)$", src, flags=re.M)
nb = nbf.v4.new_notebook()
cells = []
for i in range(1, len(blocks), 2):
    header, body = blocks[i], blocks[i + 1].strip("\n")
    if "[markdown]" in header:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append(nbf.v4.new_markdown_cell(text))
    else:
        cells.append(nbf.v4.new_code_cell(body))
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[2])
print(f"{len(cells)} cells written to {sys.argv[2]}")
