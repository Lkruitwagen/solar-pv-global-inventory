#!/usr/bin/env python

def strip_output(nb):
    for ws in nb.worksheets:
        for cell in ws.cells:
            if hasattr(cell, "outputs"):
                cell.outputs = []
            if hasattr(cell, "prompt_number"):
                del cell["prompt_number"]


if __name__ == "__main__":
    from sys import stdin, stdout
    from IPython.nbformat.current import read, write

    nb = read(stdin, "ipynb")
    strip_output(nb)
    write(nb, stdout, "ipynb")
    stdout.write("\n")