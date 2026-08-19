#!/usr/bin/env python3
"""Verify that the skeleton and the solutions differ ONLY inside TODO cells.

Both jupytext .py files are split into cells (delimited by '# %%' lines) and
compared pairwise. A cell may differ only if BOTH versions carry the
'# TODO-CELL' marker; every skeleton TODO cell must contain a
NotImplementedError placeholder. Exits non-zero on any violation.
"""
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKELETON = REPO / "connect4_reinforce.py"
SOLUTIONS = REPO / "solutions" / "connect4_reinforce_solutions.py"

EXPECTED_TODO_CELLS = 5  # TODO 1-4 + the optional bonus


def split_cells(path):
    cells, current = [], []
    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            cells.append("\n".join(current))
            current = []
        current.append(line)
    cells.append("\n".join(current))
    return cells


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


skel_cells = split_cells(SKELETON)
sol_cells = split_cells(SOLUTIONS)

if len(skel_cells) != len(sol_cells):
    fail(f"cell count differs: skeleton {len(skel_cells)} vs solutions {len(sol_cells)}")

todo_cells = 0
for i, (skel, sol) in enumerate(zip(skel_cells, sol_cells)):
    is_todo = "# TODO-CELL" in skel and "# TODO-CELL" in sol
    if is_todo:
        todo_cells += 1
        if "NotImplementedError" not in skel:
            fail(f"cell {i} is a TODO cell but the skeleton has no NotImplementedError placeholder")
        continue
    if skel != sol:
        diff = "\n".join(difflib.unified_diff(
            skel.splitlines(), sol.splitlines(),
            fromfile="skeleton", tofile="solutions", lineterm=""))
        fail(f"cell {i} differs but is not marked # TODO-CELL in both files:\n{diff}")

if todo_cells != EXPECTED_TODO_CELLS:
    fail(f"expected {EXPECTED_TODO_CELLS} TODO cells, found {todo_cells}")

print(f"OK: {len(skel_cells)} cells compared, files differ only in the {todo_cells} TODO cells.")
