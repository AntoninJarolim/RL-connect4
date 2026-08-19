# Connect 4 with REINFORCE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AntoninJarolim/RL-connect4/blob/main/connect4_reinforce.ipynb)

Hands-on notebook for the BISSIT summer school RL session: train a pure **online
REINFORCE** agent to play Connect 4 against a random opponent, then play against it
yourself.

- **Students:** click the badge above, then `File → Save a copy in Drive`. You fill in
  4 small TODOs (plus one optional bonus); each is followed by a ✅ check cell. No
  installs, no GPU needed — the full training run takes a few minutes on the free CPU
  runtime.
- **Solutions:** [`solutions/connect4_reinforce_solutions.ipynb`](solutions/connect4_reinforce_solutions.ipynb)
  ([open in Colab](https://colab.research.google.com/github/AntoninJarolim/RL-connect4/blob/main/solutions/connect4_reinforce_solutions.ipynb)).

## Development

The `.py` files (jupytext percent format) are the source of truth — never edit the
`.ipynb` files by hand:

```bash
pip install torch numpy matplotlib jupytext pytest nbconvert ipykernel

# regenerate the notebooks after editing the .py files
jupytext --to ipynb connect4_reinforce.py
jupytext --to ipynb solutions/connect4_reinforce_solutions.py

# run the test suite (executes the solutions notebook in FAST mode)
pytest tests/ -q

# skeleton and solutions must differ only inside TODO cells
python tools/check_todo_delta.py

# full headless execution of the solutions notebook
HEADLESS=1 jupyter nbconvert --to notebook --execute \
    solutions/connect4_reinforce_solutions.ipynb --output /tmp/sol_out.ipynb
```

`FAST=1` shrinks all training budgets for CI-speed runs; `HEADLESS=1` skips the
interactive play cell and live plotting.
