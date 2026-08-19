#!/usr/bin/env python3
"""Generate the student skeleton from the solutions file.

Copies solutions/connect4_reinforce_solutions.py to connect4_reinforce.py,
replacing the body of every '# TODO-CELL' cell with its skeleton version
(NotImplementedError placeholders). Run tools/check_todo_delta.py afterwards.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOLUTIONS = REPO / "solutions" / "connect4_reinforce_solutions.py"
SKELETON = REPO / "connect4_reinforce.py"

# Skeleton bodies, keyed by a substring that identifies the TODO cell.
SKELETON_CELLS = {
    "class PolicyNet": '''# %%
# TODO-CELL
class PolicyNet(nn.Module):
    """The policy pi_theta: board -> 7 masked logits (one per column)."""

    def __init__(self):
        super().__init__()
        self.trunk = ConvTrunk()      # given: board -> 128 features
        # TODO 1a: create the head — a linear layer from 128 features to 7 logits.
        raise NotImplementedError("TODO 1: create the final linear layer (128 -> 7 logits)")

    def forward(self, boards, legal_mask):
        features = self.trunk(boards)                          # (N, 128)
        # TODO 1b:
        #   1. push the features through your head -> logits, shape (N, 7)
        #   2. set the logits of illegal columns to -1e9
        #   3. return the masked logits
        raise NotImplementedError("TODO 1: compute the masked logits")

''',
    "def sample_action": '''# %%
# TODO-CELL
def sample_action(masked_logits):
    """a ~ pi_theta(.|s): sample a column per board. Returns (actions, log_probs)."""
    # TODO 2: follow the REINFORCE example at the top of
    # https://docs.pytorch.org/docs/2.13/distributions.html
    # (one difference: we have masked LOGITS, so build Categorical(logits=...))
    raise NotImplementedError("TODO 2: sample from the policy")

''',
    "def reinforce_loss": '''# %%
# TODO-CELL
def reinforce_loss(log_probs, G):
    """REINFORCE: loss = -mean( log pi_theta(a_t|s_t) * G_t )."""
    # TODO 3: one line. Remember the minus sign.
    raise NotImplementedError("TODO 3: the REINFORCE loss")

''',
    "USE_BASELINE": '''# %%
# TODO-CELL
USE_BASELINE = False  # set to True once you implement advantage() below


def advantage(G):
    """Bonus: G minus the batch-mean baseline."""
    raise NotImplementedError("Bonus TODO: return G minus its mean")

''',
    "def greedy_action": '''# %%
# TODO-CELL
def greedy_action(net, boards):
    """argmax_a pi_theta(a|s): the most likely legal column, one per board."""
    legal = legal_move_mask(boards)   # given: (N, 7) bool
    # TODO 4: get the masked logits from the net, return the argmax column per board.
    raise NotImplementedError("TODO 4: the greedy action")

''',
}


def split_cells(text):
    cells, current = [], []
    for line in text.splitlines(keepends=True):
        if line.startswith("# %%"):
            cells.append("".join(current))
            current = []
        current.append(line)
    cells.append("".join(current))
    return cells


cells = split_cells(SOLUTIONS.read_text())
used = set()
for i, cell in enumerate(cells):
    if "# TODO-CELL" not in cell:
        continue
    for key, replacement in SKELETON_CELLS.items():
        if key in cell:
            cells[i] = replacement
            used.add(key)
            break
    else:
        sys.exit(f"FAIL: TODO cell {i} matches no known skeleton replacement")

if used != set(SKELETON_CELLS):
    sys.exit(f"FAIL: unused skeleton replacements: {set(SKELETON_CELLS) - used}")

SKELETON.write_text("".join(cells))
print(f"OK: wrote {SKELETON.relative_to(REPO)} ({len(used)} TODO cells replaced)")
