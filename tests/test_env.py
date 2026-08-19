"""Tests for the Connect-4 REINFORCE notebook.

The solutions notebook (its .py jupytext pairing) is executed once in FAST +
HEADLESS mode; every test then runs against the resulting namespace. The
execution itself already exercises all in-notebook ✅ assert cells plus a small
end-to-end training run.
"""
import os
from pathlib import Path

os.environ["FAST"] = "1"
os.environ["HEADLESS"] = "1"
import matplotlib

matplotlib.use("Agg")

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SOLUTIONS = REPO / "solutions" / "connect4_reinforce_solutions.py"
SKELETON = REPO / "connect4_reinforce.py"

# A random game that ends in a draw (found offline, replayed here move by move).
DRAW_SEQUENCE = [0, 5, 2, 1, 6, 3, 1, 5, 0, 4, 3, 0, 3, 6, 6, 6, 4, 1, 0, 2, 3,
                 4, 1, 3, 5, 2, 6, 6, 0, 4, 3, 0, 5, 2, 2, 5, 1, 1, 2, 4, 4, 5]

# Same value as FINGERPRINT_TODO2 in the notebook (pinned from the reference run).
FINGERPRINT = 270192


@pytest.fixture(scope="session")
def nb():
    """Execute the full solutions notebook in FAST mode, return its namespace."""
    ns = {"__name__": "__main__"}
    exec(compile(SOLUTIONS.read_text(), str(SOLUTIONS), "exec"), ns)
    return ns


def make_plane(cells):
    plane = torch.zeros(1, 6, 7)
    for r, c in cells:
        plane[0, r, c] = 1.0
    return plane


# ---------------------------------------------------------------- win detection

@pytest.mark.parametrize("cells", [
    [(5, 0), (5, 1), (5, 2), (5, 3)],          # horizontal, bottom-left corner
    [(0, 3), (0, 4), (0, 5), (0, 6)],          # horizontal, top-right corner
    [(2, 4), (3, 4), (4, 4), (5, 4)],          # vertical, middle
    [(0, 6), (1, 6), (2, 6), (3, 6)],          # vertical, right edge
    [(0, 0), (1, 1), (2, 2), (3, 3)],          # diagonal down-right, corner
    [(2, 3), (3, 4), (4, 5), (5, 6)],          # diagonal down-right, corner
    [(5, 0), (4, 1), (3, 2), (2, 3)],          # diagonal up-right, corner
    [(3, 3), (2, 4), (1, 5), (0, 6)],          # diagonal up-right, corner
])
def test_four_in_a_row_detected(nb, cells):
    assert nb["check_four_in_a_row"](make_plane(cells)).item()


@pytest.mark.parametrize("cells", [
    [],                                        # empty board
    [(5, 0), (5, 1), (5, 2)],                  # only three horizontal
    [(5, 0), (5, 1), (5, 2), (5, 4)],          # four with a gap
    [(3, 4), (4, 4), (5, 4)],                  # only three vertical
    [(0, 0), (1, 1), (2, 2), (3, 4)],          # broken diagonal
    [(5, 3), (4, 3), (3, 3), (5, 4), (4, 4), (3, 4)],  # 3+3 block, no four
])
def test_no_false_positive(nb, cells):
    assert not nb["check_four_in_a_row"](make_plane(cells)).item()


# ------------------------------------------------------------------ environment

def test_gravity_stacks_pieces(nb):
    env = nb["VectorConnect4"](1)
    for _ in range(4):
        env.step(torch.tensor([3]))
    heights = env.board[0].sum(dim=(0, 1))
    assert heights[3] == 4 and heights.sum() == 4
    # discs occupy the four lowest rows of column 3, alternating between players
    column = env.board[0, :, :, 3]
    assert (column.sum(dim=0)[2:] == 1).all() and column.sum(dim=0)[:2].sum() == 0
    assert torch.equal(column[0, 2:], torch.tensor([0.0, 1.0, 0.0, 1.0])) or \
        torch.equal(column[0, 2:], torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_canonical_channels_flip_between_plies(nb):
    env = nb["VectorConnect4"](1)
    env.step(torch.tensor([2]))
    # It is now player 1's turn: the disc player 0 just dropped sits in channel 1.
    assert env.current_player == 1
    assert env.board[0, 1, 5, 2] == 1.0 and env.board[0, 0].sum() == 0
    assert int(env.channel0_player[0]) == 1
    env.step(torch.tensor([4]))
    # Back to player 0's view: own disc in channel 0, opponent's in channel 1.
    assert env.board[0, 0, 5, 2] == 1.0 and env.board[0, 1, 5, 4] == 1.0


def test_legal_mask_and_full_column(nb):
    env = nb["VectorConnect4"](1)
    assert env.legal_mask().all()
    for _ in range(6):
        env.step(torch.tensor([0]))
    mask = env.legal_mask()[0]
    assert not mask[0] and mask[1:].all()
    with pytest.raises(AssertionError):
        env.step(torch.tensor([0]))


def test_win_ends_game_and_sets_z(nb):
    env = nb["VectorConnect4"](1)
    for a in [0, 0, 1, 1, 2, 2, 3]:            # player 0 builds the bottom row
        env.step(torch.tensor([a]))
    assert not env.active[0] and int(env.winner[0]) == 0
    assert int(env.game_plies[0]) == 7
    assert float(env.z(0)[0]) == 1.0 and float(env.z(1)[0]) == -1.0
    board_before = env.board.clone()
    env.step(torch.tensor([6]))                # finished games are frozen
    assert torch.equal(env.board, board_before)


def test_draw_detection(nb):
    env = nb["VectorConnect4"](1)
    for a in DRAW_SEQUENCE:
        env.step(torch.tensor([a]))
    assert not env.active[0]
    assert int(env.winner[0]) == -1
    assert float(env.z(0)[0]) == 0.0 and float(env.z(1)[0]) == 0.0
    assert int(env.game_plies[0]) == 42


# ---------------------------------------------------------------------- returns

def test_returns_hand_computed(nb):
    compute_returns = nb["compute_returns"]
    # A player who made 3 moves and won: G = gamma^(moves after) * z.
    move_index = torch.tensor([0, 1, 2])
    moves_in_game = torch.tensor([3, 3, 3])
    win = torch.ones(3)
    assert torch.allclose(compute_returns(move_index, moves_in_game, win, 1.0),
                          torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(compute_returns(move_index, moves_in_game, win, 0.9),
                          torch.tensor([0.81, 0.9, 1.0]))
    # The losing side of the same game made 2 moves (both players, gamma 0.9).
    assert torch.allclose(compute_returns(torch.tensor([0, 1]), torch.tensor([2, 2]),
                                          -torch.ones(2), 0.9),
                          torch.tensor([-0.9, -1.0]))


def test_returns_in_rollout(nb):
    nb["set_seed"](5)
    net = nb["PolicyNet"]()
    ro = nb["play_games"](net, 6, nb["random_opponent"], gamma=0.9)
    counts = torch.bincount(ro.game_idx, minlength=6)
    for game in range(6):
        G = ro.G[ro.game_idx == game]          # this game's returns, in move order
        z = float(ro.z[game])
        expected = torch.tensor([0.9 ** (int(counts[game]) - 1 - k) * z
                                 for k in range(int(counts[game]))])
        assert torch.allclose(G, expected, atol=1e-6)


# ------------------------------------------------------------------------- loss

def test_reference_loss_value(nb):
    log_probs = torch.tensor([-1.0, -0.5, -2.0, -0.25])
    G = torch.tensor([1.0, -1.0, 1.0, 0.0])
    loss = nb["reinforce_loss"](log_probs, G)
    assert abs(float(loss) - 0.625) < 1e-6     # a flipped sign would give -0.625


# ---------------------------------------------------------------- reproducibility

def test_fixed_seed_reproduces_fingerprint(nb):
    rollouts = []
    for _ in range(2):
        nb["set_seed"](123)
        net = nb["PolicyNet"]()
        rollouts.append(nb["play_games"](net, 8, nb["random_opponent"], gamma=1.0))
    first, second = rollouts
    assert torch.equal(first.actions, second.actions), "same seed must give the same games"
    assert torch.equal(first.G, second.G)
    fingerprint = int(first.actions.sum()) * 1000 + int(first.game_plies.sum())
    assert fingerprint == FINGERPRINT


# --------------------------------------------------------------- smoke training

def test_smoke_training_improves(nb):
    win_rate = nb["history"]["win_rate"]       # the FAST run: 30 updates, seeded
    start = sum(win_rate[:5]) / 5
    end = sum(win_rate[-5:]) / 5
    assert end > start + 0.1, (
        f"training did not improve enough: {start:.2f} -> {end:.2f} "
        "(a sign error in the loss shows up exactly like this)")


def test_trained_agent_beats_random(nb):
    stats = nb["evaluate"](nb["net"], nb["random_opponent"], n_games=500)
    assert stats["win"] > 0.75                 # FAST budget; the full run reaches >= 0.9


# --------------------------------------------------------------------- skeleton

def test_skeleton_fails_at_todo1_only():
    ns = {"__name__": "__main__"}
    with pytest.raises(NotImplementedError, match="TODO 1"):
        exec(compile(SKELETON.read_text(), str(SKELETON), "exec"), ns)
    # everything before TODO 1 must have executed cleanly
    assert "VectorConnect4" in ns and "ConvTrunk" in ns
