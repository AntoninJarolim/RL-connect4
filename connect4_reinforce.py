# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Connect 4 with REINFORCE
#
# **BISSIT hands-on session — policy gradients from scratch**
#
# ⚠️ **First step: click `File → Save a copy in Drive`.** Otherwise your changes are lost when the session ends.
#
# In this notebook you train a neural network to play Connect 4, using the **REINFORCE**
# algorithm from the lecture. You write 4 small pieces of code (**TODO 1–4**); everything
# else is given. Each TODO is followed by a ✅ check cell that tells you if your code is
# correct. At the end, you play against your own agent.
#
# Suggested time: TODO 1 ≈ 10 min · TODO 2 ≈ 10 min · TODO 3 ≈ 5 min · training ≈ 5 min · TODO 4 + play ≈ 10 min.

# %% [markdown]
# ## 0. Setup
#
# Run the cells below. No installs are needed — everything is preinstalled in Colab.

# %%
import os
import random
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Flags used by automated runs of this notebook. In Colab both are False.
FAST = os.environ.get("FAST") == "1"          # tiny training budgets (for CI)
HEADLESS = os.environ.get("HEADLESS") == "1"  # no live plots, no input()

SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | {device} available — but everything runs on CPU: the net is tiny")

# %% [markdown]
# ### Notation
#
# The notebook uses the same names as the lecture slides:
#
# | in the notebook | on the slides | meaning |
# |---|---|---|
# | `obs`, `boards` | $o_t = s_t$ | the board — the fully observed state |
# | `logits` → `softmax` | $\pi_\theta(\cdot \mid s)$ | the policy: a distribution over the 7 columns |
# | `a = dist.sample()` | $a_t \sim \pi_\theta(\cdot \mid s_t)$ | action selection during **training** |
# | `greedy_action(...)` | $\arg\max_a \pi_\theta(a \mid s)$ | action selection at **inference** time |
# | `G` | $G(\tau_{t:}) = \sum_k \gamma^k\, r_{t+k+1}$ | discounted return; here the only reward is the final $z \in \{-1, 0, +1\}$ |
# | `gamma` | $\gamma$ | discount factor, `1.0` in the main path |
# | the net's weights | $\theta$ (green) | parameters of the policy |
# | (bonus) baseline | previews $V_\varphi$, $\varphi$ (red) | actor-critic teaser |
#
# Connect 4 is **fully observed**: the board is the whole state, $o_t = s_t$.
# So we simply write $s$ everywhere.
# As on the slides, the reward for the move at time $t$ arrives one tick later, as $r_{t+1}$.

# %% [markdown]
# ## 1. The environment: Connect 4 as an MDP
#
# Rules: two players take turns dropping a disc into one of 7 columns. The disc falls to
# the lowest free cell. Four own discs in a row — horizontally, vertically, or diagonally —
# win. A full board with no winner is a draw.
#
# The class `VectorConnect4` below **is** the MDP from the lecture:
#
# - **state $s$** — the board: a `(2, 6, 7)` tensor. Channel 0 holds the discs of the
#   player *to move*, channel 1 the opponent's discs. After every move the channels swap,
#   so the network always sees the game from the current player's side.
# - **actions $a$** — the 7 columns. A column is legal **iff its top cell is free**.
#   (That is the only legality rule in Connect 4.)
# - **transitions $P$** — `step(actions)` drops the discs and flips the board to the
#   next player's view.
# - **rewards $R$** — 0 after every move; when a game ends, $z = +1$ win / $0$ draw /
#   $-1$ loss, from each player's own point of view.
#
# It plays `N` games at once as one batch of tensors ("vectorized") — that is what makes
# training fast enough for this session. Finished games freeze and are skipped.
#
# You only **read** this section. The next cell holds boring helpers you can skip entirely.

# %% [markdown]
# ### 🔒 Helper functions (black box — you do not need to read this cell)
#
# - `set_seed(seed)` — makes runs reproducible.
# - `legal_move_mask(boards) -> mask` — `(N, 7)` bool: which columns can be played.
# - `check_four_in_a_row(planes) -> wins` — `(N,)` bool: does this player have 4 in a row.
# - `render_board(board, channel0_player)` — pretty-prints one board.
# - `read_human_move(legal)` — asks you for a column (used in Section 6).
# - `smoothed(values)` — moving average, for readable curves.
# - `update_live_plot(history)` / `plot_curves(curves)` — plotting plumbing.

# %%
def set_seed(seed):
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def legal_move_mask(boards):
    """(N, 2, 6, 7) boards -> (N, 7) bool mask: a column is legal iff its top cell is empty."""
    return boards[:, :, 0, :].sum(dim=1) == 0


# One 4x4 kernel per win direction: —, |, \ and /.
_WIN_KERNELS = torch.zeros(4, 1, 4, 4)
_WIN_KERNELS[0, 0, 0, :] = 1.0
_WIN_KERNELS[1, 0, :, 0] = 1.0
_WIN_KERNELS[2, 0] = torch.eye(4)
_WIN_KERNELS[3, 0] = torch.eye(4).flip(1)


def check_four_in_a_row(planes):
    """(N, 6, 7) planes with one player's discs -> (N,) bool: 4 in a row anywhere."""
    hits = F.conv2d(planes.unsqueeze(1), _WIN_KERNELS, padding=3)
    return (hits > 3.5).flatten(1).any(dim=1)


def render_board(board, channel0_player):
    """Print one (2, 6, 7) canonical board. X = first player, O = second player."""
    own, opp = ("X", "O") if channel0_player == 0 else ("O", "X")
    for r in range(6):
        print(" ".join(own if board[0, r, c] > 0 else opp if board[1, r, c] > 0 else "."
                       for c in range(7)))
    print(" ".join(str(c) for c in range(7)))


def read_human_move(legal):
    """Ask for a column until a legal one is typed. Returns None to quit."""
    while True:
        try:
            raw = input("Your column (0-6, q to quit): ").strip().lower()
        except Exception:  # no keyboard attached (headless run)
            print("(no interactive input available — quitting)")
            return None
        if raw == "q":
            return None
        if raw in list("0123456") and legal[int(raw)]:
            return int(raw)
        print("Not a legal move, try again.")


def smoothed(values, window=10):
    """Moving average with the given window."""
    if len(values) < 2 * window:
        return list(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_curves(curves, ylabel, target=None, title=""):
    """Plot one labelled line per entry of the dict `curves`."""
    plt.figure(figsize=(7, 3.2))
    for label, values in curves.items():
        plt.plot(values, label=label)
    if target is not None:
        plt.axhline(target, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("update")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def update_live_plot(history, target=0.9):
    """Redraw the training curve in place (plain prints when running headless)."""
    n, wr = len(history["win_rate"]), history["win_rate"]
    if HEADLESS:
        print(f"update {n}: win rate {wr[-1]:.2f}, loss {history['loss'][-1]:+.3f}")
        return
    from IPython.display import clear_output
    clear_output(wait=True)
    plt.figure(figsize=(7, 3.2))
    plt.plot(wr, alpha=0.3, label="win rate (per batch)")
    plt.plot(smoothed(wr), color="C0", label="win rate (smoothed)")
    plt.axhline(target, color="gray", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("update")
    plt.ylabel("win rate vs random")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


# %% [markdown]
# ### The environment class (given — read it, you will not modify it)

# %%
class VectorConnect4:
    """N Connect-4 games played in parallel, all state held in batched tensors.

    Canonical view: channel 0 = discs of the player to move, channel 1 = opponent.
    """

    def __init__(self, n_games):
        self.n = n_games
        self.board = torch.zeros(n_games, 2, 6, 7)
        self.active = torch.ones(n_games, dtype=torch.bool)
        self.winner = torch.full((n_games,), -1, dtype=torch.long)  # 0 / 1, or -1 = draw
        self.game_plies = torch.zeros(n_games, dtype=torch.long)    # length of finished games
        self.channel0_player = torch.zeros(n_games, dtype=torch.long)
        self.ply = 0  # all games move in lockstep, so one global move counter is enough

    @property
    def current_player(self):
        """Whose turn it is (0 = the player who moved first)."""
        return self.ply % 2

    def legal_mask(self):
        """(N, 7) bool: which columns may be played right now."""
        return legal_move_mask(self.board)

    def step(self, actions):
        """Play one move in every active game (finished games ignore their action)."""
        idx = torch.where(self.active)[0]
        cols = actions[idx]

        # Gravity: the disc falls to the lowest empty cell of the chosen column.
        filled = self.board[idx].sum(dim=(1, 2))                    # (k, 7) discs per column
        rows = (5 - filled[torch.arange(len(idx)), cols]).long()
        assert (rows >= 0).all(), "illegal move: a chosen column is already full"
        self.board[idx, 0, rows, cols] = 1.0                        # channel 0 = player to move

        # Did this move end the game? (win for the mover, or a full board = draw)
        won = check_four_in_a_row(self.board[idx, 0])
        full = self.board[idx].sum(dim=(1, 2, 3)) == 42
        self.winner[idx[won]] = self.current_player
        finished = idx[won | full]
        self.game_plies[finished] = self.ply + 1
        self.active[finished] = False
        self.ply += 1

        # Flip channels in the games that continue: the next player becomes channel 0.
        still = torch.where(self.active)[0]
        self.board[still] = self.board[still][:, [1, 0]]
        self.channel0_player[still] = 1 - self.channel0_player[still]

    def z(self, player):
        """Final reward from `player`'s point of view: +1 win, -1 loss, 0 draw.

        `player` is 0/1, a scalar or a per-game tensor. This is where the per-player
        sign handling lives — sealed here so nobody trains on flipped rewards.
        """
        z = torch.zeros(self.n)
        z[self.winner == player] = 1.0
        z[self.winner == 1 - player] = -1.0
        return z

    def render(self, i=0):
        """Pretty-print board i."""
        render_board(self.board[i], int(self.channel0_player[i]))


def random_opponent(boards, legal_mask):
    """The fixed opponent: a uniformly random legal column for every board."""
    return torch.multinomial(legal_mask.float(), 1).squeeze(1)


def center_first_opponent(boards, legal_mask):
    """A simple heuristic opponent: always the center column when possible, else random."""
    actions = random_opponent(boards, legal_mask)
    actions[legal_mask[:, 3]] = 3
    return actions


def play_random_vs_random():
    """One random-vs-random game, printed ply by ply, with its reward sequence."""
    env = VectorConnect4(1)
    rewards_for_X = []
    while env.active[0]:
        mover = "X" if env.current_player == 0 else "O"
        action = random_opponent(env.board, env.legal_mask())
        env.step(action)
        rewards_for_X.append(float(env.z(0)[0]))  # 0 on every ply until the game ends
        print(f"ply {env.ply}: {mover} plays column {int(action[0])}")
        env.render(0)
        print()
    result = {1.0: "X wins", -1.0: "O wins", 0.0: "draw"}[rewards_for_X[-1]]
    print(f"{result}. Rewards seen by player X: {rewards_for_X}")


# %% [markdown]
# Watch one random-vs-random game. Look at the **reward sequence** printed at the end:
# `0, 0, ..., 0, ±1` — the same delayed-reward structure as the Fool's-mate chess example
# in the lecture. Only the final position reveals whether the early moves were any good.

# %%
set_seed(SEED)
play_random_vs_random()

# %% [markdown]
# ## 2. The policy $\pi_\theta(a \mid s)$: a network that looks at the board
#
# The policy is a small convolutional network. Input: the board $s$. Output: 7 numbers
# (**logits**), one per column. `softmax(logits)` turns them into probabilities — that is
# $\pi_\theta(\cdot \mid s)$, a **distribution over actions**, exactly like the bar plot
# on the policy slide.
#
# One Connect-4 detail: a full column must never be played. Before the softmax we set the
# logits of illegal columns to $-10^9$ (practically $-\infty$), so their probability
# becomes exactly 0.
#
# The convolutional part is given:

# %%
class ConvTrunk(nn.Module):
    """Three conv layers that turn a (2, 6, 7) board into 128 features."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 128), nn.ReLU(),
        )

    def forward(self, boards):
        return self.layers(boards)


# %% [markdown]
# ### 🔧 TODO 1: policy head + action masking
#
# **What you write:** the last layer of the policy (128 features → 7 logits, one per
# column) and the masking that gives illegal columns probability 0.
#
# **Hint:** lecture slide *"A policy is a distribution over actions"*. Use
# `nn.Linear(128, 7)` for the head. To mask, set the logits of illegal columns to a huge
# negative constant like `-1e9`, e.g. with `logits.masked_fill(~legal_mask, -1e9)`.
# Return the masked logits — the softmax happens later, when we need probabilities.

# %%
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


# %% [markdown]
# The bar plot below is the notebook version of the "action probabilities" figure from
# the policy slide — here for the empty board, under the untrained (random) policy.

# %%
set_seed(SEED)
demo_net = PolicyNet()
print(f"PolicyNet has {sum(p.numel() for p in demo_net.parameters()):,} parameters (theta)")

empty_board = torch.zeros(1, 2, 6, 7)
probs = torch.softmax(demo_net(empty_board, legal_move_mask(empty_board)), dim=1)[0].detach()
plt.figure(figsize=(5, 2.5))
plt.bar(range(7), probs)
plt.xlabel("column $a$")
plt.ylabel(r"$\pi_\theta(a \mid s_{\mathrm{empty}})$")
plt.title("Action probabilities on the empty board (untrained)")
plt.show()

# %%
set_seed(0)
_net1 = PolicyNet()
_boards = torch.zeros(2, 2, 6, 7)
_boards[0, 0, :, 0] = 1.0        # board 0: column 0 completely full
_boards[1, 0, 0::2, 3] = 1.0     # board 1: column 3 full, discs split between both players
_boards[1, 1, 1::2, 3] = 1.0
_mask = legal_move_mask(_boards)
assert not _mask[0, 0] and not _mask[1, 3], "environment bug? full columns must be illegal"

_logits = _net1(_boards, _mask)
assert _logits.shape == (2, 7), f"expected output shape (2, 7), got {tuple(_logits.shape)}"
_probs = torch.softmax(_logits, dim=1)
assert (_probs[~_mask] == 0).all(), "illegal columns must get probability exactly 0"
assert torch.allclose(_probs.sum(dim=1), torch.ones(2), atol=1e-5), "each row must sum to 1"
print("✅ TODO 1 looks correct")

# %% [markdown]
# ## 3. Acting: sample from the policy
#
# > From the lecture: "**During training: sample.** Sampling lets the agent explore."
#
# The function below plays a batch of games in parallel: our agent (the net) against an
# opponent function. In half of the games the agent moves first, in the other half it
# moves second. For every agent move it stores the board, the legal mask, the action, and
# $\log \pi_\theta(a \mid s)$ — Section 4 needs that log-probability for the gradient.
#
# It calls `sample_action(...)` — which does not exist yet. That is your TODO 2, right below.

# %%
Rollout = namedtuple("Rollout", "boards masks actions log_probs game_idx G z game_plies")


def play_games(net, n_games, opponent_fn, gamma, choose_action=None):
    """Play n_games (agent vs opponent_fn) and collect everything REINFORCE needs.

    choose_action: how the agent picks a move from the masked logits.
    Defaults to sample_action, i.e. a ~ pi_theta (Section 7b tries an alternative).
    """
    env = VectorConnect4(n_games)
    agent_player = torch.zeros(n_games, dtype=torch.long)
    agent_player[n_games // 2:] = 1                     # second half: the opponent starts
    moves_played = torch.zeros(n_games, dtype=torch.long)

    boards_h, masks_h, actions_h, log_probs_h, game_idx_h, move_k_h = [], [], [], [], [], []

    while env.active.any():
        actions = torch.zeros(n_games, dtype=torch.long)
        legal = env.legal_mask()
        agent_turn = env.active & (agent_player == env.current_player)
        opp_turn = env.active & (agent_player != env.current_player)

        if agent_turn.any():
            boards, masks = env.board[agent_turn], legal[agent_turn]
            masked_logits = net(boards, masks)
            act = choose_action if choose_action is not None else sample_action
            a, log_p = act(masked_logits)               # <- your TODO 2 (by default)
            actions[agent_turn] = a
            # bookkeeping for the gradient later:
            boards_h.append(boards)
            masks_h.append(masks)
            actions_h.append(a)
            log_probs_h.append(log_p)
            game_idx_h.append(torch.where(agent_turn)[0])
            move_k_h.append(moves_played[agent_turn].clone())
            moves_played[agent_turn] += 1

        if opp_turn.any():
            actions[opp_turn] = opponent_fn(env.board[opp_turn], legal[opp_turn])

        env.step(actions)

    z = env.z(agent_player)                             # +1 / 0 / -1, per game
    game_idx = torch.cat(game_idx_h)                    # which game each stored move is from
    move_k = torch.cat(move_k_h)                        # its index among that game's moves
    G = compute_returns(move_k, moves_played[game_idx], z[game_idx], gamma)
    return Rollout(torch.cat(boards_h), torch.cat(masks_h), torch.cat(actions_h),
                   torch.cat(log_probs_h), game_idx, G, z, env.game_plies.clone())


# %% [markdown]
# ### 🔧 TODO 2: sample an action
#
# **What you write:** two lines — build a `Categorical` distribution from the masked
# logits, sample one action per board, and return the actions together with their
# log-probabilities.
#
# **Hint:** lecture slide *"Stochastic policy: $a_t \sim \pi_\theta(\cdot \mid s_t)$"*.
# `Categorical(logits=...)` applies the softmax for you; `dist.sample()` and
# `dist.log_prob(actions)` do all the work.

# %%
# TODO-CELL
def sample_action(masked_logits):
    """a ~ pi_theta(.|s): sample a column per board. Returns (actions, log_probs)."""
    # TODO 2:
    #   1. dist = Categorical(logits=...)   (already imported)
    #   2. sample one action per board
    #   3. return the actions and dist.log_prob(actions)
    raise NotImplementedError("TODO 2: sample from the policy")


# %% [markdown]
# ### Returns: how good was each move?
#
# The only reward is the final $z$, so the return of the agent's move number $t$
# (out of $T$ moves it made in that game) is
#
# $$G_t = \gamma^{\,T-t} \cdot z .$$
#
# With $\gamma = 1$: every move of a **won** game gets $G = +1$, every move of a **lost**
# game gets $G = -1$. The final result is credited to *all* moves — delayed reward, just
# like the Fool's-mate example. (One agent step = our move plus the opponent's reply;
# the opponent is part of the environment, so we count the agent's own moves.)

# %%
def compute_returns(move_index, moves_in_game, z, gamma):
    """Return of each stored move: G = gamma^(moves played after it) * z."""
    moves_after = (moves_in_game - 1 - move_index).float()
    return (gamma ** moves_after) * z


# %%
FINGERPRINT_TODO2 = 270192  # pinned by the reference solution

set_seed(123)
_net2 = PolicyNet()
_ro = play_games(_net2, 8, random_opponent, gamma=1.0)

_m = len(_ro.actions)
assert _ro.log_probs.shape == (_m,) and _ro.G.shape == (_m,), "one log-prob and one G per stored move"
assert torch.isfinite(_ro.log_probs).all(), "log-probs must be finite — did you sample an illegal column?"
assert _ro.masks[torch.arange(_m), _ro.actions].all(), "a sampled action was ILLEGAL — sample from the masked logits"
_expected_lp = torch.log_softmax(_net2(_ro.boards, _ro.masks), dim=1)[torch.arange(_m), _ro.actions]
assert torch.allclose(_ro.log_probs, _expected_lp, atol=1e-5), "log_probs do not match log pi(a|s) of the sampled actions"

# Same seed -> same games. (The stored value was produced by the reference solution.)
_fingerprint = int(_ro.actions.sum()) * 1000 + int(_ro.game_plies.sum())
assert _fingerprint == FINGERPRINT_TODO2, (
    f"fingerprint {_fingerprint} != expected {FINGERPRINT_TODO2}. If every check above passed, "
    "your torch version may sample differently — ask an instructor before debugging further.")
print("✅ TODO 2 looks correct")

# %% [markdown]
# > 💡 **Where is the replay buffer from the lecture?**
# >
# > There is none — on purpose. We use each batch of games for exactly **one** gradient
# > step, then throw it away. REINFORCE is **on-policy**: the policy gradient is an
# > expectation over trajectories of the *current* $\pi_\theta$. Yesterday's games came
# > from yesterday's policy — they are not valid samples for today's gradient.
# >
# > On the lecture's four-way data map we sit at **online + on-policy**. DQN sits at
# > online + **off**-policy — that is why *it* is allowed to keep a replay buffer.
# > In Section 7a we will reuse batches anyway, and *measure* how off-policy we become.

# %% [markdown]
# ## 4. The REINFORCE loss
#
# The policy gradient from the lecture:
#
# $$\nabla_\theta J = \mathbb{E}\big[\, G_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \,\big]$$
#
# We want gradient **ascent** on $J$, and optimizers do **descent** — so we minimize
#
# $$\mathcal{L} = -\,\frac{1}{M}\sum_{t} \log \pi_\theta(a_t \mid s_t)\, G_t$$
#
# Intuition: moves from **won** games ($G > 0$) get their probability pushed **up**,
# moves from **lost** games get pushed **down**.
#
# ### 🔧 TODO 3: the loss
#
# **What you write:** one line.
#
# **Hint:** lecture slide *"REINFORCE"*. Mind the minus sign: with the sign flipped, your
# agent will diligently learn to **lose** — and the training curve will still look busy.
# The check below catches exactly that.

# %%
# TODO-CELL
def reinforce_loss(log_probs, G):
    """REINFORCE: loss = -mean( log pi_theta(a_t|s_t) * G_t )."""
    # TODO 3: one line. Remember the minus sign.
    raise NotImplementedError("TODO 3: the REINFORCE loss")


# %%
_lp = torch.tensor([-1.0, -0.5, -2.0, -0.25])   # log-probs of 4 moves
_G = torch.tensor([1.0, -1.0, 1.0, 0.0])        # their returns
_loss = reinforce_loss(_lp, _G)
assert _loss.shape == (), "the loss must be a scalar"
assert abs(float(_loss) - 0.625) < 1e-4, (
    f"loss = {float(_loss):.4f}, expected 0.625. "
    "Got -0.625? Your sign is flipped — that trains an agent that learns to LOSE.")
print("✅ TODO 3 looks correct")

# %% [markdown]
# ### 🎁 Bonus TODO (optional — skip on first pass): a baseline
#
# $G$ is noisy: with $\gamma = 1$, *every* move of a won game gets $+1$ — including the
# bad moves that almost threw the game away. Subtracting a constant $b$ from $G$ does not
# change the expected gradient (the lecture shows why), but it can shrink the variance a
# lot. The simplest choice: the batch mean, $G - \bar G$.
#
# This previews **actor-critic**: there, the baseline is a learned value network
# $V_\varphi$ ($\varphi$, red on the slides) while the policy stays $\pi_\theta$
# ($\theta$, green). Two networks, two jobs.
#
# **What you write:** return `G - G.mean()`, then set `USE_BASELINE = True`.

# %%
# TODO-CELL
USE_BASELINE = False  # set to True once you implement advantage() below


def advantage(G):
    """Bonus: G minus the batch-mean baseline."""
    raise NotImplementedError("Bonus TODO: return G minus its mean")


# %%
if USE_BASELINE:
    _Gb = torch.tensor([1.0, 1.0, -1.0, 0.0])
    _adv = advantage(_Gb)
    assert torch.allclose(_adv, _Gb - 0.25, atol=1e-6), "advantage must be G - mean(G)"
    assert abs(float(_adv.mean())) < 1e-6, "after subtracting the mean, the mean is 0"
    print("✅ Bonus baseline looks correct")
else:
    print("Bonus not implemented — skipping (that is fine).")

# %% [markdown]
# ## 5. Training (given — just run it)
#
# The loop: play one batch of games with the current $\pi_\theta$ → **one** gradient step
# → throw the batch away → repeat. Generate, learn, discard: **online, on-policy**.
#
# The opponent stays **fixed** (uniformly random over legal columns). A fixed opponent is
# simply part of the environment, so the MDP stays **stationary** and learning is stable
# within our time budget. Self-play would change $P$ under our feet — we taste it in
# Section 7c.
#
# Watch the live plot: the win rate vs random should climb from ~50% to about 90%.
# The run takes a few minutes on the free CPU runtime.

# %%
N_GAMES = 256                      # games per batch
N_UPDATES = 30 if FAST else 500    # gradient steps
LEARNING_RATE = 5e-4
GAMMA = 1.0
PLOT_EVERY = 10


def train(net, opponent_fn, n_updates, n_games=N_GAMES, lr=LEARNING_RATE,
          gamma=GAMMA, use_baseline=False, show_plot=True):
    """REINFORCE training loop. Returns per-update history for plotting."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {"win_rate": [], "game_len": [], "loss": []}
    for update in range(n_updates):
        batch = play_games(net, n_games, opponent_fn, gamma)   # fresh on-policy data

        G = advantage(batch.G) if use_baseline else batch.G    # bonus TODO
        loss = reinforce_loss(batch.log_probs, G)              # your TODO 3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                                        # ...and the batch is discarded

        history["win_rate"].append((batch.z > 0).float().mean().item())
        history["game_len"].append(batch.game_plies.float().mean().item())
        history["loss"].append(loss.item())
        if show_plot and (update + 1) % PLOT_EVERY == 0:
            update_live_plot(history)
    return history


# %%
set_seed(SEED)
net = PolicyNet()
t0 = time.time()
history = train(net, random_opponent, N_UPDATES, use_baseline=USE_BASELINE)
print(f"Trained {N_UPDATES} updates in {time.time() - t0:.0f} s. "
      f"Win rate over the last 10 updates: {np.mean(history['win_rate'][-10:]):.2f}")

# %%
plot_curves({"per batch": history["win_rate"], "smoothed": smoothed(history["win_rate"])},
            ylabel="win rate vs random", target=0.9, title="REINFORCE vs the random opponent")

# %% [markdown]
# ## 6. Play against your agent
#
# The lecture asked: *the policy is a distribution — which action do we take at
# inference time?* Answer: during training we **sample** (exploration); at inference we
# take the **mode**: $\arg\max_a \pi_\theta(a \mid s)$.
#
# ### 🔧 TODO 4: greedy action
#
# **What you write:** run the net, take the argmax over the **masked** logits.
#
# **Hint:** lecture slide *"Sampling vs argmax"*. The argmax of the masked logits equals
# the argmax of the probabilities — softmax never changes which entry is largest, so you
# do not need it. Masking matters: an argmax over raw logits could pick a full column and
# crash your game demo.

# %%
# TODO-CELL
def greedy_action(net, boards):
    """argmax_a pi_theta(a|s): the most likely legal column, one per board."""
    legal = legal_move_mask(boards)   # given: (N, 7) bool
    # TODO 4: get the masked logits from the net, return the argmax column per board.
    raise NotImplementedError("TODO 4: the greedy action")


# %%
set_seed(0)
_net4 = PolicyNet()
_env4 = VectorConnect4(3)
_a = greedy_action(_net4, _env4.board)
assert _a.shape == (3,), "one column index per board"
assert torch.equal(_a, _net4(_env4.board, _env4.legal_mask()).argmax(dim=1)), \
    "greedy_action must return the argmax of the masked logits (the mode of the policy)"

# Trap: fill the column the net likes best — the returned action must still be legal.
_best = int(_a[0])
_trap = torch.zeros(1, 2, 6, 7)
_trap[0, 0, 0::2, _best] = 1.0
_trap[0, 1, 1::2, _best] = 1.0
_a_trap = int(greedy_action(_net4, _trap)[0])
assert _a_trap != _best and legal_move_mask(_trap)[0, _a_trap], \
    "your greedy action picked a FULL column — apply the legal mask before the argmax"
print("✅ TODO 4 looks correct")

# %% [markdown]
# Your move. You are X and you start; type a column number, `q` quits.

# %%
@torch.no_grad()
def play_vs_agent(net, human_first=True):
    """Play one game against the greedy agent, in the notebook."""
    env = VectorConnect4(1)
    human = 0 if human_first else 1
    print(f"You are {'X' if human == 0 else 'O'}. The agent plays argmax pi(a|s).\n")
    env.render()
    while env.active[0]:
        if env.current_player == human:
            col = read_human_move(env.legal_mask()[0])
            if col is None:
                print("Game aborted.")
                return
        else:
            col = int(greedy_action(net, env.board)[0])
            print(f"\nAgent plays column {col}")
        env.step(torch.tensor([col]))
        print()
        env.render()
    print({1.0: "\nYou win! 🎉", -1.0: "\nThe agent wins.", 0.0: "\nDraw."}[float(env.z(human)[0])])


if HEADLESS:
    print("(interactive cell — skipped in headless runs)")
else:
    play_vs_agent(net)

# %% [markdown]
# How good is it, in numbers? Greedy agent, 1000 games, half as first player:

# %%
@torch.no_grad()
def evaluate(net, opponent_fn, n_games=1000):
    """Win/draw/loss rates of the greedy agent against opponent_fn."""
    env = VectorConnect4(n_games)
    agent_player = torch.zeros(n_games, dtype=torch.long)
    agent_player[n_games // 2:] = 1
    while env.active.any():
        actions = torch.zeros(n_games, dtype=torch.long)
        legal = env.legal_mask()
        agent_turn = env.active & (agent_player == env.current_player)
        opp_turn = env.active & (agent_player != env.current_player)
        if agent_turn.any():
            actions[agent_turn] = greedy_action(net, env.board[agent_turn])
        if opp_turn.any():
            actions[opp_turn] = opponent_fn(env.board[opp_turn], legal[opp_turn])
        env.step(actions)
    z = env.z(agent_player)
    return {"win": (z > 0).float().mean().item(),
            "draw": (z == 0).float().mean().item(),
            "loss": (z < 0).float().mean().item()}


set_seed(1)
for name, opponent in [("random", random_opponent), ("center-first", center_first_opponent)]:
    stats = evaluate(net, opponent)
    print(f"vs {name:13s} win {stats['win']:5.1%}   draw {stats['draw']:5.1%}   loss {stats['loss']:5.1%}")

# %% [markdown]
# ### 🔍 Find your agent's blind spot
#
# Play again, and this time set a trap: build an **open three in a row** and watch the
# agent fail to block it. Why does this happen? The network has **no lookahead** — it
# only maps the current board to whatever move worked during training. Threats were rare
# in its training games, and blocking one almost never decided the final $z$, so the
# delayed ±1 reward barely taught it to defend. (We analysed one trained agent: more
# than half of its rare losses were single threats it could have blocked with one move.)
#
# A hand-written rule "win if you can, block if you must" beats random ~98% of the time
# — your ~90% net never found that rule, because after a few hundred updates the policy
# is almost deterministic and stops exploring. Remember this feeling: it is exactly the
# gap that **search** fills. MCTS and AlphaZero (final part of the lecture) combine a
# policy net like yours with explicit lookahead.

# %% [markdown]
# ## 7. Bonus experiments (given — run whatever fits into the remaining time)
#
# ### 7a. Reuse each batch K times — how off-policy do we get?
#
# DQN keeps a replay buffer, so why can't we? Let's try. The function below is our
# training loop with one change: each batch is used for `reuse_k` gradient steps before
# being thrown away. For $k > 1$ the log-probabilities are recomputed under the *current*
# $\theta$ — but the games were played by an *older* policy. The data is **stale**: we
# are off-policy, without any correction.
#
# "How off-policy" is a measurable number: the **importance ratio**
# $\pi_{\text{now}}(a \mid s)\,/\,\pi_{\text{data}}(a \mid s)$ from the importance-sampling
# slide. For fresh on-policy data the ratio is exactly 1 — and the REINFORCE gradient
# silently *assumes* it is 1. The code below tracks how many samples drift out of
# $[0.8,\ 1.2]$, the trust region that **PPO** clips to.

# %%
N_BONUS_UPDATES = 8 if FAST else 80


def train_with_reuse(reuse_k, n_updates, n_games=N_GAMES, lr=1e-3, seed=SEED):
    """The same loop as train(), but each batch is used for reuse_k gradient steps.

    Also measures the staleness of the data: at every reuse step k we recompute the
    importance ratio pi_now(a|s) / pi_data(a|s) of the batch's own moves and count
    how many fall outside PPO's clip range [0.8, 1.2].
    """
    set_seed(seed)
    net = PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    win_rates = []
    frac_outside_clip = torch.zeros(reuse_k)
    for update in range(n_updates):
        batch = play_games(net, n_games, random_opponent, gamma=1.0)
        log_probs_data = batch.log_probs.detach()    # log pi of the policy that PLAYED
        for k in range(reuse_k):
            if k == 0:
                log_probs = batch.log_probs          # fresh: theta == the policy that played
            else:
                # stale reuse: theta has changed, the games have not
                masked_logits = net(batch.boards, batch.masks)
                log_probs = Categorical(logits=masked_logits).log_prob(batch.actions)
            ratio = (log_probs.detach() - log_probs_data).exp()
            frac_outside_clip[k] += ((ratio - 1).abs() > 0.2).float().mean() / n_updates
            loss = reinforce_loss(log_probs, batch.G)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        win_rates.append((batch.z > 0).float().mean().item())
    return win_rates, frac_outside_clip


curve_k1, _ = train_with_reuse(reuse_k=1, n_updates=N_BONUS_UPDATES)
print(f"K=1 done, final win rate {np.mean(curve_k1[-10:]):.2f}")
curve_k5, drift_k5 = train_with_reuse(reuse_k=5, n_updates=N_BONUS_UPDATES)
print(f"K=5 done, final win rate {np.mean(curve_k5[-10:]):.2f}")

plot_curves({"K = 1 (on-policy)": smoothed(curve_k1), "K = 5 (stale reuse)": smoothed(curve_k5)},
            ylabel="win rate vs random", title="Reusing each batch K times")
plt.figure(figsize=(5, 2.5))
plt.bar(range(1, len(drift_k5) + 1), 100 * drift_k5)
plt.xlabel("gradient step on the same batch")
plt.ylabel("% outside PPO clip")
plt.title("Moves whose importance ratio left [0.8, 1.2]")
plt.show()

# %% [markdown]
# Two things to notice:
#
# 1. **The data goes stale immediately.** Fresh data sits at ratio = 1 by definition
#    (step 1 in the bar plot). One gradient step later, a noticeable share of the batch
#    is already outside the trust region that PPO refuses to step beyond — and REINFORCE
#    still weights every sample as if the ratio were 1. The gradient is **biased**.
# 2. **And yet the win-rate curve is not worse — it can even look better** (K gradient
#    steps per batch of games). That is the scary part: off-policy bias is a *silent*
#    bug. Connect 4 against a random opponent is so easy that a biased gradient still
#    points roughly uphill. Against a stronger opponent, or in a harder task, exactly
#    this bias makes training collapse — with no warning, just like here.
#
# The honest fix is to multiply every sample by $\pi_{\text{now}}/\pi_{\text{data}}$
# (importance sampling, from the lecture) — and to *clip* that ratio near 1 so single
# stale samples cannot explode the update. That clipped ratio **is the heart of PPO**.

# %% [markdown]
# ### 7b. What if we act greedily *during training*?
#
# The lecture insists: during training, **sample** — $a_t \sim \pi_\theta(\cdot \mid s_t)$.
# Would argmax not be better? It always plays the currently-best move, after all.
# Let's find out. The **only** change below is the action rule used in the training games.

# %%
def argmax_in_training(masked_logits):
    """The alternative rule: always the current best move — no exploration."""
    dist = Categorical(logits=masked_logits)
    actions = masked_logits.argmax(dim=1)
    return actions, dist.log_prob(actions)


exploration_curves = {}
for name, rule in [("sample (default)", None), ("argmax (no exploration)", argmax_in_training)]:
    set_seed(SEED)
    net_x = PolicyNet()
    optimizer = torch.optim.Adam(net_x.parameters(), lr=LEARNING_RATE)
    exploration_curves[name] = []
    for update in range(N_BONUS_UPDATES):
        batch = play_games(net_x, N_GAMES, random_opponent, GAMMA, choose_action=rule)
        G = advantage(batch.G) if USE_BASELINE else batch.G
        loss = reinforce_loss(batch.log_probs, G)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        exploration_curves[name].append((batch.z > 0).float().mean().item())
    print(f"{name}: final win rate {np.mean(exploration_curves[name][-10:]):.2f}")

plot_curves({k: smoothed(v) for k, v in exploration_curves.items()},
            ylabel="win rate vs random", title="Sampling vs argmax during training")

# %% [markdown]
# The greedy learner looks **stronger at first** — it exploits what it already knows.
# Then it flatlines: it plays the same lines forever, discovers nothing new, and the
# sampling learner overtakes it for good. That is the exploration–exploitation trade-off
# in one picture: sampling pays a small price early and buys learning.
#
# (At *inference* time exploration buys nothing — that is why your `greedy_action` from
# TODO 4 is the right rule for playing, and the wrong rule for learning.)

# %% [markdown]
# ### 7c. A taste of self-play
#
# Random is a weak teacher. Next natural step: play against a **frozen copy** of the
# trained agent. Frozen, because a live self-play opponent would change the transition
# function $P$ during training — a non-stationary MDP, one of the classic self-play
# headaches (AlphaZero-style training has to manage exactly this).

# %%
net_selfplay = PolicyNet()
net_selfplay.load_state_dict(net.state_dict())
frozen = PolicyNet()
frozen.load_state_dict(net.state_dict())


def frozen_opponent(boards, legal_mask):
    """The frozen copy plays by sampling from its own policy."""
    with torch.no_grad():
        actions, _ = sample_action(frozen(boards, legal_mask))
    return actions


set_seed(SEED)
history_selfplay = train(net_selfplay, frozen_opponent, N_BONUS_UPDATES,
                         use_baseline=USE_BASELINE, show_plot=False)
print(f"Win rate vs the frozen copy during training: "
      f"{np.mean(history_selfplay['win_rate'][:10]):.2f} (first 10 updates) -> "
      f"{np.mean(history_selfplay['win_rate'][-10:]):.2f} (last 10)")
print("After self-play training:")
for name, opponent in [("random", random_opponent), ("frozen old self", frozen_opponent)]:
    stats = evaluate(net_selfplay, opponent)
    print(f"vs {name:16s} win {stats['win']:5.1%}   draw {stats['draw']:5.1%}   loss {stats['loss']:5.1%}")

# %% [markdown]
# Surprised? The win rate against the frozen copy stays pinned at **50% — it learns
# nothing**. The reason is Section 7b in disguise: after the main training our policy is
# almost deterministic. Two near-deterministic equals replay near-identical games, no new
# situations appear, and REINFORCE has nothing to learn from. Self-play only works when
# exploration is kept **alive** — AlphaZero, for example, injects extra noise into its
# move selection for exactly this reason.

# %% [markdown]
# ### 7d. Take-home: keep going
#
# - **Kaggle ConnectX** (https://www.kaggle.com/competitions/connectx) — a running
#   Connect-4 competition; your agent from today is a valid starting point.
# - Ideas to try there: a learned baseline $V_\varphi$ (actor-critic), PPO's clipped
#   importance ratio (Section 7a showed why), opponent pools for self-play, and finally
#   lookahead search (MCTS) on top of your policy.
#
# Thanks for playing — and remember what the loss curve taught you today:
# *sample during training, argmax during the exam.*
