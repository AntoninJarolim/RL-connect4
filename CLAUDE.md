# Connect 4 REINFORCE — BISSIT teaching notebook

## What this project is

A Google Colab teaching notebook for a hands-on RL workshop at the BISSIT summer school
(FIT VUT Brno). Audience: international BSc/MSc students in IT, **seeing RL for the first
time**, mixed backgrounds, English as a second language. They have **15 min of guided
intro + ~60 min of work**. The notebook trains a pure **online REINFORCE** agent to play
Connect 4 against a fixed random opponent.

The notebook accompanies a lecture whose notation and story it must match exactly
(see "Notation contract" below). The lecture covers: MDP, trajectories, delayed reward
(Fool's mate example), stochastic vs deterministic policies, sampling vs argmax,
online/offline, on-policy/off-policy, replay buffers, policy gradients.

## Hard constraints — never violate these

1. **Zero pip installs.** Imports allowed: `torch`, `numpy`, `matplotlib`, stdlib,
   `IPython.display`. All preinstalled in Colab. If you are tempted to install anything,
   stop and redesign.
2. **CPU-safe.** Everything must work on a free-tier CPU-only Colab runtime. Full
   training run: **≤ 5 minutes on CPU**, faster on T4. GPU is a bonus, never a requirement.
3. **Pure REINFORCE, on-policy, online.** Generate a batch of games with the current
   policy → one gradient step → discard the batch. No replay buffer in the training path
   (batch reuse appears only as a deliberate *measurement experiment*, see Section 7a).
4. **Train against a FIXED random opponent** in the main path. Self-play only as a bonus
   experiment with a frozen opponent copy. Reason: stationary MDP, stable learning in the
   session's time budget.
5. **Exactly 4 mandatory student TODOs + 1 optional bonus.** Nothing else is left to
   students. Every TODO is immediately followed by an assert cell.
6. **No ipywidgets.** Interactive play uses `input()` in a loop. No gymnasium, no
   external envs, no downloads at runtime.

## Repo layout

```
.
├── CLAUDE.md                      # this file
├── README.md                      # with "Open in Colab" badge for the SKELETON notebook
├── jupytext.toml                  # pairing config (percent format)
├── connect4_reinforce.py          # SKELETON, jupytext percent format — source of truth
├── connect4_reinforce.ipynb       # generated, committed (Colab loads this)
├── solutions/
│   ├── connect4_reinforce_solutions.py
│   └── connect4_reinforce_solutions.ipynb
├── tests/
│   └── test_env.py                # pytest, runs against code extracted from the notebook
└── tools/
    ├── check_todo_delta.py        # verifies skeleton vs solutions differ ONLY in TODO cells
    └── make_skeleton.py           # regenerates the skeleton from the solutions file
```

## Workflow rules

- **Never edit `.ipynb` by hand.** Edit the `.py` jupytext files, then regenerate:
  `jupytext --to ipynb connect4_reinforce.py` (same for solutions). Commit both.
- **Never edit the skeleton by hand either.** Edit the SOLUTIONS `.py`, then run
  `python tools/make_skeleton.py` — it copies the file and swaps the 5 TODO cell bodies
  for the NotImplementedError versions (the skeleton bodies live inside that script).
- Use percent format: `# %%` for code cells, `# %% [markdown]` for markdown cells.
- Local dev on this machine: use `.venv/bin/python` (gitignored venv; system python has
  no torch). Always set `MPLBACKEND=Agg` when running the `.py` files as scripts —
  otherwise `plt.show()` blocks on a GUI backend.
- After every meaningful change:
  1. `pytest tests/ -q`
  2. Execute the **solutions** notebook headlessly end-to-end:
     `HEADLESS=1 jupyter nbconvert --to notebook --execute solutions/connect4_reinforce_solutions.ipynb --output /tmp/sol_out.ipynb --ExecutePreprocessor.timeout=600`
     It must complete with zero errors. Use `FAST=1` for CI-speed runs, but run the full
     budget at least once per milestone.
  3. Execute the **skeleton** notebook headlessly; it must run cleanly *up to* TODO 1 and
     fail there with `NotImplementedError` (TODO cells contain
     `raise NotImplementedError("TODO 1: ...")` placeholders). Any failure before the
     first TODO is a bug.
  4. `python tools/check_todo_delta.py` — skeleton and solutions must be identical except
     inside cells tagged `# TODO-CELL`.
- Interactive `input()` cell and live plotting are guarded by the `HEADLESS=1` env var,
  plus a try/except around `input()` so nbconvert without the env var still passes.

## Notation contract (must match the lecture slides)

Include this table as a markdown cell in Section 0 and use these names consistently:

| Notebook | Slides | Meaning |
|---|---|---|
| `obs`, `boards` | o_t = s_t | fully observed state (say so explicitly) |
| `logits` → softmax | π_θ(·\|s) | policy distribution over 7 columns |
| `a = dist.sample()` | a_t ∼ π_θ(·\|s_t) | training-time action selection |
| `greedy_action(...)` | argmax_a π_θ(a\|s) | inference-time action selection |
| `G` | G(τ_t:) = Σ γ^k r | discounted return; terminal-only reward z ∈ {−1,0,+1} |
| `gamma` | γ | 1.0 everywhere in the notebook |
| `theta` (policy params) | θ | (no color coding — the slides dropped it) |
| (bonus) baseline | previews V_φ | |

Rewards are indexed r_{t+1} in the slides; keep markdown consistent with that.

## Environment spec (given code, students only read it)

- Class `VectorConnect4`: `N` parallel games as tensors. Board `(N, 2, 6, 7)` float,
  **canonical representation: channel 0 = current player to move, channel 1 = opponent**.
  Column-major gravity (piece falls to lowest empty row).
- `legal_mask()` → `(N, 7)` bool: column legal iff top cell (row 0) empty. This is the
  ONLY legality rule in Connect 4 — a full column.
- Win detection: vectorized conv check of 4-in-a-row horizontal / vertical / both
  diagonals (`check_four_in_a_row`, black-box helper).
- Full board with no winner → draw, `z = 0`.
- `step(actions)` applies moves for active games only; finished games are frozen and
  masked out of subsequent stepping. All-columns-full never reaches the sampling code.
- Reward: 0 every ply, terminal z = +1 win / −1 loss / 0 draw **from each player's own
  perspective** via `env.z(player)`. The per-player sign handling lives HERE, sealed in
  given code — it is the classic silent self-play bug and must never be a student task.
- `render(i)` pretty-prints board i with column indices 0–6.
- A `play_random_vs_random()` demo helper.

## Final training settings (measured 2026-08-19, torch 2.13 CPU)

- **Main path: `N_GAMES=256`, `N_UPDATES=500`, Adam `lr=5e-4`, `gamma=1.0`.**
  Solutions set `USE_BASELINE=True` (bonus implemented); skeleton default is False.
- Wall time: ~45 s for the main training locally (16-core dev box); estimated 2–4 min on
  free Colab CPU. Full solutions notebook incl. bonus sections: ~2 min locally.
- **Measured greedy win rate vs random (4000 games, half as first player):**
  0.895–0.904 across seeds with baseline, 0.87–0.90 without. **~0.90 is a hard
  empirical ceiling** for pure REINFORCE here — extensively verified: lr, updates,
  batch size, γ, weight decay, net capacity, return normalization, lr decay, and
  randomized openings all converge to the same per-seed basin (greedy policies of
  different configs agree on 100% of visited states). Root cause: policy entropy
  collapses to ~0 by update ~20, so the agent stops exploring before it learns to
  block threats. Forensics: of ~10% lost games, >half are single blockable threats,
  and the agent blocked none of them; a hand-coded 1-ply win/block heuristic wins 98%.
  The notebook therefore promises "about 90%", and the blind-spot section teaches the
  missing-blocking / no-lookahead gap explicitly (bridge to MCTS/AlphaZero).
- Do NOT chase >90% with entropy bonuses or reward shaping — that would break the
  "pure REINFORCE" constraint. The gap IS the pedagogy.
- If a fresh Colab torch version lands the seeded run visibly below 0.88, bump `SEED`
  or `N_UPDATES` and re-check.

## Notebook structure — section by section

### Section 0 — Setup (given, ~3 min)
- Markdown: "File → Save a copy in Drive" warning, first cell.
- Imports, seed setting, device detection (informational only — everything runs on CPU
  so seeded runs stay reproducible).
- Notation table (above). One sentence: "Connect 4 is fully observed, o_t = s_t, so we
  write s everywhere."
- `FAST` / `HEADLESS` env-var flags for CI.

### Section 1 — Environment (given, run-only, ~5 min)
- 🔒 black-box helpers cell, then `VectorConnect4` (students read, don't write).
- Demo cell: one random-vs-random game rendered ply by ply.
- Markdown: "this class IS P and R from the lecture; note the reward sequence
  0, 0, …, ±1 — same delayed-reward structure as the Fool's mate chess example."

### Section 2 — TODO 1: policy head + action masking (~10 min)
- Given: `ConvTrunk`, 3 conv layers (32 ch) + Linear to 128 features, ~192k params total.
- **TODO 1**: final `nn.Linear(128, 7)` + mask illegal logits to −1e9 before softmax.
  The hint deliberately avoids copy-pasteable code (says
  `nn.Linear(number_of_features, number_of_columns)` and points at `masked_fill` by name).
- Given after: `plot_action_probs` bar charts for (a) the empty board and (b) the crafted
  `winning_board` (three own discs stacked in column 0, win-in-one) — both under the
  untrained policy. The winning board returns in Section 6.
- **Assert cell**: shape `(N, 7)`; illegal-column probs exactly 0 on crafted near-full
  boards; rows sum to 1 (atol 1e-5).

### Section 3 — TODO 2: acting by sampling (~10–15 min)
- Given: `play_games` batched rollout (env stepping, per-player bookkeeping, storage;
  optional `choose_action` hook used by experiment 7b).
- **TODO 2** (two lines): `Categorical(logits=...)`, `.sample()`, `.log_prob(...)`.
- Given after: `compute_returns`: G = γ^(moves after) · z. **Given, not a TODO.**
- **Assert cell**: shapes; finite log-probs; sampled actions legal; log-probs match an
  independent `log_softmax` computation; fixed-seed fingerprint **307207** (regenerate
  together with tests/test_env.py if torch RNG or the rollout loop ever changes).
- Markdown box: "Where's the replay buffer?" (online + on-policy; DQN contrast;
  foreshadows 7a).

### Section 4 — TODO 3: the REINFORCE loss (~10 min)
- **TODO 3** (one line): `loss = -(log_probs * G).mean()`.
- **Assert cell**: fixed 4-element batch pins the scalar **0.625** (a flipped sign gives
  −0.625 and a customized error message). Non-negotiable.
- **Bonus TODO (optional)**: `advantage(G) = G - G.mean()` + `USE_BASELINE` flag.
  Markdown previews V_φ / actor-critic with the φ (red) vs θ (green) convention.

### Section 5 — Training loop (given, run-only)
- `train()`: play batch → loss → one optimizer step → discard; live win-rate plot
  (guarded for headless). Fixed-opponent stationarity sentence per spec.
- Settings: see "Final training settings" above. Curve reaches ~0.90 batch win rate.

### Section 6 — TODO 4 + payoff (~10 min)
- **TODO 4**: `greedy_action(net, boards)` — argmax over MASKED logits (mask computed
  inside via `legal_move_mask`; no softmax needed — say why).
- **Assert cell**: equals independent argmax of masked logits; a board whose
  globally-best column is FULL still yields a legal action.
- Given: two-board "what did it learn?" demo: `find_win_in_one_position` fishes a
  win-in-one board out of the agent's OWN games (there the policy is near-certain about
  the winning move — by construction) and contrasts it with the Section 2
  `winning_board`, where the trained policy is BLIND (measured: it puts ~1.0 on its
  center habit, ~0.0 on the winning column). Do not claim the trained policy recognizes
  wins on arbitrary boards — it demonstrably does not; it learned a habit, not the rule.
- Given: play-vs-agent `input()` cell (human = X, moves first; guarded for headless).
- Given: `evaluate()` win/draw/loss vs (a) random, (b) center-first heuristic.
- Markdown: blind-spot prompt (open three) + measured facts: >half of losses are
  unblocked single threats; 1-ply heuristic gets 98%; bridge to MCTS/AlphaZero.

### Section 7 — Bonus experiments (given, run-only, overflow / take-home)
- **7a — Batch reuse: "how off-policy do we get?"** `train_with_reuse(reuse_k)` takes K
  gradient steps per batch and measures the importance ratio π_now/π_data of the batch's
  own moves at each reuse step, counting samples outside PPO's clip range [0.8, 1.2].
  Plots: win-rate K=1 vs K=5 (they overlap — deliberately!) + bar chart of % outside the
  clip range per reuse step. Message: staleness is real and measurable, the bias is
  SILENT (the curve doesn't warn you here because random is too weak to punish it), and
  the clipped importance ratio is the heart of PPO.
  **Design note:** the originally planned "watch reuse destabilize the curve" is
  EMPIRICALLY FALSE in this env — even 300 gradient steps on one never-refreshed batch
  reach ~0.87 vs random (verified at K∈{5,8,10,20,25,∞}, lr up to 1e-2, batches 32–256).
  Do not re-introduce that claim.
- **7b — Exploration: sampling vs argmax during training.** Same loop, action rule
  swapped via the `choose_action` hook. Argmax jumps to ~0.80 then flatlines; sampling
  overtakes and reaches ~0.90 — exploration/exploitation in one picture; ties to the
  lecture's "During training: sample" alertblock and to TODO 4.
  **Design note:** this replaced the planned γ=0.95 "win fast" experiment: γ has NO
  measurable effect here (mean game length ~10.7 plies for γ∈{0.25..1.0}; the agent
  already wins near-fastest at γ=1, and Adam absorbs return scaling). Verified; do not
  re-introduce a γ experiment without new evidence.
- **7c — Self-play taste.** Frozen copy of the trained net as opponent (it samples, so
  it doesn't repeat one game); continue training a copy; evaluate vs random and vs the
  frozen self. Markdown explains frozen = stationary P. **Measured outcome: the win rate
  vs the frozen copy stays pinned at 50% (verified up to 320 updates)** — the collapsed
  policy cannot explore, so self-play from it learns nothing. The section teaches
  exactly that (link back to 7b; AlphaZero's root noise). Do not promise self-play
  improvement in the markdown.
- **7d — Take-home pointer.** Kaggle ConnectX + ideas (V_φ, PPO, opponent pools, MCTS).

## Testing requirements (tests/test_env.py)

- Win detection: all four directions, edges/corners, and negative cases.
- Legality mask correctness incl. full column; draw detection replays a pinned 42-move
  drawn game; gravity stacking; canonical channel flip between plies.
- Returns: hand-computed G, both players, gamma 1.0 and 0.9; rollout consistency.
- Reference loss value 0.625 for the frozen assert batch.
- Smoke training: the FAST notebook run (30 updates, batch 256) must improve win rate
  by > 0.1; trained FAST agent beats random > 0.75 greedy.
- Determinism: same seed → identical rollouts; fingerprint 270192.
- Skeleton: compiles, executes cleanly up to TODO 1, fails there with NotImplementedError.
- Tests exec the solutions .py with FAST=1 HEADLESS=1 (session fixture), so the
  notebook's own ✅ assert cells run inside pytest too.

## Readability first — students will READ this code

Optimize every student-visible cell for readability over cleverness, compactness, or
performance style points. Concretely:

- **Pedagogical code stays visible and simple.** Anything that carries an RL concept
  from the lecture (policy forward pass, masking, sampling, log-probs, returns, the
  loss, the train loop skeleton, greedy action) must be written as plainly as possible:
  descriptive variable names matching the notation table, no one-liner tricks, no
  nested comprehensions, no clever broadcasting where an obvious version exists.
- **Boring or complex-but-unimportant code becomes black-box helpers.** If code is
  (a) not important for the lecture, or (b) too complex to read in seconds — extract it
  into a clearly separated helpers cell that students may treat as a black box.
  Candidates: win-detection convolutions/shifts, board rendering, live-plot plumbing,
  headless guards, seeding boilerplate, the fingerprint/assert machinery.
  Apply this ONLY in those two cases — do not hide anything a student should learn from.
- **Name black boxes so the signature alone tells the story**:
  `check_four_in_a_row(boards) -> winners`, `render_board(board)`,
  `update_live_plot(win_rates)`, `legal_move_mask(boards)`. A student must be able to
  read the main flow using only these names, never opening the helper.
- Mark the helpers cell with a markdown header:
  `### 🔒 Helper functions (black box — you do not need to read this cell)`,
  and one line per helper saying what it does. Keep all such helpers in one or two
  cells near the top, not scattered.
- Each helper gets a one-line docstring stating what it returns; no further inline
  documentation burden inside black boxes.
- Rule of thumb: a student skimming ONLY the non-helper cells should see a short,
  linear story that maps 1:1 to the lecture: env loop → policy → sample → returns →
  loss → train → play.

## Style rules for the notebook

- Markdown in simple English (non-native audience): short sentences, no idioms.
- Every TODO cell: header `### 🔧 TODO n: <name>`, a "what you write" line, a "hint"
  line referencing the exact lecture slide, and `raise NotImplementedError` placeholder
  in the skeleton.
- Every assert cell prints `"✅ TODO n looks correct"` on success.
- Keep total student-visible code small enough to read end-to-end (~250–350 lines).
- Comments explain WHY (the RL concept), not WHAT (the Python).

## Definition of done

- [x] Solutions notebook executes headlessly end-to-end, zero errors, ≤ 5 min CPU full run.
- [x] Skeleton executes cleanly up to TODO 1, fails there with NotImplementedError.
- [x] `check_todo_delta.py` passes: skeleton vs solutions differ only in TODO cells.
- [x] All pytest tests green.
- [x] Trained agent ≈ 90 % vs random (measured 0.895–0.904 greedy with baseline; the
      ~0.90 ceiling is inherent to pure REINFORCE here, see "Final training settings");
      loses visibly to a human exploiting no-lookahead.
- [x] README has working "Open in Colab" badge pointing at the skeleton on `main`.
- [ ] Manual Colab check (human does this): fresh runtime, Run all on solutions; play
      cell works; live plot renders; timing acceptable on CPU runtime; win rate lands
      near 0.90 (if visibly below 0.88, bump SEED/N_UPDATES and re-pin the TODO 2
      fingerprint if torch RNG differs).
