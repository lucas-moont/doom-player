# Roadmap

Goal ladder: Clear `E1M1` → Beat Doom Episode 1 → Beat the Original Doom Episodes (`E1` to `E3`, the 1993 game).

The owner's WAD is The Ultimate Doom, which adds a fourth Doom Episode (`E4`, 1995). It is the Bonus Doom Episode: attempted after the goal ladder is complete, and reported separately.

The order is deliberate: the LLM-only Contender comes before reinforcement learning so the first public result arrives early and frames the question the rest of the project answers. See `adr/0005-showcase-order.md`.

## Milestones

| # | Name | Status | Concepts learned | Result to show | Brief |
|---|---|---|---|---|---|
| M0 | Workshop | done 2026-10-01 | WSL2, `uv`, Gymnasium API, ViZDoom, W&B | Video of a random agent on `E1M1` | `milestones/M0-workshop.md` |
| M1 | Ruler and socket | done 2026-10-02 | Evals, MCP, tool design | Eval Suite + Doom MCP server; random agent on the Scoreboard | `milestones/M1-ruler-and-socket.md` |
| M2 | LLM plays E1M1 | done 2026-10-02 | Harness, memory, tool use, cost accounting | **Post 1**: how far an LLM gets, measured | `milestones/M2-llm-only.md` |
| M3 | RL on Scenarios | done 2026-10-03 | PPO, reward, training curves, Stable-Baselines3 | **Post 2**: agent learns to shoot and survive | `milestones/M3-rl-on-scenarios.md` |
| M4 | RL Clears E1M1 | done 2026-10-04 | Reward shaping, recurrent memory (LSTM), curriculum by Difficulty | **Post 3**: first original Map Cleared by a learned Driver | `milestones/M4-rl-clears-e1m1.md` |
| M5 | Maps with keys | done 2026-10-10 | Intrinsic reward (RND), ablation | `E1M2` Cleared; with-and-without comparison | `milestones/M5-maps-with-keys.md` |
| M6 | Learning by watching | next | Behavioural cloning, demo recording, fine-tuning | Does a human demo speed up learning? | `milestones/M6-learning-by-watching.md` |
| M7 | Unseen Map | planned | Generalisation, held-out evaluation, overfitting | How a learned Driver does on a Map it never trained on, next to the LLM | written when M6 is done |
| M8 | Driver + Navigator | planned | Hierarchical control, Subgoals | **Big post**: all Contenders, one Scoreboard | written when M7 is done |
| M9 | Doom Episode 1 | planned | Campaign wrapper, generalisation | Video of a full Campaign | written when M8 is done |
| M10 | Toward the full game | planned | Go-Explore, scale (Sample Factory) | Doom Episodes 2 and 3; then the Bonus Doom Episode | written when M9 is done |
| M11 | World model (optional) | planned | DreamerV3 or small diffusion model | A small "dreamed Doom" | written if reached |

Set `Status` to `done` with the completion date when a milestone's completion criteria are all met.

**Why M7 exists** (added 2026-10-04, after M4): M4's policies Cleared `E1M1` 15 of 15 by taking one route every time, trained on that Map alone. Nothing so far measures a learned Driver on a Map it has not trained on, yet that is the question the Driver + Navigator (M8) answers: a Navigator that reads the automap should help most where the Driver has not memorised the way. M7 trains on some Doom Episode 1 Maps and evaluates on one held out, like an exam with questions the homework never had, so M8 has a number to beat. If M5 or M6 already answers it, M7 shrinks to the measurement.

**Every result passes `measurement-checklist.md`** before it reaches the Scoreboard: the checks for the measurement mistakes found so far.

**Beat the game means Difficulty 3** ("Hurt Me Plenty", the game's default), the Difficulty of every Map Eval Spec so far, unless a result states otherwise (`adr/0012-beat-the-game-at-difficulty-3.md`).

## Briefs are written one milestone ahead

Only M0 to M6 have briefs. Each later brief is written when the milestone before it is done, because its design depends on what was measured. Writing a brief means: goal, concepts, steps, completion criteria, open facts to test.

## Pace

The owner works 2-3 hours on most days. Milestones are sized in working sessions, as rough guides:

| Milestone | Rough size |
|---|---|
| M0 | 2-4 sessions |
| M1 | 5-8 sessions |
| M2 | 5-8 sessions |
| M3 onward | estimated in each brief; training time on an RTX 4050 (6 GB) is the main unknown |

## Storage

The machine has one internal 477 GB NVMe SSD. All work happens there: the Ubuntu distro, the Python environment, the code, and any Training Run in progress.

**The working copy lives in the Linux filesystem** (`~/doom-workspace/doom-player` inside Ubuntu), moved at the start of M3. The Windows drive (`/mnt/c/...`) measured about 35 times slower for file access. See `adr/0004-wsl2-ubuntu.md` for the numbers and the move.

The owner has an external drive to be used as an archive only. When a milestone is marked `done`, its heavy outputs (checkpoints, Attempt videos, recorded demos) move to the archive, keeping on the SSD only what the next milestone needs. The archive is unplugged most of the time, so nothing a running process reads or writes may live on it.

The archive drive, formatted 2026-09-28: TOSHIBA External USB 3.0, 465.7 GB, NTFS, volume label `DOOM ARCHIVE`. It mounted as `D:` on first use; Windows may assign another letter later, so identify it by its label. Windows reports its media type as unspecified.

## Known difficulty

From `research/2026-09-28-state-of-the-art-doom-agents.md`:

- Default reward on original Maps is 1 at the exit and 0 elsewhere. Sparse reward is the central technical problem from M4 on.
- The closest public precedent Clears single Maps (`E1M2`, `MAP01`) with one trained agent per Map, in about 8 hours on an RTX 3080.
- Published LLM-only attempts did not Clear `E1M1`. Those used 2024-2025 models, so M2 measured this again: Claude Opus 5.5 Cleared it in 1 of 5 Attempts with screen and actions only, 4 of 5 with the automap, and 5 of 5 with the automap and a notebook, with the game paused between decisions (`milestones/M2-llm-only.md`).
- PPO from screen pixels learns ViZDoom Scenarios in 8 to 32 minutes on the RTX 4050, but a reward that pays the wrong thing gets gamed: on `DeadlyCorridor` the first policy charged forward and died, and no setting tried taught it to survive (`milestones/M3-rl-on-scenarios.md`).
- On `E1M1`, the exit reward alone teaches PPO nothing (8.5% Progress after 1M steps, below random), while a training reward for Progress toward the exit gives 15 Clears in 15 over three training seeds, about 3 hours each. Every Clear follows the same route: the policy learned one Map, not how to find its way (`milestones/M4-rl-clears-e1m1.md`).
- On `E1M2`, whose exit is behind a red door, the same recipe Clears nothing: its reward counts the locked door as open, and every policy walks to that door without the key. A training reward that pays the route through the key gives 8 Clears in 15 over three training seeds, at 10M steps and about 5 hours each; one seed of three never Clears. An exploration bonus (RND) did not help (`milestones/M5-maps-with-keys.md`).
