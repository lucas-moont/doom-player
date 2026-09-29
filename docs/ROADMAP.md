# Roadmap

Goal ladder: Clear `E1M1` → Beat Doom Episode 1 → Beat the Original Doom Episodes (`E1` to `E3`, the 1993 game).

The owner's WAD is The Ultimate Doom, which adds a fourth Doom Episode (`E4`, 1995). It is the Bonus Doom Episode: attempted after the goal ladder is complete, and reported separately.

The order is deliberate: the LLM-only Contender comes before reinforcement learning so the first public result arrives early and frames the question the rest of the project answers. See `adr/0005-showcase-order.md`.

## Milestones

| # | Name | Status | Concepts learned | Result to show | Brief |
|---|---|---|---|---|---|
| M0 | Workshop | next | WSL2, `uv`, Gymnasium API, ViZDoom, W&B | Video of a random agent on `E1M1` | `milestones/M0-workshop.md` |
| M1 | Ruler and socket | planned | Evals, MCP, tool design | Eval Suite + Doom MCP server; random agent on the Scoreboard | `milestones/M1-ruler-and-socket.md` |
| M2 | LLM plays E1M1 | planned | Harness, memory, tool use, cost accounting | **Post 1**: how far an LLM gets, measured | `milestones/M2-llm-only.md` |
| M3 | RL on Scenarios | planned | PPO, reward, training curves, Stable-Baselines3 | **Post 2**: agent learns to shoot and survive | written when M2 is done |
| M4 | RL Clears E1M1 | planned | Reward shaping, recurrent memory (LSTM), curriculum by Difficulty | **Post 3**: first original Map Cleared by a learned Driver | written when M3 is done |
| M5 | Maps with keys | planned | Intrinsic reward (RND), ablation | `E1M2` Cleared; with-and-without comparison | written when M4 is done |
| M6 | Learning by watching | planned | Behavioural cloning, demo recording, fine-tuning | Does a human demo speed up learning? | written when M5 is done |
| M7 | Driver + Navigator | planned | Hierarchical control, Subgoals | **Big post**: all Contenders, one Scoreboard | written when M6 is done |
| M8 | Doom Episode 1 | planned | Campaign wrapper, generalisation | Video of a full Campaign | written when M7 is done |
| M9 | Toward the full game | planned | Go-Explore, scale (Sample Factory) | Doom Episodes 2 and 3; then the Bonus Doom Episode | written when M8 is done |
| M10 | World model (optional) | planned | DreamerV3 or small diffusion model | A small "dreamed Doom" | written if reached |

Set `Status` to `done` with the completion date when a milestone's completion criteria are all met.

## Briefs are written one milestone ahead

Only M0 to M2 have briefs. Each later brief is written when the milestone before it is done, because its design depends on what was measured. Writing a brief means: goal, concepts, steps, completion criteria, open facts to test.

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

The owner has an external drive to be used as an archive only. When a milestone is marked `done`, its heavy outputs (checkpoints, Attempt videos, recorded demos) move to the archive, keeping on the SSD only what the next milestone needs. The archive is unplugged most of the time, so nothing a running process reads or writes may live on it. The drive's type, size and mount path are recorded here the first time it is used.

## Known difficulty

From `research/2026-09-28-state-of-the-art-doom-agents.md`:

- Default reward on original Maps is 1 at the exit and 0 elsewhere. Sparse reward is the central technical problem from M4 on.
- The closest public precedent Clears single Maps (`E1M2`, `MAP01`) with one trained agent per Map, in about 8 hours on an RTX 3080.
- Published LLM-only attempts did not Clear `E1M1`. Those used 2024-2025 models, so M2 measures this again.
