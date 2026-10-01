# doom-player

Can an AI agent beat the original Doom (1993)?

No public record of one doing so was found when this project started (see `docs/research/`). This repository is an attempt to get there in the open, one measurable milestone at a time, comparing three families of agent on the same maps with the same ruler:

- **LLM only**: a language model playing through tools
- **Reinforcement learning**: a small neural network that learns by trial and error
- **Hybrid**: a fast learned Driver steered by a slow LLM Navigator

"Beat Doom" here means the three episodes of the 1993 release. The fourth episode, added by The Ultimate Doom in 1995, is a bonus goal.

## Status

M0 - Workshop done: a random agent plays `E1M1` on video, logged to a [public W&B project](https://wandb.ai/luks-monteiro-13-my-own/doom-player). Current milestone: **M1 - Ruler and socket**.

## Scoreboard

Empty until milestone M2.

| Contender | Map | Difficulty | Clear Rate | Seeds | Observation class | Cost |
|---|---|---|---|---|---|---|

## Rules of the game

- The finished agent sees what a human sees: the screen, the HUD, and the automap.
- Information a human would not have may be used to compute training reward, and every such use is declared next to the result.
- Every number on the Scoreboard comes from one fixed Eval Suite.

## Where to look

| You want | Read |
|---|---|
| The plan | `docs/ROADMAP.md` |
| Why decisions were made | `docs/adr/` |
| The project's vocabulary | `CONTEXT.md` |
| Guided reading of the code | `docs/learn/` |
| The research behind the plan | `docs/research/` |

## Quick start

Runs on Linux (the project uses Ubuntu 24.04 on WSL2) with [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv sync                  # create .venv and install everything
uv run doom-check        # GPU, ViZDoom version, WAD present?
uv run doom-random --env VizdoomDoomE1M1-S1-v0 --seed 0
```

The last command plays one Attempt on `E1M1` with a random agent, saves the video to `videos/`, and logs it to Weights & Biases (run `uv run wandb login` once first, or set `WANDB_MODE=offline`). Without a WAD, the default `VizdoomFreedoom1E1M1-S1-v0` runs on the free Freedoom Map instead.

If the repository sits on the Windows drive (`/mnt/c/...`), set `UV_LINK_MODE=copy` to silence `uv`'s hardlink warning.

## Game data

This repository contains no game data. The original maps require a purchased copy of Doom; place `doom.wad` in `wads/`.
