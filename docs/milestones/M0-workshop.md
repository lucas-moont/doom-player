# M0 - Workshop

## Goal

A working development environment in which ViZDoom loads `E1M1`, a random agent plays it, and the Attempt is recorded as video and logged to Weights & Biases.

## Concepts the owner learns

- What WSL2 is and why the project runs on Linux
- Virtual environments and dependency management with `uv`
- The Gymnasium API: `reset`, `step`, observation, action, reward, `terminated`, `truncated`
- What a WAD is and how ViZDoom finds it
- What experiment tracking is

## Starting state (verified 2026-09-28)

- Windows 11, RTX 4050 Laptop GPU (6 GB VRAM), i5-13420H, 15.7 GB RAM
- WSL 2.6.3 (kernel 6.6.87.2) with Ubuntu 24.04.5 LTS installed on 2026-09-28 as distro `Ubuntu-24.04`. The default user is a non-root account with `sudo` rights; `sudo` asks for a password only the owner knows, so commands needing it are run by the owner. It is the default distro, so a plain `wsl` opens it.
- Ubuntu ships Python 3.12.3. Nothing else is installed in it yet.
- Real limits inside WSL2 (measured 2026-09-28): 7.6 GB RAM (the WSL default of half the machine's memory; no `.wslconfig` exists), 12 logical CPUs. `df` reports about 955 GB free, which is the virtual disk's ceiling; the real limit is free space on the Windows `C:` drive, 146.5 GB at the time of measurement. WSL virtual disks grow on demand and do not shrink on their own; reclaiming space after deleting data takes `wsl --shutdown` plus `diskpart` `compact vdisk` in an elevated shell, run by the owner. Check `C:` free space before any long Training Run.
- On Windows: Python 3.14.3, `uv`, `git`
- The WAD is available in the workspace at `../doom/DOOM.WAD` (outside this repository). Verified 2026-09-28: `IWAD`, 12,408,292 bytes, 36 Maps (`E1M1` to `E4M9`), The Ultimate Doom, MD5 `c4fe9fd920207691a9f493668e0a2083`.
- The file name is uppercase. ViZDoom looks for the exact name `doom.wad`, and Linux file names are case-sensitive, so copy it as `wads/doom.wad`.

## Steps

1. Done 2026-09-28: Ubuntu installed on WSL2, non-root user created.
2. Done 2026-09-28: `nvidia-smi` inside Ubuntu lists the GPU (see open fact 1).
3. Install `uv` and create the project with a pinned Python version supported by ViZDoom 1.3.1 (`>=3.10, <3.15`) and by PyTorch.
4. Install `vizdoom`, `gymnasium`, `torch`, `wandb`. Confirm `torch.cuda.is_available()` is true.
5. Load `VizdoomFreedoom1E1M1-S1-v0` first: it needs no purchased WAD, so it isolates environment problems from WAD problems.
6. Place the WAD in `wads/` and load `VizdoomDoomE1M1-S1-v0`.
7. Run a random agent for one full Attempt, record it as video, and log the Attempt's length and reward to a public W&B project.
8. Write the study guide `docs/learn/M0-workshop.md`.

## Open facts to test

Each is recorded below as confirmed or refuted, with the evidence.

| # | Fact | Source of doubt |
|---|---|---|
| 1 | The RTX 4050 is visible inside Ubuntu on WSL2 | **Confirmed 2026-09-28**: `nvidia-smi` reports `NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB`, driver 556.19. `torch.cuda.is_available()` is still untested |
| 2 | ViZDoom accepts The Ultimate Doom `doom.wad` and loads `E1M1` from it | **Confirmed 2026-09-30**: `VizdoomDoomE1M1-S1-v0` resets and steps with the owner's WAD (MD5 `c4fe9fd9...`); `game.get_doom_map()` returns `e1m1`, and a random-agent Attempt runs to the end |
| 3 | `libopenal1` is required for original-Map environments | **Refuted 2026-09-30**: with `libopenal1` not installed (`dpkg-query: no packages found`), the audio buffer is enabled and returns `(5040, 2)` samples per step on original `E1M1` and on Freedoom |
| 4 | A public W&B project renders for a logged-out visitor | Not stated on the pricing page |
| 5 | How ViZDoom locates a WAD kept in `wads/` instead of the package directory | **Answered 2026-09-30**: with no path given, ViZDoom fails with `FileDoesNotExistException` for `.../site-packages/vizdoom/scenarios/doom.wad`, the config file's own directory. The Gymnasium wrapper passes extra `gym.make` keyword arguments to `game.set_config()` before `game.init()`, so `doom_game_path=<absolute path to wads/doom.wad>` works with no copy into the package. See `make_env` in `src/doom_player/random_agent.py` |

## Completion criteria

- [ ] `nvidia-smi` inside Ubuntu lists the RTX 4050, and `torch.cuda.is_available()` returns true
- [ ] A single documented command runs a random agent on `E1M1` from a fresh clone plus a WAD
- [ ] A video file of that Attempt exists and plays
- [ ] The Attempt appears in a W&B project, opened successfully in a logged-out browser
- [ ] All five open facts are recorded as confirmed or refuted
- [ ] `git status` shows no WAD file tracked or staged
- [ ] `docs/learn/M0-workshop.md` exists and follows `docs/learn/README.md`
- [ ] `README.md` status line and `ROADMAP.md` status are updated

## Results

Filled in when the milestone is done.
