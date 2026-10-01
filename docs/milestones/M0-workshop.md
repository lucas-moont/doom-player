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

All done; steps 3-8 on 2026-09-30, tracked as issues #1-#6 under the GitHub Milestone "M0 - Workshop" and delivered in PR #7.

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
| 1 | The RTX 4050 is visible inside Ubuntu on WSL2 | **Confirmed 2026-09-28**: `nvidia-smi` reports `NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB`, driver 556.19. **Confirmed 2026-09-30** for PyTorch too, after a fix: PyPI's default `torch` 2.14.1 targets CUDA 13 and reported `is_available() == False` ("NVIDIA driver on your system is too old (found version 12050)"). The CUDA 12.6 build from PyTorch's `cu126` index (set in `pyproject.toml`) returns `True` and names the RTX 4050 |
| 2 | ViZDoom accepts The Ultimate Doom `doom.wad` and loads `E1M1` from it | **Confirmed 2026-09-30**: `VizdoomDoomE1M1-S1-v0` resets and steps with the owner's WAD (MD5 `c4fe9fd9...`); `game.get_doom_map()` returns `e1m1`, and a random-agent Attempt runs to the end |
| 3 | `libopenal1` is required for original-Map environments | **Refuted 2026-09-30**: with `libopenal1` not installed (`dpkg-query: no packages found`), the audio buffer is enabled and returns `(5040, 2)` samples per step on original `E1M1` and on Freedoom |
| 4 | A public W&B project renders for a logged-out visitor | **Confirmed with a caveat 2026-10-01**: after setting the project to Public, the owner opened the Attempt's page in a private browser window and it rendered, but the web video player never showed `attempt_video`. The video itself is public: the W&B API, called with no key, lists `media/videos/attempt_video_0_....mp4` (13,624,711 bytes) and its signed URL downloads as `video/mp4`. The file is H.264 `yuv420p`, a browser-playable format. Follow-up: issue #8 |
| 5 | How ViZDoom locates a WAD kept in `wads/` instead of the package directory | **Answered 2026-09-30**: with no path given, ViZDoom fails with `FileDoesNotExistException` for `.../site-packages/vizdoom/scenarios/doom.wad`, the config file's own directory. The Gymnasium wrapper passes extra `gym.make` keyword arguments to `game.set_config()` before `game.init()`, so `doom_game_path=<absolute path to wads/doom.wad>` works with no copy into the package. See `make_env` in `src/doom_player/random_agent.py` |

## Completion criteria

- [x] `nvidia-smi` inside Ubuntu lists the RTX 4050, and `torch.cuda.is_available()` returns true
- [x] A single documented command runs a random agent on `E1M1` from a fresh clone plus a WAD
- [x] A video file of that Attempt exists and plays
- [x] The Attempt appears in a W&B project, opened successfully in a logged-out browser
- [x] All five open facts are recorded as confirmed or refuted
- [x] `git status` shows no WAD file tracked or staged
- [x] `docs/learn/M0-workshop.md` exists and follows `docs/learn/README.md`
- [x] `README.md` status line and `ROADMAP.md` status are updated

## Results

Measured 2026-09-30 and 2026-10-01. These are notes, not Scoreboard results: the Eval Suite arrives in M1.

**The single command.** From a fresh clone of the M0 branch into the Linux filesystem, plus `wads/doom.wad`:

```bash
uv sync
uv run doom-random --env VizdoomDoomE1M1-S1-v0 --seed 0
```

Without the WAD it stops with `VizdoomDoomE1M1-S1-v0 needs the purchased WAD at .../wads/doom.wad`. The fresh clone contained no WAD file.

**The Attempt.** Random agent on `E1M1`, Difficulty 1, seed 0, frame skip 4, capped at 2100 steps (8400 tics, 4 minutes of game time): 2100 steps, total reward 0, `terminated=False`, `truncated=True`. The same seed gave the same numbers offline, online and from the fresh clone. Without the cap, a seeded random Attempt on Freedoom `E1M1` died after 4648 steps, identically on two runs.

**Video.** `videos/VizdoomDoomE1M1-S1-v0-seed0-<run id>-episode-0.mp4`, 233 s at 9 fps, 640x240: the screen with HUD on the left, the automap on the right. Both are Human-equivalent Observations; no Privileged Information was used anywhere in M0.

**W&B.** Public project https://wandb.ai/luks-monteiro-13-my-own/doom-player, Attempt `stellar-voice-1` (`ge69x13h`), with steps, reward, both flags, config and video.

**Cost.** No LLM tokens. Wall-clock for one capped Attempt with video: about 1 min 45 s with the repository on `/mnt/c`.

**Surprises worth keeping.**

- The PyPI `torch` build needs a newer NVIDIA driver than the one installed; the `cu126` build works on driver 556.19 (open fact 1).
- Switching `torch` builds in place left `libcusparse.so.12` deleted: the old and new NVIDIA wheels share the `nvidia/` folder, and uninstalling the old one removed a file the new one had just written. `uv sync --reinstall-package torch --reinstall-package nvidia-...` fixed it.
- File access under `/mnt/c` is slow, as ADR 0004 predicted: the first `uv sync` took 7 min 52 s, and the smoke test ran in 45.8 s on `/mnt/c` against 1.3 s in the Linux filesystem. Not yet a problem; it becomes one when training starts in M3.
- ViZDoom writes `_vizdoom.ini` into the working directory on every run; it is gitignored.
