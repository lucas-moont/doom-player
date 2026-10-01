# M0 - Workshop: study guide

## 1. What was built

A Python project that runs inside Ubuntu on WSL2 and can drive the original Doom through ViZDoom. One command, `uv run doom-random`, makes an agent press random buttons for one Attempt on `E1M1`, saves a video of it, and sends the Attempt's numbers and video to Weights & Biases. A second command, `uv run doom-check`, tells you whether the GPU, the game engine and the WAD are all in place.

## 2. Concepts

**WSL2.** Think of a Linux computer living inside a box on your Windows desktop: it has its own files and programs, but it shares the same hardware, including the graphics card. That box is WSL2 (Windows Subsystem for Linux, version 2). The project uses Linux because the tools it needs later (ViZDoom for long runs, Sample Factory) are tested on Linux first. Source: https://learn.microsoft.com/en-us/windows/wsl/about

**Virtual environment and `uv`.** A recipe lists exact ingredients. If two recipes need different brands of flour, you keep them in separate pantries. A virtual environment (`.venv/`) is one project's pantry of Python packages, kept apart from the rest of the system. `uv` is the shopper: it reads the recipe (`pyproject.toml`), writes down the exact brand and batch of every ingredient (`uv.lock`), and fills the pantry to match. Source: https://docs.astral.sh/uv/concepts/projects/

**CUDA and driver compatibility.** The graphics card only obeys programs written for a language version its driver understands. The Windows driver here (556.19) understands CUDA up to 12.5. The default PyTorch download is written for CUDA 13, so PyTorch could not see the card. The fix was to download the PyTorch build for CUDA 12.6, which runs on any 12.x driver under NVIDIA's minor-version compatibility rule. Source: https://docs.nvidia.com/deploy/cuda-compatibility/

**The Gymnasium API.** A board game with a referee. You ask the referee to set up the board (`reset`), and it shows you what you can see: the *observation*. You make a move (`step(action)`), and the referee answers with the new observation, a *reward* (points for that move), and two flags. `terminated` means the game ended by its own rules: you died or reached the exit. `truncated` means someone stopped the clock before that happened. Gymnasium calls one game an "episode"; this project calls it an **Attempt** (see `CONTEXT.md`). Source: https://gymnasium.farama.org/introduction/basic_usage/

**Action space.** The list of moves the referee accepts. For original Maps it is `MultiBinary(19)`: 19 on/off switches, one per button (attack, move forward, turn left, ...), any combination pressed at once. `action_space.sample()` flips each switch at random. Source: https://gymnasium.farama.org/api/spaces/fundamental/#multibinary

**Frame skip.** Doom runs at 35 tics (game clock ticks) per second. With frame skip 4, each action is held for 4 tics, so the agent decides about 9 times a second. Fewer decisions make learning cheaper later. Source: https://vizdoom.farama.org/api/python/doom_game/#make-action

**WAD.** The cartridge. The Doom engine is the console; the WAD ("Where's All the Data") holds the Maps, sounds and pictures. ViZDoom needs to be told where the cartridge is. Here it is `wads/doom.wad`, kept out of git because it is copyrighted. Freedoom is a free cartridge with different Maps, bundled with ViZDoom. Source: https://doomwiki.org/wiki/WAD

**Experiment tracking (Weights & Biases).** A lab notebook that fills itself in. Each Training Run (or here, each Attempt) gets a page with its settings, its results and its video, so months later you can see exactly what happened and compare. Source: https://docs.wandb.ai/guides/track/

## 3. Reading order

1. `pyproject.toml`: the recipe. Look at `dependencies`, `[project.scripts]` (where the two commands come from), and the `[tool.uv.sources]` block with its comment about CUDA. Skip `[build-system]`.
2. `src/doom_player/paths.py`: where the WAD is expected. Short; read all of it.
3. `src/doom_player/check_env.py`: the readiness check. Look at `main`. Skip how `md5_of` reads the file in chunks.
4. `src/doom_player/random_agent.py`:
   - lines 26-45, `make_env`: how an environment is created, and how the WAD path is handed to ViZDoom
   - lines 48-64, `play_attempt`: the Gymnasium loop. This is the heart of the milestone; read it until you could write it from memory
   - lines 67-112, `main`: command-line options, W&B, the video recorder. Skip the `argparse` details on first read
5. `tests/test_smoke.py`: what "the environment works" means in code.
6. `docs/milestones/M0-workshop.md`, section "Open facts to test": what was unknown and how each fact was settled.

Safe to skip: `uv.lock` (machine-written list of exact versions), `.python-version` (one line: `3.12`), `src/doom_player/__init__.py` (marks the folder as a package), `.gitignore` (files git must not track: WADs, videos, `.venv/`, `_vizdoom.ini`), `README.md` (the Quick start repeats the commands above).

## 4. Follow one Attempt

Command: `uv run doom-random --env VizdoomDoomE1M1-S1-v0 --seed 0`

1. `uv run` makes sure `.venv/` matches `uv.lock`, then calls `doom_player.random_agent:main`, as `[project.scripts]` says.
2. `main` reads the options: environment id, seed 0, frame skip 4, at most 2100 steps.
3. `wandb.init` opens a new page in the W&B project `doom-player` and records those options.
4. `make_env` sees an id starting with `VizdoomDoomE`, checks that `wads/doom.wad` exists, and calls `gym.make` with `doom_game_path` pointing at it. `max_episode_steps=2100` makes Gymnasium add a `TimeLimit` wrapper, and `render_mode="rgb_array"` lets frames be captured.
5. `RecordVideo` wraps the environment. From now on, every `step` also stores a frame.
6. `play_attempt` calls `env.reset(seed=0)`. Only now does ViZDoom start the engine (`game.init()`), load the WAD and place the player at the start of `E1M1`.
7. The loop: `env.action_space.sample()` picks 19 random switches; `env.step` holds them for 4 tics and returns the reward and the two flags. Steps and reward are added up.
8. The loop stops when `terminated` (the player died or exited) or `truncated` (`TimeLimit` hit 2100 steps) is true.
9. `env.close()` shuts the engine down and makes `RecordVideo` write the `.mp4` into `videos/`.
10. `main` saves the summary and the video to W&B with `run.summary.update`, `run.log` and `run.finish`, then prints the summary.

## 5. Try it

Predict first, then run. Use `WANDB_MODE=offline` to keep these out of the public project.

1. Run with `--frame-skip 1`. Predict: with the same `--max-steps`, does the Attempt cover more or less game time? How long is the video?
2. Run with `--seed 0` twice. Predict: are `steps` and `total_reward` the same both times? Then try `--seed 1`.
3. Run with `--max-steps 100`. Predict: `terminated` or `truncated`?
4. Run with `--env VizdoomDoomE1M1-S5-v0` (Nightmare Difficulty). Predict: does the random agent die before 2100 steps more often?
5. Rename `wads/doom.wad` for a moment and run `uv run doom-check`, then `doom-random` on `E1M1`. Predict the messages. Rename it back.

## 6. Rebuild it

In `doom-player-study/`, write `m0_random.py` by hand, without importing anything from this repository.

- **Build:** a script that creates `VizdoomFreedoom1E1M1-S1-v0` with `gym.make`, plays one Attempt with random actions, and prints the number of steps, the total reward, and which flag ended it. Then add a second agent that always presses only `MOVE_FORWARD` (find its index among the 19 buttons) and compare how long its Attempts last across five seeds.
- **Given:** `uv add vizdoom gymnasium` in the study project, the line `from vizdoom import gymnasium_wrapper` (it registers the environment ids), the Gymnasium basic-usage page linked above, and one hint: a seed must be given to both `env.reset` and `env.action_space`, or the random presses change on every run.
- **Check:** the same seed gives the same numbers twice; the random agent's numbers for seed 0 at frame skip 4 match this repository's `WANDB_MODE=disabled uv run doom-random --seed 0 --max-steps 100000` on Freedoom; the forward-only agent's five results are written down with a sentence on why they differ from random.

## 7. Check yourself

1. PyTorch installed without errors, yet `torch.cuda.is_available()` was `False`. What was wrong, and why did switching to the CUDA 12.6 build fix it without touching the Windows driver?
2. Your Attempt ends with `terminated=False, truncated=True`. Name two different things that could have caused that in this code.
3. Why is the WAD path given to ViZDoom through `gym.make` instead of by copying `doom.wad` into ViZDoom's package folder?
4. The video lasts about 233 seconds for 2100 steps. Work out where that number comes from.
5. The recorded frame shows the automap beside the screen. Is the automap a Human-equivalent Observation or Privileged Information, and where does `CONTEXT.md` say so?
