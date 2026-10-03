# M3 - RL on Scenarios

## Goal

Train small neural networks by reinforcement learning to shoot and survive on ViZDoom Scenarios, and show learning happening: a training curve that rises, and an agent that plays visibly better than the random one. This is the first learned Contender, and the base the later milestones build on to Clear original Maps.

Written 2026-10-02, after M2. M2 showed that an LLM with an automap Clears `E1M1`, at seconds and millions of tokens per decision-heavy Attempt. M3 starts the other road: a policy that decides in milliseconds and knows no language.

## Concepts the owner learns

- What reinforcement learning is: an agent, a reward, and trial and error
- The policy network: from pixels to button presses (a CNN)
- PPO: what one update does, and its main knobs (learning rate, rollout length, clipping)
- Reward design on a Scenario, and why the default reward of the original Maps (1 at the exit, 0 elsewhere) is too sparse to start with
- Reading a training curve: episode reward, episode length, and the noise between seeds
- Vectorised environments: several games in parallel to feed the GPU

## Starting state

- Eval Suite, Scoreboard, replay and W&B logging exist (M1, M2). The Scoreboard has four rows on `E1M1`.
- ViZDoom ships these Scenarios as Gymnasium environments (checked 2026-10-02 in the installed package): `VizdoomBasic-v1`, `VizdoomDefendCenter-v1`, `VizdoomDefendLine-v1`, `VizdoomHealthGathering-v1`, `VizdoomHealthGatheringSupreme-v1`, `VizdoomDeadlyCorridor-v1`, `VizdoomMyWayHome-v1`, `VizdoomTakeCover-v1`, and others.
- Stable-Baselines3 2.9.0 is current on PyPI and requires `gymnasium>=0.29.1,<2.0` and `torch>=2.8,<3.0`, which the project's `gymnasium` 1.3 and `torch` 2.14.1+cu126 satisfy on paper.
- The working copy is still on `/mnt/c`, measured about 35 times slower for file access than the Linux filesystem. ADR 0004 sets the move **before the first Training Run**.

## Steps

1. **Move the working copy into the Linux filesystem** (ADR 0004's five steps). Update the workspace `CLAUDE.md` and `.mcp.json` in the same change. Re-run `uv run pytest` and one `doom-eval` there.
2. Install Stable-Baselines3. Wrap the ViZDoom observation into what a CNN policy expects: the screen only, grayscale, downscaled (for example 84x84), with recent frames stacked. Record the observation class: screen pixels only is Human-equivalent.
3. **`Basic`** (one monster, shoot it): train PPO, log to W&B, evaluate the trained policy over fixed seeds, record a video. This is the "hello world" that proves the pipeline.
4. **`DefendCenter`** (enemies from every side, limited ammo): the Post 2 headline, "learns to shoot and survive".
5. **One harder Scenario**, `HealthGathering` or `DeadlyCorridor`, chosen from what 3 and 4 cost in training time.
6. Add Eval Specs for Scenarios (seeds, episode limit) so their results go through the Eval Suite like everything else, on a Scenario table next to the `E1M1` Scoreboard.
7. Draft Post 2 material in `docs/posts/02-rl-learns-to-shoot.md`, and write the study guide.

## Constraints

- A Contender sees Human-equivalent Observations only (ADR 0002). Any privileged reward term (for example, distance travelled from position) is declared in Results.
- Every result goes through the Eval Suite with seeds and cost; for RL, cost is training time on the RTX 4050 plus evaluation time.
- Training Runs are logged to W&B, one run per Training Run.

## Completion criteria

- [x] The working copy lives in the Linux filesystem, and the tests pass there
- [x] A trained PPO policy beats the random agent on `Basic` and on `DefendCenter`, measured by the Eval Suite over fixed seeds
- [ ] Each Training Run is in W&B with its curve, config and seed
- [x] Training is repeated with at least 3 training seeds on one Scenario, to show the spread between seeds
- [ ] A video of a trained agent exists for each Scenario
- [ ] Observation class and any privileged reward terms are stated in Results
- [ ] `docs/learn/M3-rl-on-scenarios.md` exists
- [ ] Post 2 material is drafted in `docs/posts/02-rl-learns-to-shoot.md`
- [ ] The M4 brief is written

## Open facts to test

| # | Fact | Source of doubt |
|---|---|---|
| 1 | Stable-Baselines3 2.9 accepts ViZDoom's Gymnasium environments once the observation is wrapped to a single image | ViZDoom returns a `Dict` observation; SB3's `CnnPolicy` expects an image `Box`. **Confirmed 2026-10-03**: `make_scenario_env` keeps the screen only, grayscale, 84x84, 4 frames stacked, giving a `(4, 84, 84)` uint8 `Box`; SB3 trains on it without changes (`tests/test_train.py`). Resizing needs `opencv-python-headless` |
| 2 | Training throughput on the RTX 4050 inside WSL, in environment steps per second, with N parallel environments | Not measured; 7.6 GB of RAM in WSL limits how many ViZDoom processes run at once. **Measured 2026-10-03** on `Basic`, 16k-step probes with SB3's default PPO: 340 steps/s with 4 environments, 504 with 8, 533 with 12; RAM in use stayed under 1 GB. 8 is the default. A full 200k-step Training Run with the final settings runs at about 415 steps/s (484 s) |
| 3 | How many steps PPO needs on `Basic` and `DefendCenter` to beat random clearly | Published numbers use other hardware and settings. **`Basic`, 2026-10-03**: the curve passes random's level by about 30k steps and levels off near +80 by about 70k; 200k steps (8 min) gives 80.6 on `basic-v1` against random's -218.9. **`DefendCenter`, 2026-10-03**: the training curve (episode reward while still exploring) passes +2 by 50k steps, +5 by 150k, +7.5 by 450k and reaches about +9.8 at 1M, still creeping up; `approx_kl` stayed under 0.01 throughout. 1M steps (23 min) gives 10.5 on `defend-center-v1` against random's 0.2 |
| 4 | W&B's SB3 integration logs curves and videos without extra code | Not tested in this project. **Partly confirmed 2026-10-03**: with `wandb.init(sync_tensorboard=True)` and SB3's `tensorboard_log`, every SB3 curve reaches W&B without a callback (needs the `tensorboard` package). Videos are not logged by training; `doom-eval --video` saves Attempt videos to `videos/` |
| 5 | The move to the Linux filesystem keeps W&B login, the WAD copy and the MCP server working | The paths in `.mcp.json` and the workspace `CLAUDE.md` assume `/mnt/c`. **Confirmed 2026-10-03**: in `~/doom-player`, `doom-check` finds the WAD (MD5 matches), 30 tests pass in 25 s, W&B reads its credentials from `~/.netrc`, and the MCP server answers `initialize` in 1.5 s once `.mcp.json` runs it from `~/doom-player` |

## Rough size

5 to 8 sessions, most of it waiting on Training Runs; open fact 2 decides the real number.

## Results

Filled in when the milestone is done. Notes so far:

- **Observation class**: `human-equivalent` for every Contender here. A Scenario Contender sees the stacked grayscale screen only, not even the HUD variables ViZDoom offers. Training uses each Scenario's built-in reward, rescaled by `VecNormalize`. On `Basic` and `DefendCenter` that reward has no privileged terms. **`DeadlyCorridor`'s reward is privileged**: each step pays the change in the player's x position (toward the armour at the corridor's end), minus 100 on death. Checked 2026-10-03 by walking forward and back with `POSITION_X` read alongside: reward equalled the x change on every step (+7.42 for +7.42, -15.87 for -15.87). The policy never sees the position; only its reward is computed from it (ADR 0002).
- **First `Basic` Training Run collapsed** (W&B run `6e2x96q3`). With SB3's default PPO settings the curve rose to +78 by 84k steps, then fell to -300 (the policy stopped shooting and waited out the clock) and stayed there. Around 85k steps `approx_kl`, the size of one update, jumped to 0.5-0.77 against about 0.01 before, and the value loss rose from about 150 to about 2,000. Fix: RL Zoo's Atari settings (4 epochs, clip range 0.1, entropy bonus 0.01) and reward normalisation. The second run (`wdkr5yea`) kept `approx_kl` near 0.01 and held +80 to the end. The collapsed run's Attempts are kept in `results/discarded/`.
- **`DefendCenter`, training seed 0** (W&B run `5gqzcemn`, 1,001,472 steps, 23 min): 10.5 mean reward on `defend-center-v1` (6 to 12 by seed) against random's 0.2. The Scenario gives +1 per kill (scripted in its WAD) and -1 on death (`death_penalty = 1` in `defend_the_center.cfg`), so an Attempt that ends in death scores kills minus one. Every Attempt here ended in death before the 2,100-tic timeout (the longest lasted 224 steps of 4 tics), so the policy kills about 11 monsters per Attempt (7 to 13) where random kills 0 to 2. Surviving longer, not only shooting, is what the policy has yet to learn. Videos: `videos/ppo-seed0-defend-center-seed*-episode-0.mp4`.
- **`DefendCenter`, three training seeds** (same settings, 1M steps each, run one after the other so their training times compare):

  | Training seed | W&B run | Training time | `defend-center-v1` reward | By eval seed |
  |---|---|---|---|---|
  | 0 | `5gqzcemn` | 23 min | 10.5 | 6 to 12 |
  | 1 | `en65c3af` | 25 min | 10.3 | 8 to 13 |
  | 2 | `9t6habd8` | 22 min | 10.3 | 9 to 12 |

  The three policies end within 0.2 of each other (mean 10.4), against random's 0.2. The spread shows in the *path*, not the end: seed 1's training curve reached +9 by 300k steps and levelled off near +10 from 600k; seeds 0 and 2 reached +9 only around 700k to 900k. All three curves end between +9 and +10 and `approx_kl` stayed under 0.01. Seed 1 was first evaluated while seed 2 trained on the same machine (2.25 s per Attempt); re-evaluated alone it gave the same reward on every seed, at 1.78 s. The contended Attempts are kept in `results/discarded/`. Evaluation cost runs 1.3 to 1.8 s per Attempt across the three policies, all with video recording on.
- **Harder Scenario: `DeadlyCorridor`**, chosen over `HealthGathering` on 2026-10-03. Both fit the budget (about 23 min per million steps); `DeadlyCorridor` combines moving and shooting, the closest Scenario to an original Map and so the better step toward M4. It runs at skill 5 (Nightmare) with 7 buttons. Random scores -94.8 on `deadly-corridor-v1` and dies in every Attempt, after 12 to 43 steps.
- **`DeadlyCorridor`, training seed 0: the policy games the reward** (W&B run `c6ad79c0`, 1,001,472 steps, 31 min). It scores 193.6 on `deadly-corridor-v1` (51 to 451 by seed) against random's -94.8, but a debug run of the same 10 Attempts (reading the button index, `is_player_dead` and `KILLCOUNT`) shows it presses `MOVE_FORWARD` on every one of 150 decisions, never `ATTACK`, and dies in all 10, with one kill in total. Running forward pays the distance covered, which outweighs the 100-point death penalty, so "charge and die" is a strategy the reward prefers to standing still. The training curve shows when it settled: it jumped from -94 to +142 within 50k steps and then stayed between +190 and +260, while `approx_kl` fell toward zero, a policy that had stopped changing. More steps on the same settings are unlikely to help. Kept as evidence: checkpoint `checkpoints/deadly-corridor-seed0-unshaped/`, Attempts `results/discarded/ppo-seed0-deadly-corridor-v1-unshaped.jsonl`, videos `videos/deadly-corridor-unshaped/`.
