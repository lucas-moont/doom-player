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

- [ ] The working copy lives in the Linux filesystem, and the tests pass there
- [ ] A trained PPO policy beats the random agent on `Basic` and on `DefendCenter`, measured by the Eval Suite over fixed seeds
- [ ] Each Training Run is in W&B with its curve, config and seed
- [ ] Training is repeated with at least 3 training seeds on one Scenario, to show the spread between seeds
- [ ] A video of a trained agent exists for each Scenario
- [ ] Observation class and any privileged reward terms are stated in Results
- [ ] `docs/learn/M3-rl-on-scenarios.md` exists
- [ ] Post 2 material is drafted in `docs/posts/02-rl-learns-to-shoot.md`
- [ ] The M4 brief is written

## Open facts to test

| # | Fact | Source of doubt |
|---|---|---|
| 1 | Stable-Baselines3 2.9 accepts ViZDoom's Gymnasium environments once the observation is wrapped to a single image | ViZDoom returns a `Dict` observation; SB3's `CnnPolicy` expects an image `Box` |
| 2 | Training throughput on the RTX 4050 inside WSL, in environment steps per second, with N parallel environments | Not measured; 7.6 GB of RAM in WSL limits how many ViZDoom processes run at once |
| 3 | How many steps PPO needs on `Basic` and `DefendCenter` to beat random clearly | Published numbers use other hardware and settings |
| 4 | W&B's SB3 integration logs curves and videos without extra code | Not tested in this project |
| 5 | The move to the Linux filesystem keeps W&B login, the WAD copy and the MCP server working | The paths in `.mcp.json` and the workspace `CLAUDE.md` assume `/mnt/c` |

## Rough size

5 to 8 sessions, most of it waiting on Training Runs; open fact 2 decides the real number.

## Results

Filled in when the milestone is done.
