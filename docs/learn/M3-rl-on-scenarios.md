# M3 - RL on Scenarios: study guide

## 1. What was built

A pipeline that trains a small neural network by reinforcement learning (PPO, from Stable-Baselines3) on three ViZDoom Scenarios, seeing only the screen, and measures each trained policy through the Eval Suite against the random agent on fixed seeds. The policy beat random on `Basic` (80.6 against -218.9), on `DefendCenter` three times over with three training seeds (10.5, 10.3, 10.3 against 0.2), and on `DeadlyCorridor` (298.1 against -94.8). On `DeadlyCorridor` it first gamed the reward by charging forward and dying, and learned to shoot only after the training reward was shaped.

## 2. Concepts

**Reinforcement learning.** A dog learning a trick gets no instructions, only a treat when it happens to do the right thing, and slowly does that thing more. The agent (here, a policy network) acts in an environment (a Scenario), receives a reward after each action, and changes itself to collect more reward over time. Nobody tells it what the buttons do. Source: Sutton and Barto, *Reinforcement Learning: An Introduction*, chapter 1, http://incompleteideas.net/book/the-book-2nd.html

**Policy network (a CNN).** A sorting machine at the post office looks at a letter and drops it into one of a few chutes. The policy looks at the screen and "drops" it into one of the Scenario's actions. It is a convolutional neural network (CNN): layers of small filters that slide over the image and pick out edges, then shapes, then "a monster there". SB3's `CnnPolicy` is the network DeepMind used for Atari. Source: https://www.nature.com/articles/nature14236

**Frame stacking.** One photo of a ball can't tell you which way it is moving; four photos in a row can. Each observation here is the last 4 screens, grayscale, shrunk to 84x84, so the policy can see motion. `make_scenario_env` builds it. Source: the same Atari paper, section "Methods: preprocessing".

**PPO.** Learning to cook by changing the recipe a little after each dinner, never a lot at once, because one bad night shouldn't throw away what worked. PPO plays for a while, measures which actions turned out better than expected, then nudges the policy toward them, with a limit on how far one update may move it. Its main knobs: `learning_rate` (step size), `n_steps` (how much it plays before updating), `n_epochs` (how many passes over that experience), `clip_range` (the limit on each nudge), `ent_coef` (a bonus for staying a little random, so it keeps exploring). `approx_kl` in the logs measures how far one update moved the policy. Sources: https://spinningup.openai.com/en/latest/algorithms/ppo.html and the paper, https://arxiv.org/abs/1707.06347

**Vectorised environments.** One cashier serves one customer at a time; eight cashiers fill the queue faster. Training runs 8 copies of the Scenario in parallel processes (`SubprocVecEnv`), and the policy decides for all 8 at once on the GPU. Measured here: 340 steps per second with 4 copies, 504 with 8. Source: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html

**Scenario reward, and why sparse reward is hard.** A treasure hunt with no "warmer, colder" is nearly hopeless: you only learn anything on the rare run that stumbles onto the treasure. The original Maps pay 1 at the exit and 0 everywhere else, which is that kind of hunt. Each Scenario's reward speaks up much more often: `DefendCenter` pays +1 per kill and -1 for dying (`CONTEXT.md`, **Scenario reward**). Source: Sutton and Barto, chapter 17.4, "Designing reward signals".

**Reward hacking.** A delivery driver paid by the metre, with a small fine for crashing, learns to floor it and crash. `DeadlyCorridor` pays the distance moved and charges 100 for dying, so the first policy pressed `MOVE_FORWARD` on every decision and never fired. The score went up; the behaviour was not what was meant. Only watching it (videos, button counts) showed the difference. Source: https://openai.com/index/faulty-reward-functions/

**Reward shaping.** A coach awards extra points during practice for the habits that win matches; the match itself is still scored the normal way. `RewardShaping` adds +100 per kill and -1 per health point lost during training only; the Eval Suite still scores the Scenario reward (`CONTEXT.md`, **Reward shaping**). Kills and health are read from the engine, so they are Privileged Information, used for reward, never shown to the policy (ADR 0002). Source: Ng, Harada and Russell, 1999, https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf

**Reading a training curve, and training seeds.** One coin landing heads three times says little about the coin. A training curve (reward per episode while training, against steps) is noisy, and two Training Runs with different training seeds can take different paths. Here three seeds on `DefendCenter` ended within 0.2 of each other, but one reached +9 by 300k steps and the others only by 700k to 900k. One run is a note; three show the spread. Source: Henderson et al., *Deep Reinforcement Learning that Matters*, https://arxiv.org/abs/1709.06560

## 3. Reading order

1. `src/doom_player/scenarios.py`, lines 1-45: the module docstring and `make_scenario_env`. Each wrapper does one thing: keep the screen, gray it, shrink it, stack 4. Skip `RewardShaping` (lines 48-77) for now. Then read `tests/test_scenarios.py` lines 14-45: what a policy may see, and that a seed fixes the game.
2. `src/doom_player/contenders/random.py`, lines 24-35, then `src/doom_player/contenders/ppo.py` (29 lines, all of it). Two Contenders, one protocol: `reset(seed, action_space)` and `act(observation) -> action index`. Note `deterministic=True` and where the Contender's name comes from.
3. `src/doom_player/scenario_eval.py`:
   - lines 25-49: the `ScenarioContender` protocol and the three Scenario Eval Specs
   - lines 52-88, `play_scenario`: one Attempt, start to finish, and how `--video` wraps the environment
   - lines 91-118, `run_scenario_eval`: compare it with `eval.run_eval` from M1 (resume by seed)
   - lines 121-138, `check_scenario_rules`: why Attempts by another checkpoint under the same name are refused
   - skip `scenario_row` and `main_scenario` on first read
4. `src/doom_player/eval.py`, lines 166-188, `main`: how `doom-eval` sends a Scenario spec to `main_scenario` and a Map spec to the M1 path.
5. `src/doom_player/train.py`:
   - lines 29-55, `TrainConfig`: every knob, and the comment on why the Atari settings replaced SB3's defaults
   - lines 58-116, `train`: vectorised environments, `VecNormalize`, the W&B run, `model.learn`, and the `training.json` cost record. Skip `main`
6. `src/doom_player/scenarios.py`, lines 48-77, `RewardShaping`, and `train.py` lines 68-70: why it wraps *outside* SB3's `Monitor`. Then `tests/test_scenarios.py` lines 47-66.
7. `src/doom_player/scoreboard.py`, lines 43-46 and 80-97: the Scenario table and its training-cost column.
8. `docs/adr/0010-scenario-referee-on-gymnasium.md`: why Scenarios have their own referee.
9. `docs/milestones/M3-rl-on-scenarios.md`, Open facts and Results: what each Training Run showed, including the collapse and the reward hacking.

Safe to skip: `tests/test_scenario_eval.py`, `tests/test_train.py`, `tests/test_ppo_contender.py`, `tests/test_scoreboard.py` (read them after the code they test; the training tests are marked `slow`), `docs/posts/make_figures_02.py` and `docs/posts/media/` (the post's figures), `results/` (records, curves, the `DeadlyCorridor` behaviour counts and `discarded/`), `pyproject.toml` and `uv.lock` (Stable-Baselines3, `tensorboard`, `opencv-python-headless`, the `slow` marker, `doom-train`), `.gitignore` (`checkpoints/`, `videos/`, `wandb/`), `.mcp.json`, `docs/adr/0004-wsl2-ubuntu.md` (the move to the Linux filesystem), `docs/posts/02-rl-learns-to-shoot.md`, and the edits to `README.md`, `CONTEXT.md`, `docs/ROADMAP.md` and `docs/milestones/M4-rl-clears-e1m1.md`.

## 4. Follow one Attempt

`uv run doom-eval --contender ppo --spec defend-center-v1 --checkpoint checkpoints/defend-center-seed0/model.zip`, seed 1 (12 points, 176 steps).

1. `eval.main` sees that `defend-center-v1` is in `SCENARIO_SPECS` and calls `main_scenario("ppo", DEFEND_CENTER, checkpoint, video=False)`.
2. `main_scenario` builds `PPOContender(checkpoint)`: `PPO.load` reads the network, and `training_record` reads `training.json` next to it (seed 0, 1,001,472 steps, 1,397 s). The Contender's name becomes `ppo-seed0`.
3. `run_scenario_eval` reads `results/attempts/ppo-seed0/defend-center-v1.jsonl`, runs `check_scenario_rules` (same Scenario, same checkpoint `run_id`), and plays the seeds it doesn't have yet.
4. `play_scenario` calls `make_scenario_env("defend-center")`: `gym.make("VizdoomDefendCenter-v1", frame_skip=4)`, then the four wrappers. `env.reset(seed=1)` starts the game and returns a `(4, 84, 84)` array.
5. Each step: `PPOContender.act` calls `model.predict(obs, deterministic=True)`, which runs the CNN once (0.72 ms on the CPU) and returns an action index, say 1. `env.step(1)` turns it into buttons through ViZDoom's `button_map` (index 1 is `ATTACK` in `DefendCenter`), holds them for 4 tics, and returns the next stacked screen and the reward (+1 on a kill).
6. After 176 steps the player dies: `terminated` is true. The Attempt scored 13 kills minus 1 for dying, 12. `play_scenario` returns the record with the checkpoint's `run_id`; `append_record` adds it to the file.
7. With all 10 seeds present, `scenario_row` averages them (10.5), `write_row` replaces `ppo-seed0`'s `defend-center-v1` row in `results/scoreboard.jsonl`, and `log_to_wandb` logs it. `uv run doom-scoreboard` then rewrites the README table.

## 5. Try it

Predict before you run. The first three cost seconds and write nothing; the last two train and take minutes.

1. In a Python shell, play one Attempt without writing records: `play_scenario(PPOContender(Path("checkpoints/defend-center-seed0/model.zip")), "defend-center", 1)` from `doom_player.scenario_eval`. Then edit `ppo.py` to `deterministic=False` and play it again. Predict: higher, lower, or the same reward? Different each time you run it?
2. Open `results/curves/defend-center-seed*.csv` and find, for each training seed, the first step where `ep_rew_mean` passes 9. Predict which seed is first and by how much.
3. Play `RandomScenarioContender()` on `"deadly-corridor"` for seeds 0-9 with `play_scenario` and count how many steps each Attempt lasts. Predict: will any Attempt last more than 50 steps (200 tics, under 6 seconds)?
4. Train `Basic` for 50k steps instead of 200k: `uv run doom-train --scenario basic --seed 7 --total-steps 50000` (about 2 minutes), then evaluate it with `doom-eval --contender ppo --spec basic-v1 --checkpoint checkpoints/basic-seed7/model.zip`. Predict its reward against random's -218.9 and the full run's 80.6. Delete `checkpoints/basic-seed7/` and `results/attempts/ppo-seed7/` afterwards, and restore `results/scoreboard.jsonl` with `git checkout`.
5. In `TrainConfig`, set `ent_coef=0.0` and train `Basic` (seed 7, 200k steps, 8 minutes). Predict from the brief's first collapsed run whether the curve in W&B stays up, and what `approx_kl` does. Clean up as in 4.

## 6. Rebuild it

In `doom-player-study/`, write `m3_reinforce.py` by hand, without importing anything from this repository and without Stable-Baselines3.

- **Build:** REINFORCE, the simplest policy-gradient method and PPO's ancestor, on Gymnasium's `CartPole-v1`. A small PyTorch network maps the 4 numbers of the observation to probabilities over the 2 actions. Play one episode by sampling actions, compute the discounted return from each step to the end, and push up the log-probability of each action in proportion to its return. Log the return of every episode. Then train with 3 seeds and plot the three curves on one chart.
- **Given:** Gymnasium and PyTorch (both already installed for this project), Spinning Up's "Intro to Policy Optimization" (https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) for the idea and its equations, and CartPole's documentation for the observation and reward.
- **Check:** with at least one seed, the mean return of the last 100 episodes reaches 475 or more (CartPole's ceiling is 500) within 2,000 episodes. Your chart shows the three seeds taking different paths, as the `DefendCenter` seeds did.

## 7. Check yourself

1. The first `DeadlyCorridor` policy scored 193.6 against random's -94.8 on the same seeds. Why is that not evidence that it learned to play, and which two checks in this milestone showed what it really did?
2. `RewardShaping` is applied outside SB3's `Monitor`. What would the W&B curve of the shaped run show if it were applied inside, and why would that make the shaped and unshaped curves hard to compare?
3. The three `DefendCenter` training seeds end within 0.2 of each other. What would you have concluded about how fast PPO learns `DefendCenter` if you had trained seed 1 only?
4. The collapsed `Basic` run showed `approx_kl` jumping from about 0.01 to 0.5-0.77 before the reward fell. What does `approx_kl` measure, and how do a smaller `clip_range` and fewer `n_epochs` keep it down?
5. The health-penalty-5 policy scored 320.8 and the penalty-1 policy 298.1, yet the Scoreboard keeps the penalty-1 policy. Give the argument for that choice, and say what measurement would settle which setting is better.
