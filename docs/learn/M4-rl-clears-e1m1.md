# M4 - RL Clears E1M1: study guide

## 1. What was built

A Gymnasium face on the M1 referee, so a PPO policy trains on `E1M1` inside the very `AttemptSession` that later scores it, plus a training reward for new ground toward the exit. On the exit reward alone, a policy reached 8.5% Progress, below random's 20%. With Progress shaping, three training seeds of 5M steps (about 3 hours each) Cleared 15 of 15 Attempts on `e1m1-v1`, at about a second per Attempt (0.8 to 0.9 s on average) against the LLM's minutes and millions of tokens.

## 2. Concepts

**Sparse reward.** A treasure hunt where you only hear "found it!" at the treasure, with no "warmer" or "colder" on the way. `E1M1` pays 1 at the exit and nothing else, so a policy that starts by pressing random buttons almost never feels a reward and has nothing to learn from. The sparse baseline in the brief shows it: a million steps, then staring at a wall. Source: Sutton and Barto, *Reinforcement Learning: An Introduction*, section 17.4 "Designing reward signals", http://incompleteideas.net/book/the-book-2nd.html

**Progress shaping.** A sat-nav saying "312 metres to go, 280, 250": you never see the map, but every step closer is confirmed. `ProgressShaping` pays the policy for each new best walking distance toward the exit, measured from the Map's geometry (`CONTEXT.md`, **Progress** and **Reward shaping**). Only a new best pays, so going back and forth earns nothing. The distance is Privileged Information: it shapes the reward during training, and the policy never sees it (ADR 0002). Source: Ng, Harada and Russell, 1999, https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf

**A Gymnasium face on the referee.** A travel adapter on a plug: the socket (Stable-Baselines3) only takes one shape, so the adapter gives the referee that shape without changing what it does. `MapEnv` turns each `AttemptSession` into one Gymnasium episode, so the policy trains on exactly the screen, time limit and seeds it is scored on (ADR 0011). Source: https://gymnasium.farama.org/introduction/create_custom_env/

**Action set.** A game controller with fewer buttons. Instead of 19 buttons pressed in any combination (2^19 choices), the policy picks one of 11 combinations, such as "forward + turn left" or "use" (`CONTEXT.md`, **Action set**). Exploring a small menu is much faster than a huge one. It limits what the policy does, not what it sees. Source: the discrete action spaces in https://stable-baselines3.readthedocs.io/en/master/guide/algos.html

**Training seeds versus eval seeds.** Practising on next week's exam questions would make the exam meaningless. Training games use seeds from 1000 up (`maps.TRAINING_SEEDS_FROM`) and the Eval Spec uses 0 to 4, so the policy is measured on games it never practised. A second lesson from the review: SB3 seeds copy *k* with `seed + k`, so neighbouring training seeds shared most of their games until `train.SEED_SPACING` spaced them apart. Source: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html

**Recurrent policy (LSTM).** Reading a book one page at a time, but remembering the story so far. The CNN policy sees only the last 4 frames and forgets everything older; a recurrent policy carries a small memory (its "state") from one decision to the next, so it could recall a room it passed a minute ago. An LSTM is the most common kind of that memory. `--recurrent` trains one (`RecurrentPPO` from `sb3-contrib`); M4 didn't need it, since the plain CNN already Cleared every Attempt. Source: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**When measuring changes what is measured.** A thermometer that warms the water it measures. Filming an Attempt one tic at a time redrew the status bar face differently, and a policy that reads the screen then played a different game: one policy Cleared 4 of 5 when filmed and 5 of 5 when not. Scored Attempts are never filmed that way now, and a test checks that filming leaves what the policy sees unchanged (`tests/test_maps.py`). Source: the observer effect, https://en.wikipedia.org/wiki/Observer_effect_(physics), used here only as an analogy.

## 3. Reading order

1. `src/doom_player/maps.py` (125 lines): read it all.
   - lines 18-32: the Action set and where training seeds start
   - lines 35-90, `MapEnv`: `reset` creates a new `AttemptSession` (note `options={"game_seed": s}` for the Eval Suite), `step` presses one combination for 4 tics, and the comment explains why video is one frame per decision
   - lines 92-121, `ProgressShaping`: why only a new best pays
2. `src/doom_player/session.py`, lines 132-140: `screen` and the `progress` property, the one Progress rule shared by the record and the shaping. Then lines 141-150, the docstring of `act`, for the warning about filming.
3. `src/doom_player/train.py`:
   - lines 39-109, `TrainConfig`: the Map fields, the checks in `__post_init__`, and how `checkpoint_folder`, `contender` and `env_factory` differ for Maps and Scenarios
   - lines 111-185, `train`: why the seeds are spaced, how `--init-from` keeps the settings you asked for, and which callbacks run
   - lines 243-260, `ProgressCurve`: a curve for people to read, since the Map's own reward is 0 until the first Clear
   - skip `IntermediateCheckpoints` and `main` on first read
4. `src/doom_player/contenders/base.py` (both protocols) and `src/doom_player/contenders/ppo.py`, lines 56 to the end, `PPOMapContender`: how the Action set is checked and how `play_attempt` drives a whole Attempt, carrying an LSTM's state when there is one.
5. `src/doom_player/eval.py`, lines 73-160: `run_eval` (the two kinds of Contender, the video frame rate, the checkpoint stamp) and the checks it calls.
6. `docs/adr/0011-maps-train-through-the-attempt-session.md`, then `docs/milestones/M4-rl-clears-e1m1.md`, Open facts and Results.

Safe to skip: `tests/` (read `tests/test_maps.py` after `maps.py`: the "same game as the session" and filming tests show what the env promises), `tests/conftest.py` (helpers for tiny Training Runs), `src/doom_player/video.py` (the MP4 writer, moved out of the MCP server), `src/doom_player/scenario_eval.py`, `scenarios.py`, `progress.py`, `scoreboard.py` and `contenders/random.py` (small changes: the shared policy view, a cached distance, the training-cost column, `training = None` on the random Contender), `src/doom_player/mcp_server.py` (now imports the writer), `docs/posts/03-rl-clears-e1m1.md`, `docs/posts/make_figures_03.py` and `docs/posts/media/` (the post and its figures), `docs/milestones/M5-maps-with-keys.md` (the next brief), `results/` (records, curves and `discarded/`), `pyproject.toml` and `uv.lock` (`sb3-contrib`), and the edits to `README.md`, `CONTEXT.md` and `docs/ROADMAP.md`.

## 4. Follow one Attempt

`uv run doom-eval --contender ppo --spec e1m1-v1 --checkpoint checkpoints/e1m1-d3-seed0-shaped/model.zip`, game seed 0 (Cleared in 716 tics).

1. `eval.main` builds `PPOMapContender(checkpoint)`: `load_policy` reads `training.json` (map `E1M1`, Difficulty 3, not recurrent) and loads the PPO network on the CPU; `__init__` checks the network still has 11 actions. Its name is `ppo-e1m1-d3-seed0-shaped`.
2. `run_eval` reads `results/attempts/ppo-e1m1-d3-seed0-shaped/e1m1-v1.jsonl`, runs `check_rules` and `check_trained_on`, and plays the seeds still missing.
3. `play_attempt` calls `make_map_env("E1M1", 3, 6300, contender=...)`: a `MapEnv` wrapped in the policy view. `env.reset(options={"game_seed": 0})` builds an `AttemptSession` with seed 0, exactly as Door A would.
4. Each decision: the policy reads the 4 stacked 84x84 frames and picks an action index, say 2, "forward + turn right". `MapEnv.step` looks it up in `pressed` and calls `session.act(buttons, 4)`, which runs 4 tics and measures Progress once.
5. After 179 decisions the player presses `USE` at the exit switch. `session._finish` sets `cleared=True` and Progress 1.0, and `MapEnv.step` returns the record in `info`.
6. `run_eval` stamps the record with the checkpoint's `run_id` and appends it. After the 5 seeds, `scoreboard_row` adds the training cost from `training.json`, and `doom-scoreboard` puts the row in the README.

## 5. Try it

Predict before you run. The first three use the trained checkpoints and take seconds; the last two train.

1. Play one Attempt by hand from Python: `PPOMapContender(Path("checkpoints/e1m1-d3-seed0-shaped/model.zip")).play_attempt(STANDARD_E1M1, 7)`. Game seed 7 is not one of the eval seeds. Predict: Clear or not, and roughly how many tics?
2. Play the same checkpoint at Difficulty 4 with an `EvalSpec("try-d4", "E1M1", 4, (0, 1, 2, 3, 4), 6300)` and `run_eval(..., results_dir=Path("/tmp/try"))`. Predict the Clear Rate when monsters hit harder.
3. Open `results/curves/e1m1-d3-seed1-shaped.csv` and find the first step where `progress` passes 0.9 and the first where `clear_rate` is above 0. Predict the gap between them.
4. Train with `--death-penalty 0` (training seed 7, 1M steps, about 35 minutes, `--label nodeath`). Predict whether the 1M-step policy Clears more or less often than the brief's 1M checkpoint (3 of 5).
5. In `maps.py`, remove "use" from `ACTIONS` and train for 1M steps (`--label nouse`). Predict how far Progress gets, then explain why from what the exit needs. Restore `ACTIONS` afterwards; old checkpoints will refuse to load until you do.

## 6. Rebuild it

In `doom-player-study/`, write `m4_shaping.py` by hand, without importing anything from this repository.

- **Build:** a tiny grid maze (for example 7x7, with walls, a start and an exit) as a Gymnasium environment with 4 actions. Write a breadth-first search that gives every cell its walking distance to the exit, and a wrapper that adds a Progress reward: the drop in the best distance so far. Train a tabular Q-learning agent twice, once on the exit reward alone and once with the wrapper, for the same number of episodes.
- **Given:** Gymnasium's custom environment guide (https://gymnasium.farama.org/introduction/create_custom_env/), Sutton and Barto section 6.5 (Q-learning), and the `ProgressShaping` docstring for the idea, not the code.
- **Check:** with the wrapper, the agent reaches the exit in most episodes well before the agent without it, on the same maze and seeds. A test shows that walking back and forth earns the shaped agent nothing.

## 7. Check yourself

1. The sparse baseline scored 8.5% Progress, less than random's 20%. How can a trained policy do worse than random button presses?
2. `ProgressShaping` pays only for a new best distance. Describe a policy that would farm reward if every step closer paid, even after stepping back first.
3. Why do training games use seeds from 1000 up, and what went wrong with `seed + k` for neighbouring training seeds?
4. One policy Cleared 4 of 5 when filmed tic by tic and 5 of 5 when not, though the player's position was identical either way. Explain how, and why this matters for a policy but not for the random agent.
5. All 15 Clears take nearly the same route and time. What does that tell you about what was learned, and what would you test to find out whether the policy generalises?
