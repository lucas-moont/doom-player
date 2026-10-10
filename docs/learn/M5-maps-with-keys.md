# M5 - Maps with keys: study guide

## 1. What was built

A Progress ruler that can measure `E1M2`, whose exit sits behind a red door and whose red key lies farther from the exit than the spawn does, plus an Eval Spec for it (`e1m2-v1`), an exploration bonus (RND) for training, and a switch that lets the training reward pay the route through the key. M4's recipe, which pays for getting closer with every door counted as open, learned to walk to the locked red door and no further, 0 Clears in 15 Attempts over three training seeds, and adding RND made it worse. Paying the route through the key instead, and training for 10M steps, gave 8 Clears in 15: two training seeds Clear 4 of 5, and the third circles a room past the red door until it is killed.

## 2. Concepts

**Exploration.** Looking for your keys in a new flat: if you only ever walk toward the front door, you never find the keys in the bedroom. A policy learns from what it happens to try, and a reward that says "closer to the exit is better" pushes it straight at the exit, never into the detour the Map needs. On `E1M2` all twenty failed Attempts walked to the red door first. Source: Sutton and Barto, *Reinforcement Learning: An Introduction*, section 2.2 "Action-value methods" (exploration versus exploitation), http://incompleteideas.net/book/the-book-2nd.html

**Keyed Progress (a Progress rule).** A sat-nav that knows you must collect a parcel before the delivery address: "3 km to go" counts the detour to the parcel first. `progress.KeyedRoute` measures the walk still to cover through the keys still needed, so standing at the red door without the key is worth nothing, and reaching the key is worth 24% of `E1M2`'s route. M4's rule, now called `doors-open`, counts every door as open (`CONTEXT.md`, **Progress rule**; ADR 0013). Which keys the player holds is Privileged Information, read only to measure (ADR 0002). Source: Dijkstra's shortest-path algorithm, which the distance field runs, https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

**Intrinsic reward: Random Network Distillation (RND).** A flashcard quiz: a fixed, randomly built network turns each screen into 512 numbers, and a second network keeps learning to guess them. On screens seen many times the guess is good; on a new screen it is bad, and the size of the miss is paid as a bonus, so new places pay until they become familiar (`CONTEXT.md`, **Exploration bonus**). "Intrinsic" means the reward comes from the agent's own curiosity, not from the game. Source: Burda, Edwards, Storkey and Klimov, 2018, https://arxiv.org/abs/1810.12894

**Normalising by a running spread.** Converting prices from different currencies before comparing them. The raw RND miss can be 1,000 on the first screens and 40 later; dividing it by the running spread of its own discounted sum keeps the coefficient meaning "how much the bonus weighs", whatever its scale. The catch M5 found: the bonus then stays the same size for the whole run, and the pilot that chose the coefficient was too short to show its true size. Source: the reward normalisation in CleanRL's `ppo_rnd_envpool.py`, https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/

**Ablation.** Baking the same cake twice, once without the sugar, to learn what the sugar does. Train once with RND and once without, every other setting equal, and compare on the same Eval Spec. With one training seed per arm, a difference can still be training-seed luck; the better arm is then repeated with more training seeds. Source: Meyes et al., 2019, "Ablation Studies in Artificial Neural Networks", https://arxiv.org/abs/1901.08644

**A reward that teaches the wrong thing.** A shop paying its staff per customer greeted, who then greet the same customer ten times. M3's `DeadlyCorridor` policy charged and died; M5's policies walk to a door they cannot open, because that is where the doors-open rule pays most. Watching films, not only reading scores, is how both were found. The fix on `E1M2` was not more exploration but a reward that pays the right route: `--shaping-rule keyed`. Source: Krakovna et al., "Specification gaming: the flip side of AI ingenuity", https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/

## 3. Reading order

1. `docs/milestones/img/m5-e1m2-route.png` and `m5-e1m2-misleads.png`, with the "keys, locked doors and route" note in `docs/milestones/M5-maps-with-keys.md`, Results. Look at the pictures before the code: they show what the code computes.
2. `src/doom_player/progress.py`:
   - lines 28-50: the line actions and thing types read from the WAD (locks by colour, keys by colour, lifts, remote doors)
   - lines 199-228, `_walls`: which lines block a player, and why a lift sector does not
   - lines 275-321, `distance_field`: `locked` and `goal`, and the cache
   - lines 98-165, `KeyedRoute`: read `remaining` slowly; the recursion over key sets is the whole idea
   - lines 167-197, `ProgressMeter`: why it refuses a spawn that cannot reach the exit
   - skip `_rasterize`, `_exit_sources`, `_pickup_sources` and `render` on first read
3. `src/doom_player/session.py`, lines 41-50 (the Progress rules, and the object names a key has) and 97-127 (the two meters, `for_spec`), then 171-175 (`progress_under`) and 226-250 (`_key_objects`, `_measure`, `_finish`): how keys held are read from the objects list and kept in the record.
4. `src/doom_player/maps.py`, lines 105-140, `ProgressShaping`: `progress_rule` lets the training reward read a different rule from the score.
5. `src/doom_player/eval.py`, lines 31-52 (`EvalSpec`, `e1m2-v1`, `MAP_SPECS`), 83-131 (`run_eval` and `transfer`), 132-145 (`check_rules`), 184-212 (`scoreboard_row`).
6. `src/doom_player/rnd.py`, the whole file (260 lines), in this order: the module docstring; `RND` (lines 61-141): `bonus` and `fit`; `RNDBonus` (lines 144-240): the class docstring, then `step_wait`, line by line; `RNDUpdate` (lines 243-260).
7. `src/doom_player/train.py`, lines 44-163 (`MapRules`, the new `TrainConfig` fields `tic_limit`, `shaping_rule` and `rnd_coef`, and their checks; `shaping_rule` is the one-line change that made `E1M2` Clear), then the `if config.rnd_coef:` block in `train`, the RND part of `save_checkpoint`, and `ProgressCurve` (lines 319-340), which logs `rollout/key_rate`.
8. `docs/adr/0013-progress-on-maps-with-keys.md`, then the brief's Open facts and Results.

Safe to skip: `src/doom_player/contenders/ppo.py`, `mcp_server.py` and `replay.py` (one line each: the Progress rule passed through), `src/doom_player/scoreboard.py` (the "(trained on `E1M1`)" label), `docs/milestones/make_figures_m5.py` (draws the pictures and exports the curves), `docs/adr/0008-progress-metric.md` (two bullets amended), `tests/` (read `tests/test_progress.py` after `progress.py` and `tests/test_rnd.py` after `rnd.py`: they state what each promises), `tests/conftest.py` (`logged_scalars`), `results/` (records, curves, failure categories), and the edits to `README.md` and `CONTEXT.md`.

## 4. Follow one Attempt

`uv run doom-eval --contender ppo --spec e1m2-v1 --checkpoint checkpoints/e1m2-d3-seed0-shaped/model.zip`, game seed 2: the one Attempt that picked up the red key.

1. `eval.main` builds `PPOMapContender(checkpoint)`, named `ppo-e1m2-d3-seed0-shaped`, and `run_eval` checks the records already there against `e1m2-v1`'s rules, Progress rule `keyed` included (`check_rules`).
2. `play_attempt` calls `make_map_env("E1M2", 3, 12600, progress_rule="keyed")`. `MapEnv.reset` builds an `AttemptSession` with seed 2. Its `__post_init__` makes two meters: `doors-open` (a `DistanceField`) and `keyed` (a `KeyedRoute`, whose `start_remaining` is 9,177 units). Since `E1M2` has a key, objects info is switched on, and `_key_objects` notes the red keycard's object id at the spawn.
3. Each decision: the policy reads 4 stacked 84x84 frames and picks one of 11 actions; `session.act` runs 4 tics, then `_measure` reads the position, compares the key objects with the spawn's (none missing yet, so `keys_held` is empty), and calls `visit` on both meters.
4. Early on the player reaches the red door. The `doors-open` meter is now at 20%; the `keyed` meter has not moved, because `KeyedRoute.remaining` without the key takes the route through the key, and the door is farther along that route than the spawn.
5. At decision 987 the red keycard's object leaves the list: `keys_held` becomes `{"red"}`. `remaining` now uses the field with the red door open, and the meter's closest distance drops to the key's 6,955 units: Progress 24.4%.
6. The player faces a wall beside the key, out of ammunition, until tic 12,600. `_finish` marks the Attempt `truncated`, keeps Progress 0.2436 and `keys_held: ["red"]`, and the record lands in `results/attempts/ppo-e1m2-d3-seed0-shaped/e1m2-v1.jsonl`.
7. Now run the same command with `checkpoints/e1m2-d3-seed0-keyed-shaped-10m/model.zip` and follow game seed 0 of the policy whose training reward paid keyed Progress. The red keycard leaves the objects list at decision 93, the player is past the red door by decision 222, and at decision 454 it presses the exit switch: the game ends without a death, `_finish` sets `cleared` (`terminated and not died`), and `progress_under` returns 1.0 for a Clear, at tic 1,816.

## 5. Try it

Predict before you run. The first four take seconds; the last two train.

1. In Python, build `KeyedRoute("E1M2")` and print `remaining(x, y)` and `remaining(x, y, frozenset({"red"}))` at the red key's spot (1136, 352). Predict whether the two are equal, and why the design needs them to be.
2. Print `distance_field("E1M2").distance_at(-700, 384)` and `KeyedRoute("E1M2").remaining(-700, 384)` (just before the red door). Predict which is larger and by how much, compared with the spawn's distance.
3. Open `results/curves/e1m2-d3-seed0-shaped-rnd.csv` and `e1m2-d3-seed0-shaped.csv`. Predict which one's `progress` column rises over training, then check.
4. Open `results/curves/e1m2-d3-seed0-keyed-shaped.csv` and `e1m2-d3-seed0-keyed-shaped-10m.csv`. Predict in which million steps `clear_rate` first rises above 0.5, then check.
5. Run the RND pilot again with `--rnd-coef 0.15` (200k steps, about 8 minutes, `--label pilot-rnd-c0p15`) and read `rnd/attempt_bonus` against `rnd/attempt_scaled_reward` in TensorBoard. Predict the ratio you will see.
6. Train M4's recipe on `E1M2` with `--death-penalty 0` (1M steps, about 30 minutes, `--label shaped-nodeath`). Predict whether the policy still walks to the red door first, and whether it lives longer.

## 6. Rebuild it

In `doom-player-study/`, write `m5_keys.py` by hand, without importing anything from this repository.

- **Build:** extend the M4 grid maze with a locked door cell and a key cell placed so that the key is farther from the exit than the start. Write the keyed distance: breadth-first search from the exit with the door shut, from the exit with it open, and from the key; the distance still to cover is the door-shut distance, or the walk to the key plus the key's door-open distance, whichever is shorter. Train the same tabular Q-learning agent three times: shaped with door-open distance, shaped with keyed distance, and with a count-based novelty bonus (1 / sqrt(visits) per cell) added to the door-open shaping.
- **Given:** your M4 maze and Q-learning code, the `KeyedRoute` docstring for the idea, not the code, and Burda et al. section 2 for why novelty bonuses help.
- **Check:** the keyed distance at the key is the same with or without the key held. Door-open shaping leads the agent to wait by the door; report how often each of the three agents reaches the exit over the same seeds.

## 7. Check yourself

1. Before M5, the Progress meter reported NaN on `E1M2`. What made the exit unreachable from the spawn, and why was the fix list of line actions kept closed?
2. Why does keyed Progress at the red door, without the key, come out at 0, while `doors-open` gives 20%?
3. Keys held are read from ViZDoom's objects list, not from how close the player came to the key. What could go wrong with closeness, given a policy that decides every 4 tics?
4. The RND bonus paid three times the shaped reward per Attempt, though a pilot suggested equal sizes. Name two reasons a 200k-step pilot can mislead about a 5M-step run.
5. With one training seed per arm, RND scored 0.2% and the baseline 9.2%. What can you conclude, what can you not, and what would you run next to find out?
6. The keyed-shaped policy scores 94.2% keyed Progress, and its training reward paid keyed Progress. Why is its Clear Rate the more honest number of the two?
