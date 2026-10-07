# M5 - Maps with keys

## Goal

Train a learned Driver that Clears `E1M2`, a Map whose route is not a straight walk to the exit, and measure what an exploration bonus adds by training with and without it. This is the step from "learn one route" to "find your way".

Written 2026-10-04, after M4. M4 showed that Progress shaping makes `E1M1` easy: three training seeds Cleared 15 of 15 in about 3 hours each. It also showed the limit: every Clear takes the same route, and the Progress meter treats locked doors as open (ADR 0008), so on a Map with keys it would pay the policy for walking toward a door it cannot yet open.

## Concepts the owner learns

- Exploration, and why a reward for "closer to the exit" fails when the way to the exit is a detour
- Intrinsic reward: Random Network Distillation (RND), a bonus for seeing something new
- Ablation: training with and without one ingredient, everything else equal, to measure what it adds
- Recurrent policies (LSTM), if remembering where the key was turns out to matter
- Reading a Map's structure: keys, locked doors, and which ones block the exit

## Starting state

- `MapEnv`, `ProgressShaping`, `PPOMapContender`, `doom-train --map`, the Map Scoreboard with training cost, `ProgressCurve`, intermediate checkpoints, `--init-from` and `--recurrent` (M4). Full 5M-step Training Runs averaged 440 to 456 steps per second with 8 copies, using 4.9 GB of WSL RAM.
- On `E1M1` (`e1m1-v1`): PPO with Progress shaping 15/15, Opus 5.5 H2 5/5, random 0/5.
- The Progress meter's distance field (`progress.py`) treats every door as open, locked ones included.
- A hobby PPO + RND + LSTM agent is reported to Clear `E1M2` in about 6M steps, 8 hours on an RTX 3080 (`research/2026-09-28-state-of-the-art-doom-agents.md`, R12).
- `sb3-contrib` has no RND; it would be written here or taken from a library (open fact 3).

## Steps

1. **Map `E1M2`'s route.** Which keys and doors stand between the start and the exit, and where Progress as defined today would mislead (open fact 1). Add an Eval Spec `e1m2-v1` (Difficulty 3, 5 seeds, a time limit chosen from the route's length) and measure random and the `E1M1` policies on it.
2. **Baseline: M4's recipe on `E1M2`.** Progress shaping alone, 5M steps. Watch where it gets stuck.
3. **Exploration bonus.** Add RND as an intrinsic reward and train with it, everything else equal: the ablation. Decide, when its first test is written, whether the bonus is a wrapper next to `ProgressShaping` or part of the training loop (open fact 3).
4. **Memory, if needed.** If failures show the policy forgetting where it has been, try `--recurrent`.
5. **Measure through the Eval Suite.** Both settings on `e1m2-v1`, 3 training seeds for the better one, videos, failure categories.
6. Draft Post 4 material and write the study guide.

## Constraints

- A Contender sees Human-equivalent Observations only (ADR 0002). Positions, keys held and the distance field may shape training reward or draw debugging curves; each use is declared in Results.
- Every result goes through the Eval Suite with seeds and cost; scored Attempts are not filmed tic by tic (ADR 0011).
- Training Runs are logged to W&B, with the Map's own reward and the Progress curve.
- Training seeds use spaced game seeds (`train.SEED_SPACING`), so each training seed's games are its own.
- Every result passes `docs/measurement-checklist.md` before it reaches the Scoreboard.

## Completion criteria

- [x] `E1M2`'s keys, locked doors and route are recorded, with where Progress misleads
- [x] An Eval Spec for `E1M2` exists, with random and the `E1M1` policies measured on it
- [x] M4's recipe and the exploration bonus are both trained on `E1M2` with the same settings otherwise, and both rows are on the Map Scoreboard
- [x] The better setting is repeated with at least 3 training seeds
- [x] Videos of a Clear (if any) and a failure exist, and every failed Attempt has a failure category
- [x] Every privileged reward term is stated in Results
- [ ] `docs/learn/M5-maps-with-keys.md` exists
- [ ] Post 4 material is drafted in `docs/posts/04-maps-with-keys.md`
- [ ] The M6 brief is written

## Open facts to test

| # | Fact | Source of doubt |
|---|---|---|
| 1 | `E1M2` needs at least one key to reach the exit, and Progress as defined rewards walking toward a locked door | **(not verified)**: from memory of the Map, not checked against the WAD; ADR 0008 says locked doors count as open. **Confirmed 2026-10-07**, from the WAD: one key, the red keycard, and with the red door shut the exit is unreachable from the spawn. Progress as defined could not even measure `E1M2` (it read the lifts as walls, see Results); once it could, it paid up to 20.1% for reaching the red door without the key, and nothing for the detour to the key |
| 2 | The `E1M1` policies Clear nothing on `E1M2` | Expected, since they learned one route; not measured. **Confirmed 2026-10-07**: 0 Clears in 15 Attempts, Progress 0.6% to 4.2% by policy, around random's 2.3%; none picked up the red key (Results) |
| 3 | RND can be added with a small amount of code on top of SB3's PPO, or comes from a maintained library | `sb3-contrib` 2.9 has no RND; CleanRL has `ppo_rnd_envpool.py` (research note); not checked against this project's setup. **Answered 2026-10-07**: a small amount of code, written here after CleanRL's reference; no maintained library fits (Results) |
| 4 | Steps and time a Clear of `E1M2` needs | R12 reports about 6M steps with RND and an LSTM; at M4's speed that is about 3.7 hours per training seed |
| 5 | A key-aware Progress (distance through the doors actually openable with the keys held) is computable from the WAD | The distance field already reads the WAD's lines; whether the lock type of each door is available there is not checked. **Answered 2026-10-07**: yes. Each door's lock is its line's action number (28 is "red key, opens and closes"), and the keys are things with their own types and positions. `progress.KeyedRoute` measures the route through the keys; which keys the player holds is read from ViZDoom's list of objects, since ViZDoom has no game variable for keys (ADR 0013) |

## Rough size

6 to 10 sessions, most of it waiting on Training Runs of 3 to 4 hours; open facts 1 and 3 decide the real number.

## Results

In progress. The notes below are in the order they were measured.

- **Progress could not measure `E1M2`** (found 2026-10-07). The meter read lifts and doors opened by a switch as walls, so the exit room, behind a lift (sector tag 13), was unreachable from the spawn: Progress came out as NaN. Lifts (line actions 62 and 88) and switch- or shot-opened doors (103 and 46) are now passable. The list is closed on purpose: adding action 63, which `E1M1` uses, would change `E1M1`'s field, and a test pins that field to the one every `E1M1` result was measured with.
- **`E1M2`'s keys, locked doors and route**, read from the WAD (`docs/milestones/make_figures_m5.py` draws both pictures):

  | What | Where (map units) | WAD |
  |---|---|---|
  | Spawn | (-32, -240), facing north | thing type 1 |
  | Red keycard, the Map's only key | (1136, 352), east of the spawn, at every Difficulty | thing type 13 |
  | Red door, the only way to the exit | about (-728, 384), west of the spawn | linedefs 527 and 528, action 28 |
  | Exit switch | (-256, 2304 to 2368), north-west | linedef 873, action 11 |
  | Lifts and remote doors on the route | | lifts on tags 1, 4, 10, 11, 13; switch doors on tags 5, 7, 12; a shot-opened door on tag 6 |

  With every door open, the walk from the spawn to the exit is 4,959 units. Through the red key it is 9,177: 2,222 east to the key, then 6,955 back west through the red door and north to the exit. The key lies farther from the exit than the spawn does.

  ![E1M2 route](img/m5-e1m2-route.png)

  *Walking distance to the exit with every door open (bright = near), and the keyed route in cyan. Rings: spawn green, red key red, red door yellow.*

- **Where the doors-open rule misleads.** Before the red key, the doors-open rule pays for any ground closer to the exit than the spawn: the magenta area, which ends at the red door. There it has paid 20.1% of its Progress for a place the player cannot pass. The walk to the key pays nothing at all, since the key is 40% farther from the exit than the spawn, and after the key the rule pays again only once the player is back past the red door. A policy trained on it (M4's recipe) is paid to go to the door and stay there.

  ![Where doors-open Progress misleads on E1M2](img/m5-e1m2-misleads.png)

- **`e1m2-v1`**, chosen 2026-10-07 before any Attempt on it: `E1M2`, Difficulty 3, seeds 0 to 4, the `keyed` Progress rule, and a tic limit of 12,600 (6 minutes). `e1m1-v1` gives 6,300 tics to a 4,761-unit route, 1.32 tics per unit; the same allowance for `E1M2`'s 9,177-unit keyed route is 12,143 tics, rounded up to whole minutes. Six times the par time, which also gives `E1M1`'s limit, would have given 15,750 (par 75 s); the route length is the measure this project already uses. A learned Driver decides every 4 tics, so an Attempt has up to 3,150 decisions.
- **Keyed Progress** (`progress.KeyedRoute`, ADR 0013) measures the walk still to cover through the keys the player still needs, so on `E1M2` reaching the red key is worth 24.2% and standing at the red door without it is worth 0. On a Map without keys it equals the doors-open rule (tested on `E1M1`). Which keys the player holds comes from ViZDoom's list of objects: a picked-up key leaves the list (tested by walking into the red key), reading it costs no measurable speed, and it does not change the game (tested). The position and the keys held are Privileged Information, read for measuring only.
- **Random on `e1m2-v1`**: Clear Rate 0, Progress 2.3% (0.3% to 4.9% by seed), against 20% on `e1m1-v1`. It died in all five Attempts, after 668 to 3,336 tics: `E1M2` puts armed zombies near the spawn, where `E1M1` gave random three minutes of wandering. No Attempt picked up the red key.
- **The `E1M1` policies on `e1m2-v1`**, measured with `doom-eval --transfer` (their rows say "trained on `E1M1`"), alone on the machine, without video:

  | Policy | Clear Rate | Progress | How the Attempts ended |
  |---|---|---|---|
  | `ppo-e1m1-d3-seed0-shaped` | 0/5 | 0.6% | died in all five, after 1,356 to 3,072 tics |
  | `ppo-e1m1-d3-seed1-shaped` | 0/5 | 2.0% | out of time in four, pressed against the same wall near the spawn for 6 minutes; died in one |
  | `ppo-e1m1-d3-seed2-shaped` | 0/5 | 4.2% | died in all five, after 1,128 to 4,672 tics |

  None picked up the red key. Three Attempts were filmed afterwards, one frame per decision, and each film matches its scored record: seed 1's policy walks a few steps and pushes into a wall until the time runs out; seed 0's turns into a dark side passage and faces its wall while zombies shoot it dead; seed 2's picks up the green armour near the start, then stalls against walls and dies at 10% health. The `E1M1` route is no use here: the policies learned one Map, as M4 suspected. Open fact 2 holds.
- **How RND joins SB3's PPO** (open fact 3, decided 2026-10-07 before writing any of it). RND, Random Network Distillation (Burda et al. 2018), works like a flashcard quiz: a fixed, randomly built network turns each screen into a list of numbers, and a second network keeps learning to guess those numbers. On screens seen many times its guess is good; on a new screen it is bad, and the size of the miss is paid as a bonus, so new places pay until they become familiar.
  - **Options read.** `sb3-contrib` 2.9 has no RND. `rllte-core`, the one library found that offers RND for SB3, last released in May 2024 and pins old versions of several dependencies (`huggingface-hub==0.14.1`, `matplotlib==3.6.0`), so it would not install next to this project's. CleanRL's `ppo_rnd_envpool.py` is a single-file reference, written for its own PPO loop and EnvPool, not a library to import.
  - **Choice: a VecEnv wrapper written here, after CleanRL** (`src/doom_player/rnd.py`). SB3 computes the advantages of a rollout before any callback sees it, so a bonus added from a callback would arrive too late; the bonus is added to the reward as each step happens, by a wrapper around the batch of game copies, in the main process (one network for all 8 copies). A callback trains the guessing network once per rollout. The wrapper sits outside `VecNormalize`, which rescales the game's reward, so that rescaling means the same in both arms of the ablation and each stream is scaled on its own, as in the paper.
  - **Kept from CleanRL**: the two networks (three convolutions, then 512 numbers), the last frame of the policy view as input, each pixel normalised by a running mean and spread and clipped to ±5, the bonus scaled by the spread of its own discounted sum (discount 0.99), and the guessing network trained with Adam at a learning rate of 1e-4. Adam is the usual way to nudge a network's weights after each batch, like a hiker who adjusts each stride to how steep the last few steps were; the learning rate sets how big the strides are.
  - **Simplified, to be declared with the result**: one value head instead of the paper's two. A value head is the part of the policy network that predicts how much reward is still to come, like a running forecast of a match's final score; PPO learns from the gap between that forecast and what happens. The paper forecasts the game's reward and the bonus separately; SB3's PPO has one forecast, so the bonus is added to the reward; the bonus's normalisation updated every step rather than once per rollout; no bonus paid during the first rollout, while the pixel statistics settle (the bonus's own scale starts after it, as in CleanRL), instead of CleanRL's 50 rollouts of random play; every collected frame trains the guessing network (CleanRL uses a quarter, with 128 copies; here there are 8).
  - **Human-equivalent**: the bonus reads only the screen the policy sees. It is still a training-only reward term and is declared with the results.
- **M4's recipe on `E1M2`** (`ppo-e1m2-d3-seed0-shaped`, W&B run `9a3e5i8r`): PPO with doors-open Progress shaping (`--progress-reward 100 --death-penalty 25`), training seed 0, 5M steps in 157 minutes, scored with the final checkpoint as decided before training. On `e1m2-v1`: Clear Rate 0/5, keyed Progress 9.2% (0.3% to 24.4% by seed), against random's 2.3%. Three Attempts died, after 1,668 to 2,720 tics; two ran out of time. The training curve (`results/curves/e1m2-d3-seed0-shaped.csv`) never shows a Clear; the keyed Progress of training Attempts averages 1% to 4% per million steps. Where it stalls, from the five Attempts filmed afterwards (one frame per decision, each film matching its scored record) and their positions, read for debugging:
  - **It learned the trap the doors-open rule set.** All five Attempts reach the red door without the key: 19.6% to 20.0% doors-open Progress, of the 20.1% the rule pays there. Keyed Progress pays nothing for that.
  - **Then it wanders the central room and fires until its ammunition runs out.** Seed 0 spends the second half of the Attempt in a small alcove on the room's north side, fists out, facing a wall; seeds 1, 3 and 4 die in the same room.
  - **Seed 2 picked up the red key**, by wandering east, about a third of the way into the Attempt (keyed Progress 24.4%). It then faced a wall beside the key, out of ammunition, until the time ran out. The training reward pays nothing for that detour (the key is farther from the exit than the spawn), so nothing taught it what the key is for.
  - **Measurement checklist**: the Training Run started from commit `06e6d1d` and ran while RND's code and fast tests were written (and a few slow tests ran); the eval and the filming ran alone on the machine. The run started before `rollout/key_rate` existed, so its curve has no key rate. Privileged Information: the doors-open Progress pays the training reward; keyed Progress and keys held measure the score; positions drew the debugging paths.
- **RND's coefficient, from a pilot** (decided 2026-10-07, before the ablation). A coefficient sets how much the novelty bonus weighs against the reward it is added to, like the volume knob on one of two speakers. Pilot: `E1M2`, training seed 0, the baseline's settings plus `--rnd-coef 0.5`, 200k steps (label `pilot-rnd-c0p5`, W&B run `62t3hojo`; a pilot, not a result). By its end the bonus paid about 1.6 to 1.8 per Attempt, and the reward it is added to (shaping and the death penalty, as `VecNormalize` rescales it) moved between about -2 and +0.5 per Attempt: the two are the same size, so neither drowns the other. The raw bonus fell from about 1,060 per frame to 41 as the guessing network learned, and will keep falling as screens become familiar. The ablation keeps 0.5. RND costs speed: the pilot ran at about 430 steps per second, against the baseline's 530, so the ablation takes about 3.2 hours.
- **The ablation: M4's recipe plus RND** (`ppo-e1m2-d3-seed0-shaped-rnd`, W&B run `uqsc8q3g`): the baseline's command plus `--rnd-coef 0.5`, nothing else changed, 5M steps in 207 minutes. On `e1m2-v1`: Clear Rate 0/5, keyed Progress 0.2% (0.1% to 0.3% by seed), below the baseline's 9.2% and random's 2.3%. It died in all five Attempts, after 1,412 to 4,912 tics, and none picked up the red key.

  | Setting | Clear Rate | Keyed Progress | Red key picked up | How the Attempts ended |
  |---|---|---|---|---|
  | M4's recipe (`shaped`) | 0/5 | 9.2% | 1 of 5 | died in three, out of time in two |
  | M4's recipe + RND (`shaped-rnd`) | 0/5 | 0.2% | 0 of 5 | died in all five |

  - **Where it stalls**, from the five Attempts filmed afterwards (each film matching its scored record) and their positions, read for debugging: every Attempt walks to the red door (19.8% to 20.0% doors-open Progress, the same trap as the baseline), then, within about 85 decisions, into the alcove on the central room's north side that holds the green armour. It picks the armour up and stays there, 78% to 94% of each Attempt, facing the walls while zombies shoot it, until it dies. The baseline also ended two Attempts in that alcove, but spent 6% to 41% of the other three there and walked out; one of those walks found the key.
  - **During training** (`results/curves/e1m2-d3-seed0-shaped-rnd.csv`) the keyed Progress of training Attempts fell from 2.9% in the first million steps to 1.5% in the last, while the baseline's rose from 1.7% to 4.4%. Training Attempts picked up the red key in 12 of 2,428 rollouts, never more than a third of a rollout's Attempts.
  - **The bonus was bigger than the pilot said.** Over the whole run RND paid about 3.5 per Attempt, against a rescaled shaped reward of about -1.1 per Attempt: three times the size, not the same size. The pilot's 200k steps saw a bonus still settling. The per-Attempt bonus stayed flat for 5M steps because it is divided by the spread of its own sum: the raw bonus falls, and so does its scale. Why the bonus led to the alcove is not measured: a screen that flashes red when the player is hit, and the armour pickup, are new screens each time, a candidate known as the "noisy TV" problem **(inference, untested)**.
  - **What the ablation can say**: with one training seed per arm, RND at coefficient 0.5 did not help this recipe on `E1M2`; the run that had it ended worse on every measure. One seed cannot separate RND's effect from training-seed luck, and the coefficient was set from a pilot that underestimated the bonus by about three times. The baseline is the better setting, and step 5 repeats it with more training seeds.
  - **Measurement checklist**: the run started from commit `a66008e`. From about 10:30 to 11:30 the owner's work shared the machine and the run slowed to about 270 steps per second, against about 430 before and after, so its 207 minutes overstate RND's cost by about 20 minutes. The eval ran alone on the machine; the filming ran after it. Privileged Information as in the baseline; the RND bonus reads only the screen the policy sees.
- **No recurrent policy** (step 4, decided 2026-10-07 from the ten films above). A recurrent policy carries a memory from one decision to the next, like a hiker who remembers which paths were already tried; it helps when a policy loops back to places it has been. Neither policy does that: both walk straight to the red door, which the training reward pays for, and then stall in one spot, facing a wall or in the armour's alcove. Those are failures of what the reward asks for, not of memory, so `--recurrent` was not trained. M4 expected memory to matter more on Maps with keys; on `E1M2` the reward fails first, and memory may matter once a policy reaches the key and must walk back.
- **M4's recipe with three training seeds** (step 5): the baseline, the better setting, repeated with training seeds 1 and 2 (W&B runs `9ikt5jpx` and `c4qulqd2`), the same command but the seed, 5M steps each, run one after the other. Seed 1 shared the machine with the owner's work for its first two hours or so; seed 2 had it to itself, and both ran faster than seed 0 (about 600 against 530 steps per second), so their training times are not inflated. Each was evaluated alone on the machine and filmed afterwards; all ten films match their scored records.

  | Training seed | Clear Rate | Keyed Progress | Red key picked up | Training |
  |---|---|---|---|---|
  | 0 | 0/5 | 9.2% (0.3% to 24.4%) | 1 of 5 | 157 min |
  | 1 | 0/5 | 1.5% (0.5% to 4.1%) | 0 of 5 | 140 min |
  | 2 | 0/5 | 2.0% (0.4% to 4.8%) | 0 of 5 | 134 min |

  Seed 0's 9.2% rests on one Attempt that found the key; without it, the three policies score between 0.3% and 4.8% per Attempt, around random's 2.3%. During training, seed 1's Attempts picked up the key in 2 of 2,428 rollouts and seed 2's in none; neither showed a Clear. All fifteen Attempts walk to the red door first, within 35 to 114 decisions. M4's recipe does not Clear `E1M2`: across three training seeds it learns the trap its reward sets, and no more.
- **Failure categories** of the twenty failed Attempts of the four M5 policies (three training seeds of M4's recipe, and the RND ablation), each given by watching its film, with the keys held and how the Attempt ended checked against the scored record (`results/failures/e1m2-v1.csv`). Every one of them walked to the red door first.

  | Failure category | Count | Which |
  |---|---|---|
  | Killed while wandering without the key | 9 | recipe seed 0 (2), seed 1 (3), seed 2 (4): after the red door they roam the central rooms and corridors, firing, until zombies and shotgun guys kill them |
  | Killed in the armour's alcove | 6 | RND, all five; recipe seed 0 (1): they pick up the green armour on the central room's north side and stay in its alcove facing the walls while they are shot |
  | Facing a wall without the key until the time ran out | 4 | recipe seed 0 (1), seed 1 (2), seed 2 (1): pressed against one wall for the rest of the 6 minutes, in the alcove, a dark corner west of the spawn, or a corridor wall |
  | Had the red key, stalled beside it until the time ran out | 1 | recipe seed 0: out of ammunition, facing a wall next to where the key lay |

  No Attempt reached the red door with the key, so "had the key, did not reach the door" in the brief's plan has one member and "opened the door" none. Videos of a failure for each policy are kept locally in `videos/` (gitignored), one frame per decision; there is no Clear to film.
- **Privileged Information used in M5**, all of it outside what a Contender sees (ADR 0002). Every policy saw only the screen, in the policy view, with its HUD.
  - **Training reward**: the player's position, through the doors-open distance field, pays M4's Progress shaping in every Training Run above. A run with `--shaping-rule keyed` pays keyed Progress instead, which also reads which keys the player holds.
  - **Scoring**: the position and the keys held measure keyed Progress on `e1m2-v1`, and the keys held fill each record's `keys_held`.
  - **Training curves**: `rollout/progress` (keyed) and `rollout/key_rate` read the same two.
  - **Debugging**: positions traced the paths in the films above. They were read while filming only, after each eval, and drew no committed figure.
  - **Not privileged**: the RND bonus reads only the newest frame of the policy view.
