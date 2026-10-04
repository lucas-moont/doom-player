# M4 - RL Clears E1M1

## Goal

Train a learned Driver that Clears `E1M1` from screen pixels alone, measured by the same Eval Suite and Scoreboard as the LLM Contenders of M2. This is Post 3: the first original Map Cleared by a policy that decides in under a millisecond.

Written 2026-10-03, after M3. M3 showed that PPO learns Scenarios in 8 to 30 minutes on the RTX 4050, that three training seeds land in the same place, and that a reward which pays the wrong thing gets gamed: on `DeadlyCorridor` the first policy charged forward and died, and even with shaped reward it never learned to survive the second pair of guards. An original Map is harder on every count: one reward at the exit and none before it, doors and turns, and minutes of play instead of seconds.

## Concepts the owner learns

- Sparse reward, and reward shaping toward the exit with a term the policy cannot see
- Recurrent policies (LSTM): a memory of the last seconds, for a Map that does not fit on one screen
- Curriculum by Difficulty: learn on an easier skill level first, then raise it
- Reading behaviour, not only the score: videos and failure categories for a learned Driver, as M2 did for the LLM
- Training budgets measured in hours: checkpoints, resuming, and deciding when to stop

## Starting state

- Eval Suite with Map and Scenario Eval Specs, Scoreboard with both tables, W&B logging, `doom-train` and the PPO Contender (M1-M3). `e1m1-v1` is Difficulty 3, 6,300 tics (3 minutes), 5 seeds.
- On `E1M1`: random 0/5 (Progress 20%), Opus 5.5 H0 1/5, H1 4/5, H2 5/5.
- The M1 Progress meter measures how much of the walking route from spawn to exit an Attempt covered, from the Map's geometry. It is Privileged Information, available for reward and for scoring, never for observation.
- PPO trains at about 450-500 steps per second with 8 Scenario copies (M3, open fact 2); `RewardShaping` and its tests exist; the training curve stays on the environment's own reward.
- ViZDoom 1.3.1 registers the original Maps as Gymnasium environments, for example `VizdoomDoomE1M1-S1-v0` (checked 2026-10-03 in the installed package: 340 `VizdoomDoom*` and `VizdoomFreedoom*` IDs). Whether they load this project's `wads/doom.wad` is not tested.
- ADR 0010 leaves this choice to M4: a Gymnasium environment for Maps whose observations match `AttemptSession.observe`, or a Gymnasium face on the session itself.

## Steps

1. **A training environment for `E1M1`.** Decide between ViZDoom's `VizdoomDoomE1M1-S*` environments and a Gymnasium face on `AttemptSession`, so the policy trains on what the Eval Suite will show it (open fact 1). Record the decision in an ADR.
2. **The sparse baseline.** Train PPO on the exit reward alone for a fixed budget and record what it learns (expected: very little). This is the "before" for the shaping.
3. **Shaping toward the exit.** Add a training reward for Progress gained (new ground on the route to the exit), with a cost for dying, using the Progress meter's distance field. Watch videos and count behaviours from the first run on: M3 showed that a shaped reward can be gamed.
4. **Memory.** Try a recurrent policy (`RecurrentPPO` from `sb3-contrib`) against the frame-stacked CNN on the same budget.
5. **Curriculum by Difficulty.** If Difficulty 3 does not Clear, train on Difficulty 1 first, then continue on Difficulty 3. The same lever is the open candidate for surviving `DeadlyCorridor`.
6. **Measure through the Eval Suite.** The learned Driver plays `e1m1-v1` through `AttemptSession` and lands on the Map Scoreboard next to random and the LLM rungs. If it Clears only at a lower Difficulty, that is a new Eval Spec, named as such, and its `e1m1-v1` row is reported too.
7. Draft Post 3 material and write the study guide.

## Constraints

- A Contender sees Human-equivalent Observations only (ADR 0002). Progress, positions and the distance field may shape training reward; each such term is declared in Results.
- Every result goes through the Eval Suite with seeds and cost; cost is training time on the RTX 4050 plus evaluation time.
- Training Runs are logged to W&B, one run per Training Run, with the curve on the environment's own reward.
- Runs longer than two hours start detached inside WSL and are watched through the files they write.

## Completion criteria

- [x] The training environment for Maps is decided and recorded in an ADR, and its observation matches what the Eval Suite shows a Contender
- [x] A learned Driver Clears `E1M1` in at least one Attempt of a Map Eval Spec; its row is on the Map Scoreboard with Clear Rate, Progress, seeds, observation class and training cost
- [x] The sparse-reward baseline and the shaped run are both recorded, with their curves in W&B
- [x] The final setting is repeated with at least 3 training seeds
- [x] Videos of the learned Driver exist for a Clear and for a failure, and failed Attempts are given failure categories
- [x] Every privileged reward term is stated in Results
- [ ] `docs/learn/M4-rl-clears-e1m1.md` exists
- [ ] Post 3 material is drafted in `docs/posts/03-rl-clears-e1m1.md`
- [ ] The M5 brief is written

## Open facts to test

| # | Fact | Source of doubt |
|---|---|---|
| 1 | ViZDoom's `VizdoomDoomE1M1-S3-v0` loads `wads/doom.wad` and gives the same screen, frame skip and episode length as `AttemptSession` | The registration exists (checked 2026-10-03); which WAD it looks for, its time limit and its buttons are not checked. **Answered 2026-10-03**: it loads the WAD through the `doom_game_path` keyword, but differs from the session in time limit (126,000 tics), seeding, audio, the last frame (all zeros) and actions (19 free buttons). Not used: Maps train through a Gymnasium face of the session instead (ADR 0011) |
| 2 | Steps per second on `E1M1` with 8 copies | M3 measured 450-500 on Scenarios; a full Map renders more and may be slower. **Measured 2026-10-03**, with short test Training Runs of 16k steps on `MapEnv` (like a test drive before a long trip): 350 steps/s with 4 copies (3.5 GB of WSL RAM in use at peak), 512 with 8 (4.9 GB). 12 copies were not tried: each E1M1 copy takes about 350 MB, so 12 would pass 6 GB of WSL's 7.6, and a first attempt at the full probe was stopped when Windows ran low on memory. 8 is the setting; a million steps takes about 33 minutes |
| 3 | PPO with exit-only reward makes no measurable Progress within a few million steps | Expected from the sparse-reward literature; not measured here. **Supported 2026-10-03**, at 1M steps only: Progress 8.5% on `e1m1-v1`, below random's 20% (Results). The plan said about 1.5M steps; 1M was enough to show the policy learns nothing useful from the exit alone, so "a few million" was not tested |
| 4 | Progress gained is a shaping term the policy cannot game cheaply | M3's `DeadlyCorridor` paid distance and got "charge and die"; Progress counts new ground on the route, which may or may not behave better. **Held 2026-10-03**: no gaming seen in three training seeds. Progress pays only the best distance reached, so the shaped policies have nothing to farm; they take the route to the exit and Clear 15 of 15 Attempts. What they learned to leave out is fighting: they barely fire and run past monsters (Results) |
| 5 | `sb3-contrib`'s `RecurrentPPO` installs against SB3 2.9 and runs with the project's wrappers | Not tested in this project. **Confirmed 2026-10-03**: `sb3-contrib` 2.9.0 installs, and `RecurrentPPO` with `CnnLstmPolicy` trains and reloads on `MapEnv` (`tests/test_train_maps.py`). No Training Run used it: the frame-stacked CNN Cleared without memory |
| 6 | How many steps a Clear needs | A hobby PPO + RND + LSTM agent Cleared `E1M2` in about 6M steps, 8 hours on an RTX 3080 with 20 environments (`research/2026-09-28-state-of-the-art-doom-agents.md`, R12); at M3's throughput 6M steps is about 3.5 hours. **Measured 2026-10-03**, with Progress shaping on `E1M1` (which needs no keys, unlike `E1M2`): the first Clears during training came at about 765k and 1.02M steps (training seeds 1 and 2), a 1M-step checkpoint already Cleared 3 of 5 on `e1m1-v1`, and 5M steps (about 3 hours each) Cleared 15 of 15 over three training seeds |

## Rough size

6 to 10 sessions, most of it waiting on Training Runs of hours rather than minutes; open facts 2 and 6 decide the real number.

## Results

Filled in when the milestone is done. Notes so far:

- **Observation class**: `human-equivalent`. A learned Map Contender sees the screen with its HUD, in the policy view (gray, 84x84, last 4 frames), and chooses from the Action set of 11 button combinations (`maps.ACTIONS`, no weapon switching).
- **Filming changed the game, so scored Attempts are not filmed tic by tic** (found 2026-10-04). The first evaluations below were run with `--video`, which stepped the game one tic at a time. Re-run without it, every policy played different games: seed 0's shaped policy Cleared 5 of 5 instead of 4. The player's position stays identical either way, but the status bar face is redrawn differently (73 pixels around rows 221-236, columns 151-168, first at the 23rd decision of a test plan), and a policy that reads the screen, HUD included, then decides differently. The numbers below are all from unfilmed evaluations, the condition the policies trained in; the filmed Attempts are kept in `results/discarded/*-filmed.jsonl` and their videos in `videos/filmed-per-tic/`. A learned Map Contender is now filmed one frame per decision instead, which leaves its game unchanged (tested; the refilmed Attempts of all four policies match their Scoreboard records exactly). LLM Attempts in M2 were filmed tic by tic in every Attempt, so their records are the games they played.
- **Sparse baseline** (W&B run `an8j66gw`, exit reward only, Difficulty 3, 1,001,472 steps, 29 min, against about 1.5M in the plan): Progress 8.5% on `e1m1-v1` (7.6% to 9.0% by seed), no Clear, every Attempt out of time, against random's 20%. A checkpoint at 250k steps scored 14.1% (a note, measured outside the Scoreboard). The video of seed 1 shows the policy walking to the edge of the water near the start, picking up a health and an armour bonus, firing 33 bullets, then facing a wall for the rest of the 3 minutes. With nothing paid before the exit, the policy has nothing to learn from: open fact 3 holds.
- **Progress shaping works fast** (W&B run `oj834br1`, Difficulty 3, 100 for the whole route, 25 for dying). Halfway checkpoints, measured outside the Scoreboard as notes: 3 of 5 Cleared at 1M steps (37 min of training; the two failures die within 4% of the exit), 1/5 at 2M, 5/5 at 3M and 4/5 at 4M. The training curve on the Map's own reward stays at 0 until Clears appear, so it says nothing about learning before then; Progress got its own curve after this Training Run (see below).
- **First learned Clears of `E1M1`: three training seeds, same setting** (5M steps each, run one after the other; Windows slept for about 2 h during seed 2, which paused WSL and is not counted in its training time). Evaluated alone on the machine, without video:

  | Training seed | W&B run | Training time | `e1m1-v1` Clear Rate | Progress | Clear times (tics) | Cost per Attempt |
  |---|---|---|---|---|---|---|
  | 0 | `oj834br1` | 189 min | 5/5 | 100% | 716 to 768 | 0.8 s |
  | 1 | `7zv0jyx5` | 183 min | 5/5 | 100% | 704 to 780 | 0.9 s |
  | 2 | `ffj5shht` | 188 min | 5/5 | 100% | 708 to 1,012 | 0.9 s |

  **15 Clears in 15 Attempts**, each in about 20 s of game time (the fastest LLM Clear in M2 took 1,507 tics). The Scoreboard keeps each final checkpoint, as decided before the runs, not the best of the halfway ones. Seeds 1 and 2, the two trained after the Progress curve was added, follow the same shape: Progress during training climbs from about 15% to 88-90% within the first 300k steps, and the first training Clears appear at about 765k steps (seed 1) and 1.02M (seed 2). The videos show one route every time: start room, corridor, computer room, the path over the nukage, the exit door, the switch. The one slow Clear (seed 2, eval seed 2, 1,012 tics) takes the same route, then lingers in the exit room trading fire, health down to 56%, before pressing the switch. The policies barely fire (36 to 45 of 50 bullets left at the end): they learned to run the route and outpace the monsters. The near-identical Clear times show one route learned through this one Map; nothing here says it would find its way through another.
- **How training seeds were seeded.** SB3 seeds game copy *k* of a Training Run with `seed + k`, so with 8 copies, training seeds 0, 1 and 2 drew overlapping streams of games: copy *k* of seed 1 played the games of copy *k+1* of seed 0. The three Training Runs still differ in their starting weights and in the actions sampled while learning, which soon sends the same game down different paths, but they are less independent than three separate seeds suggest. The same held for M3's three `DefendCenter` seeds. From 2026-10-04, Training Runs space their copies' seeds 1,000 apart per training seed (`train.SEED_SPACING`); training seed 0 is unchanged.
- **Privileged reward terms**: `ProgressShaping` pays 100 times the Progress gained (the best walking distance to the exit so far, measured from the player's position and the Map's geometry) and charges 25 on death. Both read Privileged Information during training only; the policy sees the screen alone, and the Eval Suite scores Clear Rate and Progress as for every other Contender (ADR 0002).
- **Privileged debugging curve**: `train.ProgressCurve` logs each finished training Attempt's Progress and Clear to W&B (`rollout/progress`, `rollout/clear_rate`), for people to read how a Map Training Run is going, since the Map's own reward stays at 0 until the first Clear. It reads the same Privileged Information as the shaping and never reaches the policy. Added after training seed 0, so only seeds 1 and 2 have it; those two were trained from the working copy before the curve was committed, so their Scoreboard rows name the commit before it (`104e383`). The curve changes nothing the policy learns.
- **Failure categories**, as in M2, over every failed `e1m1-v1` Attempt of the M4 Contenders:

  | How it failed | Count | Where |
  |---|---|---|
  | Stuck near the start until the time ran out | 5 | sparse baseline, all five (Progress 7.6% to 9.0%; the video of seed 1 shows it facing a wall after the first room) |

  The shaped policies did not fail on `e1m1-v1`. Their halfway checkpoints and the filmed evaluations died within a few percent of the exit; the one filmed failure examined (training seed 0, eval seed 3) was **killed in combat near the exit**, H1's only failure category in M2: imps close in in the room before the exit door and the player dies wedged beside a candelabra.
- **Step 4 (Memory) was not run.** A recurrent policy (`RecurrentPPO`) trains in the tests but was not tried in a Training Run: the frame-stacked CNN, whose only memory is its last 4 frames, already Cleared 15 of 15, so there was no failure for memory to fix on `E1M1`. Memory is likelier to matter on Maps with keys and backtracking (M5).
