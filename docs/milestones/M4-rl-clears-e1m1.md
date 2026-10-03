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

- [ ] The training environment for Maps is decided and recorded in an ADR, and its observation matches what the Eval Suite shows a Contender
- [ ] A learned Driver Clears `E1M1` in at least one Attempt of a Map Eval Spec; its row is on the Map Scoreboard with Clear Rate, Progress, seeds, observation class and training cost
- [ ] The sparse-reward baseline and the shaped run are both recorded, with their curves in W&B
- [ ] The final setting is repeated with at least 3 training seeds
- [ ] Videos of the learned Driver exist for a Clear and for a failure, and failed Attempts are given failure categories
- [ ] Every privileged reward term is stated in Results
- [ ] `docs/learn/M4-rl-clears-e1m1.md` exists
- [ ] Post 3 material is drafted in `docs/posts/03-rl-clears-e1m1.md`
- [ ] The M5 brief is written

## Open facts to test

| # | Fact | Source of doubt |
|---|---|---|
| 1 | ViZDoom's `VizdoomDoomE1M1-S3-v0` loads `wads/doom.wad` and gives the same screen, frame skip and episode length as `AttemptSession` | The registration exists (checked 2026-10-03); which WAD it looks for, its time limit and its buttons are not checked |
| 2 | Steps per second on `E1M1` with 8 copies | M3 measured 450-500 on Scenarios; a full Map renders more and may be slower |
| 3 | PPO with exit-only reward makes no measurable Progress within a few million steps | Expected from the sparse-reward literature; not measured here |
| 4 | Progress gained is a shaping term the policy cannot game cheaply | M3's `DeadlyCorridor` paid distance and got "charge and die"; Progress counts new ground on the route, which may or may not behave better |
| 5 | `sb3-contrib`'s `RecurrentPPO` installs against SB3 2.9 and runs with the project's wrappers | Not tested in this project |
| 6 | How many steps a Clear needs | A hobby PPO + RND + LSTM agent Cleared `E1M2` in about 6M steps, 8 hours on an RTX 3080 with 20 environments (`research/2026-09-28-state-of-the-art-doom-agents.md`, R12); at M3's throughput 6M steps is about 3.5 hours |

## Rough size

6 to 10 sessions, most of it waiting on Training Runs of hours rather than minutes; open facts 2 and 6 decide the real number.

## Results

Filled in when the milestone is done.
