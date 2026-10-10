# M6 - Learning by watching

## Goal

Find out whether recordings of the owner playing `E1M2` can stand in for the map knowledge M5 built into the training reward, and whether they make training faster. The question for M6 is "does a human demo help a learned Driver?", answered on the same Eval Spec as M5.

Written 2026-10-10, after M5. M5 Cleared `E1M2` only once the training reward paid the route through the red key (`keyed` shaping): 8 Clears in 15 Attempts over three training seeds, at 10M steps and about 4.8 hours each. M4's reward, which counts every door as open, Cleared none. That reward walked every policy to the locked door. RND, an exploration bonus, made it worse. Two things M5 left open lead here.
- The keyed reward is hand-built from the WAD. It knows where the key is, which a player new to the Map does not.
- One keyed-shaping training seed of three never Cleared. It circles a room past the red door until it is killed.

A demo carries the same knowledge, the key first and then the door, in the form a human would pass it on.

## Concepts the owner learns

- **Behavioural cloning (BC)**: training a policy to copy recorded actions, like learning a dance by copying a video frame by frame. It is supervised learning on (screen, action) pairs. Why its mistakes compound, once the policy drifts to screens the demo never showed.
- **Fine-tuning**: starting PPO from the cloned policy's weights instead of random ones. Why the copied skill can be forgotten in the first updates.
- **Learning from demo states (the "backward algorithm")**: training Attempts start from states along one recorded Clear. The start moves back toward the spawn as the policy succeeds, like learning to park by starting almost inside the space.
- **Recording human play**: spectator mode, mapping a person's key presses onto the policy's 11 actions, and keeping demos apart from the Eval Spec's seeds.
- **Sample efficiency**: measuring "faster" as training steps to the first training Clear and Clear Rate at a fixed budget, not as wall-clock time alone.

## Starting state

- **Tools from M4 and M5:**
  - `MapEnv`, `ProgressShaping` with `--shaping-rule doors-open | keyed`, `PPOMapContender`, and `doom-train --map` with `--init-from`, `--tic-limit` and `--rnd-coef`;
  - `ProgressCurve` with `rollout/key_rate`;
  - the Map Scoreboard with training cost.
  - Continued Training Runs keep their parent's reward scale and record `init_from`.
- **`e1m2-v1`:** `E1M2`, Difficulty 3, game seeds 0 to 4, 12,600 tics, `keyed` Progress.
- **Scores on `e1m2-v1`:**
  - random 0/5;
  - the doors-open recipe 0/15 over three training seeds;
  - keyed shaping at 10M steps 4/5, 4/5 and 0/5 by training seed;
  - keyed shaping's first training Clears at 4.77M steps (seed 0), common from about 8M (seed 1, after one lone Clear at 5.2M) and never in 10M (seed 2).
- **Training speed:** 5M steps take about 2.4 hours with 8 copies on the RTX 4050, when the machine is idle.
- **ViZDoom 1.3.1 tools for demos and states:**
  - `Mode.SPECTATOR`, for a person to play while the program watches;
  - `new_episode(recording_file_path)` and `replay_episode()`, to record and replay `.lmp` demos;
  - `get_last_action()`, the buttons pressed during the last tic;
  - `DoomGame.save` / `load`, to store and restore a game state.
  - All of these were checked to exist in the installed package on 2026-10-07 and 2026-10-10; none has been tried here.
- **Sources:**
  - The research note lists BC work in ViZDoom: Spick et al. 2024, R45; and R15.
  - The idea of starting from demo states comes from Salimans and Chen, 2018, "Learning Montezuma's Revenge from a Single Demonstration" (open fact 5).

## Steps

1. **Record demos.** A recorder lets the owner play `E1M2` at Difficulty 3 in spectator mode, keyboard only. Each tic, it maps the keys pressed onto the nearest of the 11 actions, and stores the policy view (4 stacked 84x84 frames) every 4 tics with the action. The Attempt's record and a `.lmp` replay are saved beside it.
   - Demos use game seeds outside `e1m2-v1`'s (100 upward), so no demo shows an evaluated game.
   - Target: about 10 Clears, recorded across one or two sessions; failed recordings are kept but not trained on.
   - Record the owner's own Clear Rate and time on those recordings, as a reference.
   - Open facts 1 and 2.
2. **BC alone.** Train the M5 network (`CnnPolicy`, the same 11 actions) to copy the demos with a maintained library, and evaluate it on `e1m2-v1` (open fact 3). This tests whether copying alone gets past the red door.
3. **BC, then PPO, with the reward that failed.** Start PPO from the BC weights with M4's doors-open shaping, M5's settings otherwise, 5M steps. If this Clears, the demo replaced the map knowledge in the reward.
4. **BC, then PPO, with keyed shaping.** The same, with `keyed` shaping, compared against M5's three keyed seeds. Compare steps to the first training Clear and the Clear Rate at 5M and 10M steps. This answers "faster".
5. **Learning from demo states, if steps 3 and 4 do not Clear.** Training Attempts start from saved states along one recorded Clear, moving back toward the spawn as the policy Clears from each (open fact 4). Eval Attempts always start at the spawn.
6. **Measure through the Eval Suite.**
   - Every setting goes on `e1m2-v1`, and the best one gets 3 training seeds.
   - Videos and failure categories, as in M5.
   - Check whether keyed training seed 2's circling recurs.
7. Draft Post 5 material and write the study guide.

## Constraints

- **Observations:** a Contender sees Human-equivalent Observations only (ADR 0002). Demos are recorded from the same policy view the policy sees.
  - Restoring a saved game state is a training-time aid that a player does not have, like reward shaping. Each use is declared in Results.
- **Seeds:** demos never use an Eval Spec's game seeds. Training seeds use spaced game seeds (`train.SEED_SPACING`).
- **Eval Suite:** every result goes through it with seeds and cost. Cost includes the owner's time recording demos, stated next to the training time.
- **W&B logging:** Training Runs are logged there, with the Map's own reward and the Progress curve. A BC run logs its copying loss and its held-out action accuracy.
- **Measurement checklist:** every result passes `docs/measurement-checklist.md` before it reaches the Scoreboard.

## Completion criteria

- [ ] A demo set of `E1M2` exists, recorded on game seeds disjoint from `e1m2-v1`'s, with its size, the owner's Clear Rate and the time it took
- [ ] BC alone is measured on `e1m2-v1`
- [ ] BC-initialised PPO is trained with doors-open and with keyed shaping, everything else as in M5, and both rows are on the Map Scoreboard
- [ ] Steps to the first training Clear and the Clear Rate at a fixed budget are compared with M5's keyed seeds
- [ ] The best setting is repeated with at least 3 training seeds
- [ ] Videos of a Clear (if any) and a failure exist, and every failed Attempt has a failure category
- [ ] Every training-time aid (shaping terms, restored states) is stated in Results
- [ ] `docs/learn/M6-learning-by-watching.md` exists
- [ ] Post 5 material is drafted in `docs/posts/05-learning-by-watching.md`
- [ ] The M7 brief is written

## Open facts to test

| # | Fact | Source of doubt |
|---|---|---|
| 1 | Spectator mode opens a playable window under WSL2 (WSLg) and records keyboard play at full speed | **(not verified)**: every game so far ran without a window or was filmed afterwards |
| 2 | A person's key presses map onto the 11 actions without losing what matters (for example turning while strafing, or the run key) | **(not verified)**: the action set was chosen for a policy, not for a player; a person presses several keys at once and changes them between the policy's 4-tic decisions |
| 3 | A maintained library does BC on an SB3 2.9 policy with this project's Gymnasium version | **(not verified)**: the `imitation` library is the usual choice; its support for SB3 2.9 is not checked |
| 4 | `DoomGame.save` / `load` restores a mid-Map state exactly, monsters and keys held included, and fast enough to start Training Attempts from | The methods exist in ViZDoom 1.3.1 (checked 2026-10-07); behaviour and speed not tested |
| 5 | One demonstration is enough for the backward algorithm on a hard-exploration game | Salimans and Chen 2018 report it on Montezuma's Revenge (not verified here, and its arXiv number still to check before citing); a Super Mario World write-up seen by the owner reports 0% to 85% on one level of three with one demo each |
| 6 | Keyed training seed 2 circles because of how it learned, not because the room cannot be passed | **(inference, untested)**: seeds 0 and 1 pass the same room in every Attempt, in about 55 decisions |

## Rough size

6 to 10 sessions, plus one or two sessions of the owner playing. Training Runs are 2.5 to 5 hours each. Open facts 1 to 3 decide whether step 1 takes one session or three.

## Results

Not started.
