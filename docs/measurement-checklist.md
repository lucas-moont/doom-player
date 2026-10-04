# Measurement checklist

Checks every result passes before it reaches the Scoreboard or a post. Most come from a mistake this project actually made, with the milestone where it was found in brackets; the rest are safeguards or rules, marked as such. Like a pilot's pre-flight list: short, boring, and there because someone once skipped the step.

## Before training

- [ ] **Training games are not eval games.** Training draws game seeds from 1000 up, and each training seed's copies are spaced apart (`train.SEED_SPACING`), so neither the Eval Spec's games nor another training seed's games leak in. [M4]
- [ ] **The setting is new or labelled.** A Training Run never reuses a checkpoint folder; a different setting gets a `--label`. [safeguard, M4 review]

## During training

- [ ] **The curve can show learning.** If the game's own reward is 0 until success (an original Map), a second curve shows progress (`rollout/progress`), or the Training Run is blind until it succeeds. [M4]
- [ ] **The machine stays awake.** A sleeping Windows pauses WSL; training time stays correct but the Training Run takes longer. Turn sleep off for long Training Runs and back on after. [M4]

## Evaluating

- [ ] **Scored Attempts are not filmed tic by tic.** Filming that way redraws the status bar face, and a policy that reads the screen then plays another game. Film a learned Contender one frame per decision, or film separately and check the records match. [M4]
- [ ] **The machine is otherwise idle.** Cost per Attempt measured while a Training Run shares the machine is inflated; re-measure alone. [M3]
- [ ] **The checkpoint is the one decided before the Training Run** (usually the final one), not the best of several checkpoints or settings looked at afterwards. [M3, M4]
- [ ] **The number matches the behaviour.** Watch videos and count what the policy does (shots, kills, deaths, route) before calling a higher score better. A score can rise for the wrong reason. [M3: charge-and-die]

## Reporting

- [ ] **Every number in a doc traces to a file in `results/`** or is marked as a note measured outside the Scoreboard. [M3, M4]
- [ ] **Privileged Information is declared**: every reward term or curve computed from it is listed in Results. [rule, ADR 0002]
- [ ] **One seed is not a trend.** A gap smaller than the spread between eval seeds, on one training seed, is reported as a tie. [M3]
