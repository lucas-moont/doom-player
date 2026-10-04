# Doom Player

The language of a project that builds and compares agents playing the original Doom. Doom, reinforcement learning, and LLM tooling each bring their own meaning for the same words; this glossary picks one.

## Language

### The game

**Map**:
One original Doom level, named like `E1M1`.
_Avoid_: level, stage, scenario

**Scenario**:
A small purpose-built ViZDoom training level, such as `Basic` or `DefendCenter`. Never an original Map.
_Avoid_: mini-map, toy level

**Doom Episode**:
A set of nine Maps sold as one chapter of the game, such as Knee-Deep in the Dead (`E1M1` to `E1M9`). Always written with "Doom" in front.
_Avoid_: episode (alone), chapter

**Original Doom Episodes**:
The three Doom Episodes released in 1993: `E1`, `E2` and `E3`, 27 Maps in total.
_Avoid_: the full game, all episodes

**Bonus Doom Episode**:
The fourth Doom Episode, Thy Flesh Consumed (`E4M1` to `E4M9`), added by The Ultimate Doom in 1995. Outside the scope of Beat the game.
_Avoid_: episode 4 (alone), final episode

**Difficulty**:
The game's 1-to-5 challenge setting, which ViZDoom calls `skill`.
_Avoid_: skill, skill level

**WAD**:
The data file holding the game's Maps and assets. `doom.wad` is the purchased original, here The Ultimate Doom with 36 Maps; Freedoom is the free substitute with different Maps.

### Playing

**Attempt**:
One playthrough of one Map or Scenario, from spawn until exit, death, or timeout. This is what Gymnasium and ViZDoom call an "episode".
_Avoid_: episode, run, game, rollout

**Clear**:
An Attempt that ends by reaching the Map's exit alive.
_Avoid_: win, solve, complete, beat

**Campaign**:
A chain of Attempts through the Maps of a Doom Episode in order, carrying health, ammo and weapons from each Clear into the next Map.
_Avoid_: playthrough, full run

**Beat the game**:
A Campaign that Clears every required Map of the target scope without restarting. The scope is stated each time: one Map, Doom Episode 1, or the Original Doom Episodes.
_Avoid_: finish, zerar

### What the agent knows

**Human-equivalent Observation**:
Information a human player has while playing: screen pixels, HUD values (health, ammo, keys), and the automap.
_Avoid_: fair input, pixels-only

**Privileged Information**:
Anything the engine exposes that a human player lacks: exact coordinates, depth buffer, object labels, sector data, save states.
_Avoid_: cheats, ground truth, internal state

### The agents

**Contender**:
One complete approach to playing, entered on the Scoreboard under a fixed name, such as "LLM only" or "PPO + RND".
_Avoid_: agent (when comparing), model, method, baseline

**Driver**:
The fast part of a Contender, choosing a button press every tic.
_Avoid_: low-level policy, controller, skill

**Navigator**:
The slow part of a Contender, choosing the next Subgoal from a wider view of the Map.
_Avoid_: planner, high-level policy, brain

**Subgoal**:
A target the Navigator hands to the Driver, such as "reach the blue key" or "open the door to the east".
_Avoid_: task, objective, waypoint

**Harness**:
Everything wrapped around an LLM to let it play: the decision loop, the memory, and the tools it may call.
_Avoid_: scaffold, wrapper, framework

**Harness rung**:
One step on a ladder of Harnesses that differ by a single feature, measured as its own Contender so the feature's effect can be read off the Scoreboard. M2's rungs: H0 (screen and actions), H1 (plus the automap), H2 (plus a notebook).
_Avoid_: level, tier, variant, config

### Measuring

**Eval Suite**:
The single fixed procedure that measures any Contender: same Maps, same seeds, same Difficulty, same metrics.
_Avoid_: benchmark, tests

**Eval Spec**:
One named, frozen set of Eval Suite parameters: Map, Difficulty, seeds, and tic limit, such as `e1m1-v1`; or, for a Scenario, the Scenario and seeds, such as `basic-v1`. Scoreboard rows are comparable only within one Eval Spec; changing any parameter means a new name.
_Avoid_: config, settings, benchmark version

**Attempt Session**:
The referee of one Attempt: it fixes the rules, builds what the Contender may see, measures Progress, and writes the Attempt's record. Every Contender plays through one, whether called in a loop or driving it through MCP tools.
_Avoid_: environment, episode runner, game wrapper

**Observation class**:
The label on a Scoreboard row saying what its Contender could see: `human-equivalent`, or `privileged` for teaching Contenders that see Privileged Information.
_Avoid_: input type, observation mode

**Progress**:
The share of the walking distance from spawn to the exit that an Attempt closed at its best moment, from 0 to 1, and 1 for a Clear. Measured from Privileged Information; never shown to a Contender.
_Avoid_: completion, distance travelled, coverage

**Scoreboard**:
The table of Eval Suite results, one row per Contender per Eval Spec. Map rows and Scenario rows are shown as two tables.
_Avoid_: leaderboard, results table

**Clear Rate**:
The fraction of Attempts that are Clears, over the Eval Suite's seeds.
_Avoid_: success rate, win rate

**Scenario reward**:
The score a Scenario defines for itself, summed over one Attempt: in `Basic`, +101 for the kill, -1 per tic and -5 per missed shot; in `DefendCenter`, +1 per kill and -1 for dying; in `DeadlyCorridor`, the distance moved along the corridor (computed from Privileged Information, so declared in Results) and -100 for dying. The Scenario table reports its mean over an Eval Spec's seeds.
_Avoid_: score (alone), return

**Reward shaping**:
Extra reward added during a Training Run only, on top of the Scenario reward or a Map's exit reward, to steer what the policy learns: for example, points per kill and a cost per health point lost on a Scenario, or points for Progress gained and a cost for dying on a Map. The Eval Suite never sees it; a checkpoint's `training.json` records it. Terms read from the engine are Privileged Information and are declared in Results.
_Avoid_: custom reward, bonus (alone)

**Reward hacking**:
A policy collecting reward in a way the reward's designer did not intend, such as charging forward to die in `DeadlyCorridor` because distance pays more than death costs. Caught by watching behaviour (videos, button and kill counts), not by the score, which goes up.
_Avoid_: cheating, exploit (alone)

**Training Run**:
One training process, tracked as one entry in Weights & Biases.
_Avoid_: run (alone), experiment, job

**Training seed**:
The seed of a Training Run: it fixes the network's starting weights and the games played while learning. Separate from an Eval Spec's seeds, which fix the games a trained policy is measured on. A PPO Contender is named after its training seed, such as `ppo-seed0` on a Scenario or `ppo-e1m1-d3-seed0` on a Map (Map, training Difficulty, seed, and a label when settings differ). On a Map, the games played while learning use game seeds from 1000 up, never an Eval Spec's.
_Avoid_: seed (alone, when both are in play)

**Action set**:
The fixed list of button combinations a learned Driver chooses from at each decision on a Map, such as "forward + turn left" or "use" (`maps.ACTIONS`, 11 entries). It limits what the policy does, not what it sees; an LLM Contender presses any of the 19 buttons.
_Avoid_: action space (alone), moves

**Checkpoint**:
The saved network a Training Run ends in (`model.zip`), with the run's cost beside it (`training.json`). A learned Contender plays from one.
_Avoid_: model (alone), weights file, snapshot

**Milestone**:
One step of the roadmap, ending in a measurable result and a study guide.
_Avoid_: phase, sprint, stage
