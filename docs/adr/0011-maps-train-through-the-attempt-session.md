# Maps train through a Gymnasium face of the AttemptSession

A learned Driver on an original Map trains on `MapEnv` (`src/doom_player/maps.py`). It is like a universal adapter fitted to the referee: Stable-Baselines3 only plugs into Gymnasium environments, so `MapEnv` gives the `AttemptSession`, the same referee the Eval Suite uses, a Gymnasium shape, with one Attempt per Gymnasium episode. `make_map_env` then adds the policy view Scenarios already use (`scenarios.policy_view`: the screen in gray, shrunk to 84x84, the last 4 frames stacked, each frame held for 4 tics). The Eval Suite measures a learned Map Contender by letting it play whole Attempts through that same env (`PPOMapContender.play_attempt`), so the screen, frame skip, time limit, game seeds and Progress a policy is measured on are the ones it trained on, by construction. Training games draw their seeds from 1000 up, so a policy never practises on an Eval Spec's games. Settled with the owner on 2026-10-03, after ADR 0010 left the choice to M4.

## Considered Options

- **ViZDoom's own Map environments** (`VizdoomDoomE1M1-S3-v0`): they load the project's WAD through the `doom_game_path` keyword and share `doom.cfg`, but differ from the session in time limit (126,000 tics), seeding (the env's seed *N* is not the session's seed *N*), audio (on), the last frame (all zeros) and actions (19 free buttons). Evaluating through the session would then mean rebuilding the policy view outside the env, the mismatch ADR 0010 avoided for Scenarios.

## Consequences

- This amends ADR 0007 for learned Map Contenders. They no longer enter through Door A, one decision at a time; they implement `WholeAttemptContender` (`contenders/base.py`) and play each Attempt through `MapEnv`. The rules still live in one place: `run_eval` names the game seed (`options={"game_seed": s}`), the session enforces the tic limit and writes the record, and the policy receives only the env's observations. The session's Progress and position stay inside the env; only training reward reads them (`ProgressShaping`, declared in Results).
- ADR 0007's rule that training observations equal `AttemptSession.observe` holds for the part a policy sees: `MapEnv` takes `AttemptSession.screen()`, the same screen `observe` returns, without copying the automap a policy does not use.
- One `AttemptSession` per episode builds a new `DoomGame` each time: 142 ms, against 1.13 s for a full 6,300-tic Attempt in one process (measured 2026-10-03). Reusing the game across episodes was left out, since it would change the session's one-Attempt-per-object contract.
- The policy chooses from an Action set of 11 button combinations (`maps.ACTIONS`), like a game controller with fewer buttons: it limits what the policy does, not what it sees. Weapon switching is left out.
- Map Attempt records carry the checkpoint they were played by (stamped by `run_eval`), and the Map Scoreboard gains a training-cost column, as the Scenario table has.
