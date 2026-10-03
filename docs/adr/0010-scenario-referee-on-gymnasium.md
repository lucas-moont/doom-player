# Scenarios are refereed by their own Gymnasium environment

The `AttemptSession` of ADR 0007 knows original Maps only: it loads the purchased WAD and measures Progress toward a Map's exit, which a Scenario does not have. A PPO policy trains on ViZDoom's registered Gymnasium environment for the Scenario, seen through `make_scenario_env` (`src/doom_player/scenarios.py`: screen only, grayscale, 84x84, the last 4 frames stacked). So Scenarios get their own referee, `run_scenario_eval` (`src/doom_player/scenario_eval.py`), which plays each Attempt on that same environment and scores the Scenario's built-in reward. A policy is then measured on exactly the observations, frame skip and reward it was trained on, and a mismatch between a training wrapper and an evaluation wrapper cannot creep in. Scenario Eval Specs, the Attempts file layout, resume and the Scoreboard file are shared with Maps. Settled with the owner on 2026-10-03.

## Considered Options

- **Teach `AttemptSession` to load Scenarios**: one referee for everything, but the session drives `DoomGame` directly, so a policy would be trained in one environment and measured in another, and every change to the observation would have to be made twice.

## Consequences

- ADR 0007's rule that training observations stay equal to `AttemptSession.observe` holds for Maps. On Scenarios the referee and the training environment are the same object.
- Scenario Contenders implement a small protocol of their own: `reset(seed, action_space)` and `act(observation) -> action index`.
- When a learned Driver moves to original Maps (M4), it needs a Gymnasium environment for Maps whose observations match `AttemptSession.observe`, or the session gains a Gymnasium face. That choice is the M4 brief's.
