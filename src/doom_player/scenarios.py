"""Scenario environments shaped for a learned Driver.

A Scenario is a small ViZDoom training level, such as `Basic`. ViZDoom
registers each one as a Gymnasium environment whose observation is a
dictionary: the screen plus a few game variables. A CNN policy wants one
image instead, so `make_scenario_env` keeps the screen only, turns it gray,
shrinks it to 84x84 pixels, and stacks the last 4 frames so that one
observation shows motion.
"""

import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
    TransformObservation,
)
from vizdoom import gymnasium_wrapper  # noqa: F401  (registers the Vizdoom* environments)

SCENARIOS = {
    "basic": "VizdoomBasic-v1",
    "defend-center": "VizdoomDefendCenter-v1",
    # Its built-in reward follows the player's distance along the corridor,
    # a privileged term: declared in the M3 Results (ADR 0002).
    "deadly-corridor": "VizdoomDeadlyCorridor-v1",
}

FRAME_SIZE = 84
FRAMES_STACKED = 4
FRAME_SKIP = 4  # tics each action is held, as the random Contender does on Maps


def make_scenario_env(
    scenario: str, frame_skip: int = FRAME_SKIP, render_mode: str | None = None
) -> gym.Env:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown Scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")
    env = gym.make(SCENARIOS[scenario], frame_skip=frame_skip, render_mode=render_mode)
    env = TransformObservation(
        env, lambda obs: obs["screen"], env.observation_space["screen"]
    )
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    return FrameStackObservation(env, FRAMES_STACKED)
