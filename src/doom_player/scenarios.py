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
from vizdoom import GameVariable
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
    return policy_view(env)


def policy_view(env: gym.Env) -> gym.Env:
    """What a CNN policy sees of an RGB screen: gray, 84x84, the last 4 frames.

    Shared by Scenarios and Maps, so a policy always gets the same picture.
    """
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    return FrameStackObservation(env, FRAMES_STACKED)


class RewardShaping(gym.Wrapper):
    """Training only: add `kill_reward` per kill and charge `health_penalty` per health point lost.

    Like a coach's extra points on top of the match score: the Scenario's own
    reward stays in `info["raw_reward"]`, and the Eval Suite never sees the
    extra points. Kills and health are read from the game engine, not shown to
    the policy, so both terms are privileged and declared in Results (ADR 0002).
    """

    def __init__(self, env: gym.Env, kill_reward: float, health_penalty: float):
        super().__init__(env)
        self.kill_reward = kill_reward
        self.health_penalty = health_penalty

    def _read(self) -> tuple[float, float]:
        game = self.unwrapped.game
        # Health can drop below zero on the killing blow; count it from zero.
        return game.get_game_variable(GameVariable.KILLCOUNT), max(0.0, game.get_game_variable(GameVariable.HEALTH))

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._kills, self._health = self._read()
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        kills, health = self._read()
        bonus = self.kill_reward * (kills - self._kills) - self.health_penalty * max(0.0, self._health - health)
        self._kills, self._health = kills, health
        return obs, reward + bonus, terminated, truncated, {**info, "raw_reward": reward}
