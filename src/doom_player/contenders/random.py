"""The random Contender: every button pressed or not by a coin flip.

It is the floor every later Contender must beat.
"""

import gymnasium as gym
import numpy as np

from doom_player.session import Observation


class RandomContender:
    name = "random"
    tics_per_action = 4
    training = None

    def reset(self, seed: int, buttons: list[str]) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = len(buttons)

    def act(self, observation: Observation) -> list[bool]:
        return self._rng.integers(0, 2, self._n).astype(bool).tolist()


class RandomScenarioContender:
    """The same floor on a Scenario: one of the Scenario's actions, uniformly at random."""

    name = "random"
    training = None

    def reset(self, seed: int, action_space: gym.spaces.Discrete) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = int(action_space.n)

    def act(self, observation: np.ndarray) -> int:
        return int(self._rng.integers(self._n))
