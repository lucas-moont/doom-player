"""The random Contender: every button pressed or not by a coin flip.

It is the floor every later Contender must beat.
"""

import numpy as np

from doom_player.session import Observation


class RandomContender:
    name = "random"
    tics_per_action = 4

    def reset(self, seed: int, buttons: list[str]) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = len(buttons)

    def act(self, observation: Observation) -> list[bool]:
        return self._rng.integers(0, 2, self._n).astype(bool).tolist()
