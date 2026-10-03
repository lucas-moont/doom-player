"""The PPO Contender: a CNN policy trained by `doom-train`, playing from its checkpoint.

It sees what the Scenario environment gives it, the stacked grayscale screen,
so its Observation class is human-equivalent. It always picks its most likely
action, so the same seeds replay the same Attempts.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from doom_player.train import training_record


class PPOContender:
    def __init__(self, checkpoint: Path):
        self._model = PPO.load(checkpoint, device="cpu")
        self.training = training_record(checkpoint)  # the cost column of the Scoreboard
        # One name per training seed: each policy keeps its own Attempts file and Scoreboard row.
        self.name = f"ppo-seed{self.training['seed']}"

    def reset(self, seed: int, action_space: gym.spaces.Discrete) -> None:
        pass  # the policy keeps no memory between decisions; recent frames are in the observation

    def act(self, observation: np.ndarray) -> int:
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)
