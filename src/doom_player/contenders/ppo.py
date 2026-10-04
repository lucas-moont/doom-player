"""PPO Contenders: CNN policies trained by `doom-train`, playing from their checkpoint.

They see the screen only, in the policy view (gray, 84x84, last 4 frames), so
their Observation class is human-equivalent. They always pick their most likely
action, so the same seeds replay the same Attempts.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from doom_player.maps import make_map_env
from doom_player.train import training_record

if TYPE_CHECKING:
    from doom_player.eval import EvalSpec


def load_policy(checkpoint: Path):
    """The checkpoint's policy and its Training Run record; an LSTM policy loads as RecurrentPPO."""
    training = training_record(checkpoint)
    if training.get("recurrent"):
        return RecurrentPPO.load(checkpoint, device="cpu"), training
    return PPO.load(checkpoint, device="cpu"), training


def contender_name(training: dict) -> str:
    # Checkpoints from before M4 have no name in their record; theirs was `ppo-seed<N>`.
    return training.get("contender") or f"ppo-seed{training['seed']}"


class _FromCheckpoint:
    """A Contender that plays from a checkpoint, named after its Training Run."""

    def __init__(self, checkpoint: Path):
        self._model, self.training = load_policy(checkpoint)  # training: the cost columns
        self.name = contender_name(self.training)  # one name per setting and training seed


class PPOContender(_FromCheckpoint):
    """A policy trained on a Scenario, deciding one step at a time."""

    def reset(self, seed: int, action_space: gym.spaces.Discrete) -> None:
        pass  # the policy keeps no memory between decisions; recent frames are in the observation

    def act(self, observation: np.ndarray) -> int:
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)


class PPOMapContender(_FromCheckpoint):
    """A policy trained on an original Map, playing each Attempt whole through the env it trained in.

    The Map env wraps the same AttemptSession the Eval Suite uses and adds the
    policy view, so the policy is measured on exactly what it trained on (ADR 0011).
    """

    def play_attempt(self, spec: "EvalSpec", seed: int, on_frame: Callable | None = None) -> dict:
        trained_on = self.training.get("map") or self.training.get("scenario")
        if trained_on != spec.map:
            raise SystemExit(f"{self.name} was trained on {trained_on}, not {spec.map}")
        env = make_map_env(spec.map, spec.difficulty, spec.tic_limit, contender=self.name)
        try:
            if self._model.action_space != env.action_space:
                raise SystemExit(f"{self.name} was trained with another Action set than maps.ACTIONS")
            obs, _ = env.reset(options={"game_seed": seed})
            env.unwrapped.on_frame = on_frame
            state, start, done = None, True, False
            while not done:
                # A recurrent policy carries its memory in `state`; a plain one ignores it.
                action, state = self._model.predict(
                    obs, state=state, episode_start=np.array([start]), deterministic=True
                )
                obs, _, terminated, truncated, info = env.step(int(action))
                start, done = False, terminated or truncated
            return info["record"]
        finally:
            env.close()
