"""Random Network Distillation (RND): a training bonus for seeing something new.

Like a flashcard quiz: a fixed, randomly built network (the target) turns
each screen into 512 numbers, and a second network (the predictor) keeps
learning to guess them. On screens seen many times its guess is good; on a
new screen it is bad, and the size of the miss is paid as a bonus, so new
places pay until they become familiar (Burda et al. 2018,
https://arxiv.org/abs/1810.12894). It reads only the screen the policy sees.

Written after CleanRL's `ppo_rnd_envpool.py`; how it joins SB3's PPO, and
what is simplified, is recorded in the M5 brief.
"""

from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, unwrap_vec_wrapper
from torch import nn

from doom_player.scenarios import FRAME_SIZE  # one gray frame of the policy view

FEATURES = 512  # how many numbers the target turns a screen into
CLIP = 5.0  # normalised pixels are clipped to this many spreads from the mean
# CleanRL's settings for the predictor and the bonus's discount.
LEARNING_RATE = 1e-4
EPOCHS = 4  # passes over a rollout's frames per predictor update
BATCH_SIZE = 256
INTRINSIC_GAMMA = 0.99


def _convolutions() -> list[nn.Module]:
    # Nature-DQN's three convolutions, as in CleanRL: 84x84 in, 64 maps of 7x7 out.
    # SB3's NatureCNN uses ReLU and ends in a ReLU, which the target must not.
    return [
        nn.Conv2d(1, 32, 8, stride=4),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.LeakyReLU(),
        nn.Flatten(),
    ]


def _rms_state(rms: RunningMeanStd) -> tuple:
    # Tensors and plain numbers only, so the file loads with torch's safe loader.
    return torch.as_tensor(rms.mean), torch.as_tensor(rms.var), float(rms.count)


def _load_rms(rms: RunningMeanStd, state: tuple) -> None:
    mean, var, rms.count = state
    rms.mean, rms.var = mean.numpy().copy(), var.numpy().copy()


class RND:
    """The target, the predictor that learns to imitate it, and the pixel statistics both read through."""

    def __init__(self, seed: int = 0, device: str = "auto"):
        self.device = get_device(device)
        with torch.random.fork_rng(devices=[]):  # seeded weights, without moving the caller's random state
            torch.manual_seed(seed)
            flat = 64 * 7 * 7
            self.target = nn.Sequential(*_convolutions(), nn.Linear(flat, FEATURES))
            self.predictor = nn.Sequential(
                *_convolutions(),
                nn.Linear(flat, FEATURES),
                nn.ReLU(),
                nn.Linear(FEATURES, FEATURES),
                nn.ReLU(),
                nn.Linear(FEATURES, FEATURES),
            )
            for net in (self.target, self.predictor):
                net.apply(partial(BasePolicy.init_weights, gain=np.sqrt(2)))
        self.target.to(self.device).requires_grad_(False)
        self.predictor.to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=LEARNING_RATE)
        self.pixels = RunningMeanStd(shape=(FRAME_SIZE, FRAME_SIZE))
        self._pixels_changed()
        self._rng = np.random.default_rng(seed)  # minibatch order

    def update_obs_stats(self, frames: np.ndarray) -> None:
        """Fold gray frames, uint8 [n, 84, 84], into the running mean and spread of each pixel."""
        self.pixels.update(frames.astype(np.float64))
        self._pixels_changed()

    def _pixels_changed(self) -> None:
        # On the device once per change, not once per frame batch read.
        self._mean = torch.as_tensor(self.pixels.mean, dtype=torch.float32, device=self.device)
        self._std = torch.as_tensor(np.sqrt(self.pixels.var), dtype=torch.float32, device=self.device)

    def _normalise(self, frames: torch.Tensor) -> torch.Tensor:
        return ((frames.float() - self._mean) / self._std).clamp(-CLIP, CLIP).unsqueeze(1)

    def _on_device(self, frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frames).to(self.device)  # still uint8: a quarter of the bytes to copy

    @torch.no_grad()
    def bonus(self, frames: np.ndarray) -> np.ndarray:
        """How badly the predictor guesses each frame's target numbers: the raw bonus, one per frame."""
        x = self._normalise(self._on_device(frames))
        return ((self.target(x) - self.predictor(x)).pow(2).sum(1) / 2).cpu().numpy()

    def fit(self, frames: np.ndarray) -> float:
        """Train the predictor on `frames` for a few passes; return the mean loss of the last pass."""
        frames = self._on_device(frames)
        for _ in range(EPOCHS):
            order = torch.from_numpy(self._rng.permutation(len(frames))).to(self.device)
            losses = []
            for batch in order.split(BATCH_SIZE):
                x = self._normalise(frames[batch])  # one minibatch at a time keeps GPU memory small
                with torch.no_grad():
                    wanted = self.target(x)
                loss = (self.predictor(x) - wanted).pow(2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.detach())
        return torch.stack(losses).mean().item()

    def state_dict(self) -> dict:
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "pixels": _rms_state(self.pixels),
            "rng": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self.optimizer.load_state_dict(state["optimizer"])
        _load_rms(self.pixels, state["pixels"])
        self._pixels_changed()
        self._rng.bit_generator.state = state["rng"]


class RNDBonus(VecEnvWrapper):
    """Training only: add `coef` times the normalised RND bonus to every step's reward.

    Wraps the batch of game copies in the main process, so one predictor
    learns from every copy. The bonus is added as each step happens, because
    SB3 computes a rollout's advantages before any callback sees it; the
    predictor trains once per rollout (`train_on_rollout`, called by
    `RNDUpdate`). Each step pays for the screen it reached: when an Attempt
    ends, that is the info's terminal observation, not the next spawn.

    The raw bonus is divided by the running spread of its own discounted sum,
    so `coef` means the same whatever scale the errors have. For the first
    `warmup_frames` frames only the pixel statistics learn: nothing is paid,
    and the spread starts from the settled bonus, not from the warm-up's
    noise, which a running spread would never forget (as in CleanRL).
    """

    def __init__(self, venv: VecEnv, coef: float, rnd: RND, warmup_frames: int = 0):
        super().__init__(venv)
        self.rnd, self.coef, self.warmup_frames = rnd, coef, warmup_frames
        self.frames_seen = 0
        self.returns = RunningMeanStd(shape=())  # spread of the discounted raw bonus
        self._discounted = np.zeros(self.num_envs)
        self._attempt_bonus = np.zeros(self.num_envs)  # paid so far in each copy's Attempt
        self._attempt_reward = np.zeros(self.num_envs)  # the reward it is added to: shaped, then rescaled
        self._rollout_frames: list[np.ndarray] = []  # what the predictor learns from next
        self._rollout_raw: list[np.ndarray] = []

    @property
    def settings(self) -> dict:
        """How the bonus was paid, as `training.json` records it."""
        return {
            "coef": self.coef,
            "frame": "latest",  # the newest frame of the policy view's stack
            "value_heads": 1,  # added to the reward; the paper keeps a second value head
            "warmup_frames": self.warmup_frames,
            "intrinsic_gamma": INTRINSIC_GAMMA,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        ended = np.flatnonzero(dones)
        frames = obs[:, -1].copy()  # the newest frame of each copy's stack
        for i in ended:
            frames[i] = infos[i]["terminal_observation"][-1]
        self.rnd.update_obs_stats(frames)
        self.frames_seen += len(frames)
        raw = self.rnd.bonus(frames)
        if self.frames_seen > self.warmup_frames:
            self._discounted = self._discounted * INTRINSIC_GAMMA + raw
            self.returns.update(self._discounted)
            paid = self.coef * raw / np.sqrt(self.returns.var + 1e-8)
        else:
            paid = np.zeros_like(raw)
        self._rollout_frames.append(frames)
        self._rollout_raw.append(raw)

        self._attempt_bonus += paid
        self._attempt_reward += rewards
        for i in ended:
            infos[i]["rnd"] = {"bonus": float(self._attempt_bonus[i]), "scaled_reward": float(self._attempt_reward[i])}
            self._attempt_bonus[i] = self._attempt_reward[i] = 0.0
        return obs, (rewards + paid).astype(np.float32), dones, infos

    def train_on_rollout(self) -> dict[str, float]:
        """Train the predictor on the frames since the last call; return numbers to log."""
        frames, raw = np.concatenate(self._rollout_frames), np.concatenate(self._rollout_raw)
        self._rollout_frames, self._rollout_raw = [], []
        return {
            "rnd/predictor_loss": self.rnd.fit(frames),
            "rnd/bonus_raw": float(raw.mean()),  # before scaling: falls as screens become familiar
            "rnd/bonus_scale": float(np.sqrt(self.returns.var)),
        }

    def save(self, path: Path) -> None:
        """Everything a continued Training Run needs to pay the same bonus."""
        state = {
            "rnd": self.rnd.state_dict(),
            "returns": _rms_state(self.returns),
            "discounted": torch.from_numpy(self._discounted),
            "frames_seen": self.frames_seen,
        }
        torch.save(state, path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu")  # the networks move themselves; the statistics stay numpy
        self.rnd.load_state_dict(state["rnd"])
        _load_rms(self.returns, state["returns"])
        if len(state["discounted"]) == self.num_envs:  # else a different number of copies: start the sums afresh
            self._discounted = state["discounted"].numpy().copy()
        self.frames_seen = state["frames_seen"]


class RNDUpdate(BaseCallback):
    """Train RND's predictor after every rollout, and log what it paid per Attempt.

    `rnd/attempt_scaled_reward` is the reward the bonus is added to, on the same
    scale: the Map's reward plus shaping, as VecNormalize rescaled it. It is not
    the game's reward that `rollout/ep_rew_mean` shows.
    """

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if paid := info.get("rnd"):  # the Attempt that copy was playing just ended
                self.logger.record_mean("rnd/attempt_bonus", paid["bonus"])
                self.logger.record_mean("rnd/attempt_scaled_reward", paid["scaled_reward"])
        return True

    def _on_rollout_end(self) -> None:
        for key, value in unwrap_vec_wrapper(self.model.get_env(), RNDBonus).train_on_rollout().items():
            self.logger.record(key, value)
