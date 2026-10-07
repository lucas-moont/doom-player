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

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd
from torch import nn

FRAME_SIZE = 84  # one gray frame of the policy view
FEATURES = 512  # how many numbers the target turns a screen into
CLIP = 5.0  # normalised pixels are clipped to this many spreads from the mean


def _layer(layer: nn.Module) -> nn.Module:
    nn.init.orthogonal_(layer.weight, np.sqrt(2))
    nn.init.zeros_(layer.bias)
    return layer


def _convolutions() -> list[nn.Module]:
    # Nature-DQN's three convolutions, as in CleanRL: 84x84 in, 64 maps of 7x7 out.
    return [
        _layer(nn.Conv2d(1, 32, 8, stride=4)),
        nn.LeakyReLU(),
        _layer(nn.Conv2d(32, 64, 4, stride=2)),
        nn.LeakyReLU(),
        _layer(nn.Conv2d(64, 64, 3, stride=1)),
        nn.LeakyReLU(),
        nn.Flatten(),
    ]


class RND:
    """The target, the predictor that learns to imitate it, and the pixel statistics both read through."""

    def __init__(
        self,
        seed: int = 0,
        device: str = "auto",
        learning_rate: float = 1e-4,
        epochs: int = 4,
        batch_size: int = 256,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):  # seeded weights, without moving the caller's random state
            torch.manual_seed(seed)
            flat = 64 * 7 * 7
            self.target = nn.Sequential(*_convolutions(), _layer(nn.Linear(flat, FEATURES)))
            self.predictor = nn.Sequential(
                *_convolutions(),
                _layer(nn.Linear(flat, FEATURES)),
                nn.ReLU(),
                _layer(nn.Linear(FEATURES, FEATURES)),
                nn.ReLU(),
                _layer(nn.Linear(FEATURES, FEATURES)),
            )
        self.target.to(self.device).requires_grad_(False)
        self.predictor.to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.epochs, self.batch_size = epochs, batch_size
        self.pixels = RunningMeanStd(shape=(FRAME_SIZE, FRAME_SIZE))
        self._rng = np.random.default_rng(seed)  # minibatch order

    def update_obs_stats(self, frames: np.ndarray) -> None:
        """Fold gray frames, uint8 [n, 84, 84], into the running mean and spread of each pixel."""
        self.pixels.update(frames.astype(np.float64))

    def _normalise(self, frames: np.ndarray) -> torch.Tensor:
        mean = torch.as_tensor(self.pixels.mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(np.sqrt(self.pixels.var), dtype=torch.float32, device=self.device)
        x = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        return ((x - mean) / std).clamp(-CLIP, CLIP).unsqueeze(1)

    @torch.no_grad()
    def bonus(self, frames: np.ndarray) -> np.ndarray:
        """How badly the predictor guesses each frame's target numbers: the raw bonus, one per frame."""
        x = self._normalise(frames)
        return ((self.target(x) - self.predictor(x)).pow(2).sum(1) / 2).cpu().numpy()

    def fit(self, frames: np.ndarray) -> float:
        """Train the predictor on `frames` for a few passes; return the mean loss of the last pass."""
        x = self._normalise(frames)
        with torch.no_grad():
            wanted = self.target(x)
        for _ in range(self.epochs):
            order = self._rng.permutation(len(x))
            losses = []
            for start in range(0, len(x), self.batch_size):
                batch = order[start : start + self.batch_size]
                loss = (self.predictor(x[batch]) - wanted[batch]).pow(2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    def state_dict(self) -> dict:
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "pixels": (torch.from_numpy(self.pixels.mean), torch.from_numpy(self.pixels.var), self.pixels.count),
            "rng": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self.optimizer.load_state_dict(state["optimizer"])
        mean, var, self.pixels.count = state["pixels"]
        self.pixels.mean, self.pixels.var = mean.numpy().copy(), var.numpy().copy()
        self._rng.bit_generator.state = state["rng"]
