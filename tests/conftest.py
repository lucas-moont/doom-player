"""Helpers shared by the tests that train tiny policies.

A tiny Training Run lasts a few hundred steps: it checks the pipeline, not
that the policy learns. W&B runs offline in tests, writing under the test's
temporary folder.
"""

from pathlib import Path

import pytest

from doom_player.train import TrainConfig, train


def _offline(monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(folder))


@pytest.fixture
def offline_wandb(monkeypatch, tmp_path):
    _offline(monkeypatch, tmp_path)


def train_tiny(out: Path, **config) -> Path:
    """Train a tiny policy into `out`, with W&B offline; usable from module-scoped fixtures."""
    settings = dict(seed=0, total_steps=64, n_envs=1, n_steps=64, batch_size=64)
    with pytest.MonkeyPatch.context() as mp:
        _offline(mp, out)
        return train(TrainConfig(**{**settings, **config, "out_dir": out}))


def predicts_an_action(model, env) -> bool:
    """The policy picks an action the env accepts, from the env's first observation."""
    try:
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, deterministic=True)
        return env.action_space.contains(int(action))
    finally:
        env.close()
