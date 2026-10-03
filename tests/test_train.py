"""A Training Run ends in a checkpoint that a Contender can load and play from.

These runs train for a few hundred steps only: they check the pipeline, not
that the policy learns. Scenarios ship inside the vizdoom package, so no WAD.
"""

import pytest
from stable_baselines3 import PPO

from doom_player.scenarios import make_scenario_env
from doom_player.train import TrainConfig, train, training_record

pytestmark = pytest.mark.slow

TINY = dict(scenario="basic", seed=0, total_steps=128, n_envs=2, n_steps=64, batch_size=64)


@pytest.fixture(autouse=True)
def offline_wandb(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))


def test_short_training_run_saves_a_loadable_checkpoint(tmp_path):
    checkpoint = train(TrainConfig(**TINY, out_dir=tmp_path))

    model = PPO.load(checkpoint)
    env = make_scenario_env("basic")
    try:
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(int(action))
    finally:
        env.close()


def test_training_cost_is_recorded_with_the_checkpoint(tmp_path):
    checkpoint = train(TrainConfig(**TINY, out_dir=tmp_path))

    record = training_record(checkpoint)
    assert record["scenario"] == "basic"
    assert record["seed"] == 0
    assert record["steps"] >= 128
    assert record["wall_clock_s"] > 0
