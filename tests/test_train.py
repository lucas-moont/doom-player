"""A Training Run ends in a checkpoint that a Contender can load and play from.

These runs train for a few hundred steps only: they check the pipeline, not
that the policy learns. Scenarios ship inside the vizdoom package, so no WAD.
"""

import pytest
from conftest import predicts_an_action
from stable_baselines3 import PPO

from doom_player.scenarios import make_scenario_env
from doom_player.train import TrainConfig, train, training_record

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("offline_wandb")]

TINY = dict(scenario="basic", seed=0, total_steps=128, n_envs=2, n_steps=64, batch_size=64)


def test_short_training_run_saves_a_loadable_checkpoint(tmp_path):
    checkpoint = train(TrainConfig(**TINY, out_dir=tmp_path))

    assert predicts_an_action(PPO.load(checkpoint), make_scenario_env("basic"))


def test_training_cost_is_recorded_with_the_checkpoint(tmp_path):
    checkpoint = train(TrainConfig(**TINY, out_dir=tmp_path))

    record = training_record(checkpoint)
    assert record["scenario"] == "basic"
    assert record["seed"] == 0
    assert record["steps"] >= 128
    assert record["wall_clock_s"] > 0
    assert record["reward_shaping"] is None  # trained on the Scenario's own reward


def test_reward_shaping_is_recorded_with_the_checkpoint(tmp_path):
    config = TrainConfig(**TINY, kill_reward=100.0, health_penalty=1.0, out_dir=tmp_path)
    checkpoint = train(config)

    assert training_record(checkpoint)["reward_shaping"] == {"kill_reward": 100.0, "health_penalty": 1.0}
