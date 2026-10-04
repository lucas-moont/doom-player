"""Training on an original Map ends in a checkpoint the Eval Suite can play.

A few hundred steps only: these check the pipeline, not that the policy
learns. Original Maps need the purchased WAD.
"""

import pytest
from conftest import predicts_an_action
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from doom_player.maps import make_map_env
from doom_player.paths import WAD_PATH
from doom_player.train import TrainConfig, train, training_record

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad"),
    pytest.mark.usefixtures("offline_wandb"),
]

TINY = dict(map="E1M1", seed=0, total_steps=128, n_envs=2, n_steps=64, batch_size=64)


def test_short_map_training_run_saves_a_loadable_checkpoint(tmp_path):
    checkpoint = train(TrainConfig(**TINY, out_dir=tmp_path))

    assert checkpoint.parent.name == "e1m1-d3-seed0"
    assert predicts_an_action(PPO.load(checkpoint), make_map_env("E1M1"))


def test_map_training_records_map_difficulty_and_shaping(tmp_path):
    config = TrainConfig(**TINY, difficulty=2, progress_reward=100.0, death_penalty=50.0, label="shaped", out_dir=tmp_path)
    record = training_record(train(config))

    assert (record["scenario"], record["map"], record["difficulty"]) == (None, "E1M1", 2)
    assert record["reward_shaping"] == {"progress_reward": 100.0, "death_penalty": 50.0}
    assert record["contender"] == "ppo-e1m1-d2-seed0-shaped"  # one Scoreboard name per setting
    assert record["recurrent"] is False and record["init_from"] is None


def test_recurrent_training_run_saves_a_loadable_checkpoint(tmp_path):
    checkpoint = train(TrainConfig(**TINY, recurrent=True, label="lstm", out_dir=tmp_path))

    assert training_record(checkpoint)["recurrent"] is True
    assert predicts_an_action(RecurrentPPO.load(checkpoint), make_map_env("E1M1"))


def test_map_training_logs_progress_and_clear_rate(tmp_path):
    # A Map's own reward is 0 until the exit, so its curve alone says nothing early on.
    # 3,200 steps over 2 copies: every copy finishes at least one 1,575-step Attempt.
    checkpoint = train(TrainConfig(**TINY | {"total_steps": 3200}, out_dir=tmp_path))

    (events,) = (checkpoint.parent / "tensorboard").rglob("events.*")
    accumulator = EventAccumulator(str(events))
    accumulator.Reload()
    assert {"rollout/progress", "rollout/clear_rate"} <= set(accumulator.Tags()["scalars"])
    for tag in ("rollout/progress", "rollout/clear_rate"):
        assert all(0 <= e.value <= 1 for e in accumulator.Scalars(tag))


def test_intermediate_checkpoints_are_named_apart_from_the_final_one(tmp_path):
    final = train(TrainConfig(**TINY, checkpoint_every=64, out_dir=tmp_path))

    halfway = training_record(final.parent / "steps-64" / "model.zip")
    assert halfway["contender"] == "ppo-e1m1-d3-seed0-at-64"  # its own Scoreboard name
    assert training_record(final)["contender"] == "ppo-e1m1-d3-seed0"


def test_training_continues_from_a_checkpoint(tmp_path):
    # A curriculum by Difficulty: learn on Difficulty 1, then continue on 3.
    easy = train(TrainConfig(**TINY, difficulty=1, out_dir=tmp_path))
    hard = train(TrainConfig(**TINY, difficulty=3, init_from=easy, label="from-d1", out_dir=tmp_path))

    before, after = training_record(easy), training_record(hard)
    assert after["init_from"] == before["run_id"]
    assert after["steps"] >= before["steps"] + 128  # the counts carry on
    assert predicts_an_action(PPO.load(hard), make_map_env("E1M1"))
