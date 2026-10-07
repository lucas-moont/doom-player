"""Training on an original Map ends in a checkpoint the Eval Suite can play.

A few hundred steps only: these check the pipeline, not that the policy
learns. Original Maps need the purchased WAD.
"""

import pytest
import torch
from conftest import logged_scalars, predicts_an_action
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from doom_player.contenders.ppo import PPOMapContender
from doom_player.eval import EvalSpec
from doom_player.maps import make_map_env
from doom_player.paths import WAD_PATH
from doom_player.train import RND_FILE, TrainConfig, train, training_record

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
    assert record["reward_shaping"] == {"progress_reward": 100.0, "death_penalty": 50.0, "progress_rule": "doors-open"}
    assert (record["tic_limit"], record["progress_rule"]) == (6300, "doors-open")  # e1m1-v1's rules
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

    accumulator = logged_scalars(checkpoint)
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
    # The reward scale the value network learned on is saved, so the next stage can start from it.
    assert (easy.parent / "vecnormalize.pkl").exists() and (hard.parent / "vecnormalize.pkl").exists()
    assert predicts_an_action(PPO.load(hard), make_map_env("E1M1"))


def test_rnd_training_run_logs_the_bonus_and_saves_it_with_each_checkpoint(tmp_path):
    config = TrainConfig(**TINY, rnd_coef=0.5, label="rnd", checkpoint_every=64, out_dir=tmp_path)
    checkpoint = train(config)

    assert training_record(checkpoint)["rnd"]["coef"] == 0.5
    assert (checkpoint.parent / RND_FILE).exists()
    assert all((halfway / RND_FILE).exists() for halfway in checkpoint.parent.glob("steps-*"))
    accumulator = logged_scalars(checkpoint)
    assert {"rnd/predictor_loss", "rnd/bonus_raw"} <= set(accumulator.Tags()["scalars"])
    # The bonus is training only: the policy plays the Eval Suite like any other.
    record = PPOMapContender(checkpoint).play_attempt(EvalSpec("test-short", "E1M1", 3, (0,), 140), 0)
    assert record["actions"] > 0


def test_a_continued_rnd_training_run_keeps_its_parents_bonus(tmp_path):
    parent = train(TrainConfig(**TINY, rnd_coef=0.5, label="rnd", out_dir=tmp_path))
    child = train(TrainConfig(**TINY, rnd_coef=0.5, init_from=parent, label="rnd-more", out_dir=tmp_path))

    seen = [torch.load(c.parent / RND_FILE)["frames_seen"] for c in (parent, child)]
    assert seen[1] == seen[0] + 128  # counted on from the parent's frames, not from 0


def test_a_map_with_keys_also_logs_how_often_one_was_picked_up(tmp_path):
    # Short Attempts (--tic-limit) so that 2 copies finish some within 256 steps.
    checkpoint = train(TrainConfig(**TINY | {"map": "E1M2", "total_steps": 256}, tic_limit=280, label="short", out_dir=tmp_path))

    accumulator = logged_scalars(checkpoint)
    assert all(e.value == 0.0 for e in accumulator.Scalars("rollout/key_rate"))  # the key is far from the spawn
