"""A Training Run's settings, checked before any game starts: fast, no training."""

import pytest

from doom_player.train import TrainConfig


def test_a_map_trains_under_its_eval_specs_rules():
    assert TrainConfig(map="E1M1").map_rules == (6300, "doors-open")  # e1m1-v1
    assert TrainConfig(map="E1M2").map_rules == (12600, "keyed")  # e1m2-v1


def test_tic_limit_can_be_set_for_a_map_only():
    assert TrainConfig(map="E1M2", tic_limit=700, label="short").map_rules == (700, "keyed")
    with pytest.raises(ValueError, match="label"):
        TrainConfig(map="E1M2", tic_limit=700)
    assert TrainConfig(map="E1M3").map_rules == (6300, "doors-open")  # no Eval Spec yet
    with pytest.raises(ValueError, match="Maps only"):
        TrainConfig(scenario="basic", tic_limit=700, label="short")


def test_shaping_pays_doors_open_progress_whatever_the_score_uses():
    shaping = TrainConfig(map="E1M2", progress_reward=100.0, death_penalty=25.0).reward_shaping
    assert shaping == {"progress_reward": 100.0, "death_penalty": 25.0, "progress_rule": "doors-open"}
    assert TrainConfig(map="E1M2").reward_shaping is None


def test_shaping_can_pay_keyed_progress_where_the_attempts_measure_it():
    shaped = dict(map="E1M2", progress_reward=100.0, death_penalty=25.0)
    shaping = TrainConfig(**shaped, shaping_rule="keyed", label="keyed-shaped").reward_shaping
    assert shaping == {"progress_reward": 100.0, "death_penalty": 25.0, "progress_rule": "keyed"}
    with pytest.raises(ValueError, match="label"):
        TrainConfig(**shaped, shaping_rule="keyed", label="shaped")
    with pytest.raises(ValueError, match="label"):  # the name would claim a reward the run never paid
        TrainConfig(**shaped, label="keyed-shaped")
    with pytest.raises(ValueError, match="E1M1"):  # e1m1-v1's Attempts measure doors-open Progress only
        TrainConfig(map="E1M1", progress_reward=100.0, shaping_rule="keyed", label="keyed")
    with pytest.raises(ValueError, match="progress-reward"):
        TrainConfig(map="E1M2", shaping_rule="keyed", label="keyed")
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig(**shaped, shaping_rule="nearest", label="nearest")


def test_an_rnd_training_run_is_named_for_it():
    with pytest.raises(ValueError, match="label"):
        TrainConfig(map="E1M2", rnd_coef=0.5, label="shaped")
    with pytest.raises(ValueError, match="label"):  # the name would claim a bonus the run never paid
        TrainConfig(map="E1M2", label="shaped-rnd")
    with pytest.raises(ValueError, match="label"):
        TrainConfig(map="E1M2", rnd_coef=0.5, label="grnd")
    with pytest.raises(ValueError, match="positive"):
        TrainConfig(map="E1M2", rnd_coef=-1.0, label="rnd")
    assert TrainConfig(map="E1M2", rnd_coef=0.5, label="shaped-rnd").contender == "ppo-e1m2-d3-seed0-shaped-rnd"
