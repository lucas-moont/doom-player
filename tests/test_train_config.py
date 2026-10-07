"""A Training Run's settings, checked before any game starts: fast, no training."""

import pytest

from doom_player.train import TrainConfig


def test_a_map_trains_under_its_eval_specs_rules():
    assert TrainConfig(map="E1M1").map_rules == (6300, "doors-open")  # e1m1-v1
    assert TrainConfig(map="E1M2").map_rules == (12600, "keyed")  # e1m2-v1


def test_tic_limit_can_be_set_for_a_map_only():
    assert TrainConfig(map="E1M2", tic_limit=700).map_rules == (700, "keyed")
    assert TrainConfig(map="E1M3").map_rules == (6300, "doors-open")  # no Eval Spec yet
    with pytest.raises(ValueError, match="Maps only"):
        TrainConfig(scenario="basic", tic_limit=700)


def test_shaping_pays_doors_open_progress_whatever_the_score_uses():
    shaping = TrainConfig(map="E1M2", progress_reward=100.0, death_penalty=25.0).reward_shaping
    assert shaping == {"progress_reward": 100.0, "death_penalty": 25.0, "progress_rule": "doors-open"}
    assert TrainConfig(map="E1M2").reward_shaping is None
