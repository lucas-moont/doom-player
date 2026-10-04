"""A policy trained on a Map enters the Map Scoreboard, next to random and the LLM rungs."""

import pytest
from conftest import train_tiny

from doom_player.contenders.ppo import PPOMapContender
from doom_player.eval import EvalSpec, attempts_path, read_records, run_eval
from doom_player.paths import WAD_PATH
from doom_player.train import training_record

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad"),
]

SHORT = EvalSpec("test-short", "E1M1", 3, (0, 1), 280)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return train_tiny(tmp_path_factory.mktemp("train"), map="E1M1")


def test_ppo_contender_plays_a_map_from_a_checkpoint(checkpoint, tmp_path):
    row = run_eval(PPOMapContender(checkpoint), SHORT, tmp_path / "a", video_dir=tmp_path / "videos")

    assert row["contender"] == "ppo-e1m1-d3-seed0"
    assert row["seeds"] == [0, 1]
    assert 0 <= row["progress_mean"] <= 1 and 0 <= row["clear_rate"] <= 1
    assert row["training_steps"] >= 64 and row["training_wall_clock_s"] > 0
    records = read_records(attempts_path(tmp_path / "a", row["contender"], SHORT))
    assert {r["checkpoint"] for r in records} == {training_record(checkpoint)["run_id"]}
    assert row["tokens_per_attempt"] == 0
    assert len(list((tmp_path / "videos").glob("ppo-e1m1-d3-seed0-test-short-seed*.mp4"))) == 2

    # The policy picks its most likely action, so the same seeds repeat exactly.
    again = run_eval(PPOMapContender(checkpoint), SHORT, tmp_path / "b")
    assert again["progress_by_seed"] == row["progress_by_seed"]


def test_attempts_of_another_checkpoint_under_the_same_name_are_refused_on_maps(checkpoint, tmp_path):
    run_eval(PPOMapContender(checkpoint), SHORT, tmp_path, stop_after=1)

    retrained = train_tiny(tmp_path / "retrained", map="E1M1")  # same setting and seed, so the same name
    with pytest.raises(SystemExit, match="another checkpoint"):
        run_eval(PPOMapContender(retrained), SHORT, tmp_path)
