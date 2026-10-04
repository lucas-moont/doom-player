"""A trained PPO policy enters the Eval Suite as a Contender, with its training cost."""

import pytest
from conftest import train_tiny

from doom_player.contenders.ppo import PPOContender
from doom_player.scenario_eval import ScenarioSpec, run_scenario_eval

pytestmark = pytest.mark.slow

SHORT = ScenarioSpec(name="test-basic", scenario="basic", seeds=(0, 1, 2))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return train_tiny(tmp_path_factory.mktemp("train"), scenario="basic", total_steps=128, n_envs=2)


def test_ppo_contender_plays_a_scenario_from_a_checkpoint(checkpoint, tmp_path):
    row = run_scenario_eval(PPOContender(checkpoint), SHORT, tmp_path / "a")
    # Named after its training seed, so policies trained with other seeds get rows of their own.
    assert row["contender"] == "ppo-seed0"
    assert row["seeds"] == [0, 1, 2]
    assert row["training_steps"] >= 128
    assert row["training_wall_clock_s"] > 0

    # The policy picks its most likely action, so the same seeds repeat exactly.
    again = run_scenario_eval(PPOContender(checkpoint), SHORT, tmp_path / "b")
    assert again["reward_by_seed"] == row["reward_by_seed"]


def test_attempts_of_another_checkpoint_under_the_same_name_are_refused(checkpoint, tmp_path):
    run_scenario_eval(PPOContender(checkpoint), SHORT, tmp_path, stop_after=1)

    # Retrain with the same training seed: same Contender name, different policy.
    retrained = train_tiny(tmp_path / "retrained", scenario="basic")
    with pytest.raises(SystemExit, match="another checkpoint"):
        run_scenario_eval(PPOContender(retrained), SHORT, tmp_path)
