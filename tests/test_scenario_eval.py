"""The Eval Suite measures Contenders on Scenarios the way it does on Maps.

Scenarios ship inside the vizdoom package, so these tests need no WAD.
"""

from dataclasses import replace

import pytest

from doom_player.contenders.random import RandomScenarioContender
from doom_player.eval import attempts_path, read_records
from doom_player.scenario_eval import ScenarioSpec, run_scenario_eval

SHORT = ScenarioSpec(name="test-basic", scenario="basic", seeds=(0, 1, 2))


def test_random_contender_scores_a_scenario_over_fixed_seeds(tmp_path):
    row = run_scenario_eval(RandomScenarioContender(), SHORT, tmp_path)
    assert row["contender"] == "random"
    assert row["spec"] == "test-basic"
    assert row["scenario"] == "basic"
    assert row["seeds"] == [0, 1, 2]
    assert row["observation_class"] == "human-equivalent"
    # Basic pays +101 for the kill, -1 per tic alive and -5 per missed shot; it
    # ends after 300 tics and starts with 50 bullets, so a total lies in [-550, 101].
    assert len(row["reward_by_seed"]) == 3
    assert all(-550 <= r <= 101 for r in row["reward_by_seed"])
    assert row["wall_clock_s_per_attempt"] > 0


METRICS = ("reward_mean", "reward_by_seed", "seeds", "observation_class")


def test_same_seeds_give_identical_scenario_metrics(tmp_path):
    first = run_scenario_eval(RandomScenarioContender(), SHORT, tmp_path / "a")
    second = run_scenario_eval(RandomScenarioContender(), SHORT, tmp_path / "b")
    assert {k: first[k] for k in METRICS} == {k: second[k] for k in METRICS}


def test_stopped_scenario_evaluation_resumes_without_repeats(tmp_path):
    assert run_scenario_eval(RandomScenarioContender(), SHORT, tmp_path, stop_after=2) is None
    path = attempts_path(tmp_path, "random", SHORT)
    assert [r["seed"] for r in read_records(path)] == [0, 1]

    row = run_scenario_eval(RandomScenarioContender(), SHORT, tmp_path)
    assert [r["seed"] for r in read_records(path)] == [0, 1, 2]

    uninterrupted = run_scenario_eval(RandomScenarioContender(), SHORT, tmp_path / "fresh")
    assert row["reward_by_seed"] == uninterrupted["reward_by_seed"]


def test_changed_scenario_rules_under_the_same_name_are_refused(tmp_path):
    run_scenario_eval(RandomScenarioContender(), SHORT, tmp_path, stop_after=1)
    other = replace(SHORT, scenario="defend-center")  # same name, different rules
    with pytest.raises(SystemExit, match="new name"):
        run_scenario_eval(RandomScenarioContender(), other, tmp_path)
