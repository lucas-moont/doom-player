"""The Eval Suite repeats itself exactly and resumes without repeating Attempts."""

import json
from dataclasses import replace

import pytest

from doom_player.contenders import RandomContender
from doom_player.eval import EvalSpec, attempts_path, read_records, run_eval, write_row
from doom_player.paths import WAD_PATH

pytestmark = pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad")

SHORT = EvalSpec(name="test-short", map="E1M1", difficulty=3, seeds=(0, 1, 2), tic_limit=280)
METRICS = ("clear_rate", "progress_mean", "progress_by_seed", "seeds", "observation_class")


def test_same_seeds_give_identical_metrics(tmp_path):
    first = run_eval(RandomContender(), SHORT, tmp_path / "a")
    second = run_eval(RandomContender(), SHORT, tmp_path / "b")
    assert {k: first[k] for k in METRICS} == {k: second[k] for k in METRICS}


def test_stopped_evaluation_resumes_without_repeats(tmp_path):
    assert run_eval(RandomContender(), SHORT, tmp_path, stop_after=2) is None
    path = attempts_path(tmp_path, "random", SHORT)
    assert [r["seed"] for r in read_records(path)] == [0, 1]

    row = run_eval(RandomContender(), SHORT, tmp_path)
    assert [r["seed"] for r in read_records(path)] == [0, 1, 2]

    uninterrupted = run_eval(RandomContender(), SHORT, tmp_path / "fresh")
    assert row["progress_by_seed"] == uninterrupted["progress_by_seed"]


def test_scoreboard_keeps_one_row_per_contender_and_spec(tmp_path):
    row = run_eval(RandomContender(), SHORT, tmp_path)
    write_row(row, tmp_path)
    write_row(row, tmp_path)
    write_row({**row, "contender": "other"}, tmp_path)
    rows = read_records(tmp_path / "scoreboard.jsonl")
    assert sorted(r["contender"] for r in rows) == ["other", "random"]


def test_changed_rules_under_the_same_name_are_refused(tmp_path):
    run_eval(RandomContender(), SHORT, tmp_path, stop_after=1)
    longer = replace(SHORT, tic_limit=560)  # same name, different rules
    with pytest.raises(SystemExit, match="new name"):
        run_eval(RandomContender(), longer, tmp_path)


def test_a_changed_progress_rule_under_the_same_name_is_refused(tmp_path):
    run_eval(RandomContender(), SHORT, tmp_path, stop_after=1)
    keyed = replace(SHORT, progress_rule="keyed")  # same name, Progress measured another way
    with pytest.raises(SystemExit, match="new name"):
        run_eval(RandomContender(), keyed, tmp_path)


def test_records_from_before_progress_rules_count_as_doors_open(tmp_path):
    run_eval(RandomContender(), SHORT, tmp_path, stop_after=1)
    path = attempts_path(tmp_path, "random", SHORT)
    old = [{k: v for k, v in r.items() if k not in ("progress_rule", "keys_held")} for r in read_records(path)]
    path.write_text("".join(json.dumps(r) + "\n" for r in old))
    assert run_eval(RandomContender(), SHORT, tmp_path) is not None
