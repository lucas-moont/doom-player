"""AttemptSession: fixed rules, Human-equivalent Observations, repeatable records.

These need the purchased WAD and are skipped without it.
"""

from dataclasses import fields

import pytest

from doom_player.contenders import RandomContender
from doom_player.paths import WAD_PATH
from doom_player.session import HUD_VARIABLES, AttemptSession, Observation

pytestmark = pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad")


def play(seed: int, tic_limit: int) -> dict:
    contender = RandomContender()
    session = AttemptSession(contender.name, seed=seed, tic_limit=tic_limit)
    contender.reset(seed, session.buttons)
    try:
        while not session.finished:
            session.act(contender.act(session.observe()), contender.tics_per_action)
    finally:
        session.close()
    record = session.record.to_dict()
    record.pop("wall_clock_s")
    return record


def test_observation_holds_human_equivalent_fields_only():
    session = AttemptSession("probe", tic_limit=35)
    try:
        obs = session.observe()
    finally:
        session.close()
    assert [f.name for f in fields(Observation)] == ["screen", "automap", "hud", "tics_left"]
    assert obs.screen.shape == obs.automap.shape == (240, 320, 3)
    assert set(obs.hud) == set(HUD_VARIABLES)
    # E1M1 pistol start, as the status bar shows it
    assert obs.hud["HEALTH"] == 100
    assert obs.hud["SELECTED_WEAPON"] == 2
    assert obs.hud["BULLETS"] == obs.hud["SELECTED_WEAPON_AMMO"] == 50
    assert obs.tics_left == 35


def test_same_seed_same_record():
    assert play(seed=1, tic_limit=700) == play(seed=1, tic_limit=700)


def test_tic_limit_truncates():
    record = play(seed=0, tic_limit=140)
    assert record["truncated"] and not record["cleared"]
    assert record["tics"] == 140
    assert record["actions"] == 140 // RandomContender.tics_per_action


def test_rejects_bad_actions():
    session = AttemptSession("probe", tic_limit=35)
    try:
        with pytest.raises(ValueError):
            session.press(["FLY"], 4)
        with pytest.raises(ValueError):
            session.press(["ATTACK"], 36)
    finally:
        session.close()


def test_one_tic_steps_play_the_same_game_as_one_long_step():
    # Smooth video steps one tic at a time; the game must not notice.
    def record(on_frame):
        contender = RandomContender()
        session = AttemptSession(contender.name, seed=2, tic_limit=1400)
        contender.reset(2, session.buttons)
        try:
            while not session.finished:
                session.act(contender.act(session.observe()), 8, on_frame)
        finally:
            session.close()
        r = session.record.to_dict()
        r.pop("wall_clock_s")
        return r

    frames = []
    assert record(frames.append) == record(None)
    assert len(frames) >= 1400 - 8
